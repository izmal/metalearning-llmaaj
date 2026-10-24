"""Write explanation.md for the jury from the artifacts of the same adapt run.

Covers the five criteria: clear criteria, clear priorities, links to training examples,
clear boundaries (where the learned rules and the expert disagree on training data),
and consistency with the implementation (hash of the exact prompts eval uses).
"""
import hashlib
from collections import defaultdict

from data import TOPICS
from prompts import GENERAL_RULES


def _last_assistant(row):
    return next((m["content"] for m in reversed(row["conversation"]) if m["role"] == "assistant"), "")


def _short(text, limit=220):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def build(rows, candidates, rulebook, prompts_text, self_check):
    by_id = {r["item_id"]: r for r in rows}
    evidence = {c["item_id"]: c for c in candidates}
    digest = hashlib.sha256(prompts_text.encode()).hexdigest()
    checked = {r["item_id"]: r for r in self_check}

    correct = defaultdict(lambda: [0, 0])
    for item_id, result in checked.items():
        topic = by_id[item_id]["topic"]
        correct[topic][0] += result["label"] == by_id[item_id]["label"]
        correct[topic][1] += 1
    macro = sum(c / n for c, n in correct.values()) / max(1, len(correct))

    out = ["# How this judge decides — learned from the expert's labels", ""]
    out += [f"Learned automatically by `adapt` from {len(rows)} labeled training conversations "
            f"({sum(r['label'] == 'bad' for r in rows)} bad). "
            f"Agreement with the expert on this training set: macro-topic accuracy **{macro:.3f}**.", ""]

    out += ["## 1. Decision procedure and priorities", "",
            "1. The judge checks every assistant reply in the whole conversation against the rules of the conversation's topic, plus the general rule G1.",
            "2. **One clear violation of any rule makes the conversation `bad`.** Good qualities (politeness, completeness, extra detail) never offset a violation.",
            "3. If no rule is violated, the conversation is `good`. Most conversations are good; a violation must be verified (numbers recomputed from the client data) before it counts.",
            "4. Priority when several problems exist: G1 (facts contradicting the client's data) first, then the topic rules in the listed order. "
            "Order reflects how often and how clearly the expert penalized the problem in training; it only affects which violation is reported, not the label.",
            "5. Style issues not covered by a rule are ignored.", ""]

    out += ["## 2. Rules and the training examples behind them", ""]
    for rule in GENERAL_RULES:
        out += [f"**{rule['id']} (all topics).** {rule['rule']}  ", f"_Why:_ {rule['why']}", ""]
    for topic in TOPICS:
        rules = rulebook["topics"].get(topic, {}).get("rules", [])
        if not rules:
            continue
        acc = correct.get(topic)
        acc_text = f" — training agreement {acc[0]}/{acc[1]}" if acc else ""
        out += [f"### Topic `{topic}`{acc_text}", ""]
        for rule in rules:
            out += [f"**{rule['id']}.** {rule['rule']}  ", f"_Why:_ {rule.get('why', '')}", ""]
            for item_id in rule.get("examples", [])[:2]:
                row = by_id.get(item_id)
                if not row:
                    continue
                ev = evidence.get(item_id, {})
                out.append(f"- `{item_id}` — expert label **{row['label']}**. "
                           f"Problem: {_short(ev.get('evidence', ''))}")
            out.append("")
        good = next((r for r in rows if r["topic"] == topic and r["label"] == "good"
                     and checked.get(r["item_id"], {}).get("label") == "good"), None)
        if good:
            out += [f"- Contrast: `{good['item_id']}` — expert label **good**, no rule violated. "
                    f"Reply: “{_short(_last_assistant(good), 180)}”", ""]

    out += ["## 3. Boundaries: where these rules do not reproduce the expert", "",
            "Training conversations where the learned rules and the expert's label disagree. "
            "They mark the limits of the rulebook: either the expert applies a criterion the rules do not capture, "
            "or similar conversations received different labels.", ""]
    misses = [(item_id, r) for item_id, r in checked.items() if r["label"] != by_id[item_id]["label"]]
    if not misses:
        out += ["No disagreements on the training set.", ""]
    too_strict = defaultdict(int)
    too_lenient = defaultdict(int)
    for item_id, result in misses:
        if by_id[item_id]["label"] == "good":
            for rule_id in {v.get("rule") for v in result["violations"]}:
                too_strict[rule_id] += 1
        else:
            too_lenient[by_id[item_id]["topic"]] += 1
    if too_strict:
        out.append("**Rules stricter than the expert** (expert said good, rule fired): "
                   + ", ".join(f"{rule} ×{n}" for rule, n in sorted(too_strict.items(), key=lambda x: -x[1]))
                   + ". In these cases the expert accepted replies that only partly follow the rule, "
                     "so treat these rules as soft boundaries; the examples below show where.")
    if too_lenient:
        out.append("**Expert problems the rules miss** (expert said bad, no rule fired), by topic: "
                   + ", ".join(f"{topic} ×{n}" for topic, n in sorted(too_lenient.items(), key=lambda x: -x[1])) + ".")
    out.append("")
    for item_id, result in sorted(misses, key=lambda x: by_id[x[0]]["topic"]):
        row = by_id[item_id]
        if row["label"] == "bad":
            reason = evidence.get(item_id, {}).get("evidence", "")
            out.append(f"- `{item_id}` ({row['topic']}): expert **bad**, rules say good — the rules miss this problem: {_short(reason)}")
        else:
            flags = "; ".join(f"{v.get('rule')}: {_short(v.get('evidence', ''), 120)}" for v in result["violations"])
            out.append(f"- `{item_id}` ({row['topic']}): expert **good**, rules flag {flags} — the expert tolerated this here, so the rule is stricter than the expert in such cases.")
    out.append("")

    out += ["## 4. Consistency with the implementation", "",
            "- `eval` labels test conversations using only `judge_prompts.json` from this artifacts directory; the rules above are rendered into it verbatim.",
            f"- SHA-256 of `judge_prompts.json`: `{digest}`.",
            "- The label is computed in code from the judge's reported violations (`bad` if the list is non-empty), "
            "so every `bad` prediction names the rule ids above that triggered it (see `violations.jsonl` next to the predictions).",
            "- The training agreement figures in this file come from running that same judge, with the same prompts, on the training set.", ""]
    return "\n".join(out)
