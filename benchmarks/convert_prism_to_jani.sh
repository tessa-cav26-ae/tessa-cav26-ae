#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# storm-conv's JANI exporter does not handle PRISM's `exactlyOneOf(...)` predicate.
# Herman's `label "stable"` uses it; rewrite to an equivalent sum-equals-1 form
# in a temp copy before conversion. Source files stay untouched.
rewrite_exactly_one_of() {
    python3 -c '
import re, sys
src = open(sys.argv[1]).read()
def expand(m):
    args = [a.strip() for a in m.group(1).split(",")]
    return "(" + " + ".join(f"({a}?1:0)" for a in args) + ") = 1"
open(sys.argv[2], "w").write(re.sub(r"exactlyOneOf\(([^)]*)\)", expand, src))
' "$1" "$2"
}

# storm-conv emits PRISM labels as transient bool globals defined via
# `transient-values` on each automaton location, but leaves the JANI top-level
# `properties` array empty. Tessa's compiler looks up `--property NAME` in
# `properties`, so we lift each label's expression into a properties entry.
inject_label_properties() {
    python3 -c '
import json, sys
path = sys.argv[1]
model = json.loads(open(path).read())
label_names = {v["name"] for v in model.get("variables", [])
               if v.get("transient") and v.get("type") == "bool"}
expressions = {}
for automaton in model.get("automata", []):
    for location in automaton.get("locations", []):
        for tv in location.get("transient-values", []):
            if tv["ref"] in label_names and tv["ref"] not in expressions:
                expressions[tv["ref"]] = tv["value"]
existing = {p["name"] for p in model.get("properties", [])}
properties = model.setdefault("properties", [])
for name in label_names:
    if name in expressions and name not in existing:
        properties.append({"name": name, "expression": expressions[name]})
open(path, "w").write(json.dumps(model, indent=2))
' "$1"
}

for suite in herman meeting weather_factory parqueues; do
    log="$suite/convert.log"
    : > "$log"
    for src in "$suite"/*.prism; do
        dst="${src%.prism}.jani"
        echo "=> $src" | tee -a "$log"
        if [ "$suite" = "herman" ]; then
            tmp=$(mktemp --suffix=.prism)
            rewrite_exactly_one_of "$src" "$tmp"
            storm-conv --prism "$tmp" --tojani "$dst" >>"$log" 2>&1
            rm -f "$tmp"
        else
            storm-conv --prism "$src" --tojani "$dst" >>"$log" 2>&1
        fi
        inject_label_properties "$dst"
    done
done
