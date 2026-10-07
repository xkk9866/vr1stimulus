import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
d = json.loads((R / "incremental.json").read_text())
NICE = {"y_val": "V-2", "y_aro": "A-2", "y_val3": "V-3", "y_aro3": "A-3", "y5": "5-cls"}
want = ["prior only", "prior+phys (lr)", "prior+beh (lr)", "prior+all (lr)", "prior+all (hgb)",
        "phys only (lr)", "beh only (lr)", "all only (lr)"]

print("%-18s %s" % ("row", "  ".join("%-22s" % NICE[t] for t in NICE)))
for w in want:
    cells = []
    for t in NICE:
        r = d[t]["rows"].get(w)
        cells.append("%6.2f/%.4f/%.4f" % (r["acc"], r["logloss"], r["brier"]) if r else "-")
    print("%-18s %s" % (w, "  ".join("%-22s" % c for c in cells)))

print("\n-- conditional permutation (prior+all, lr), log-loss --")
for t in NICE:
    p = d[t]["perm"]
    print("%-6s obs=%.4f null=%.4f+-%.4f q05=%.4f p_lower=%.3f n=%d"
          % (NICE[t], p["observed_logloss"], p["null_mean"], p["null_sd"], p["null_q05"],
             p["p_lower"], p["n_perm"]))

print("\n-- paired tests vs prior only (prior+all lr) --")
for t in NICE:
    for m in ("logloss", "brier", "acc"):
        k = "prior+all (lr) :: " + m
        v = d[t]["tests"][k]
        print("%-6s %-8s delta=%+.4f p=%.2e" % (NICE[t], m, v["delta"], v["p"]))
