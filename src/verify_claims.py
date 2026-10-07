"""Check every bottleneck number asserted in the paper against the JSON.

Any mismatch is a bug in the manuscript, not a rounding preference, so the
script exits non-zero and names the offending claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
bn = json.loads((ROOT / "results" / "bottleneck.json").read_text(encoding="utf-8"))

CLAIMS = []


def dec(corpus, key):
    return bn[corpus]["decomposition"][key]


def add(label, claimed, actual, tol=0.005):
    CLAIMS.append((label, claimed, actual, tol))


cv, ca = dec("ceap", "All/V"), dec("ceap", "All/A")
add("CEAP V ceiling 68.49", 68.49, cv["ceiling_acc"])
add("CEAP V LOSO 67.24", 67.24, cv["loso"]["acc"])
add("CEAP V credit -1.25", -1.25, cv["credit_selected"]["mean"])
add("CEAP V CI lo -3.1", -3.1, cv["credit_selected"]["ci95"][0], 0.05)
add("CEAP V CI hi +0.6", 0.6, cv["credit_selected"]["ci95"][1], 0.05)
add("CEAP V equiv 2.87", 2.87, cv["credit_selected"]["equiv_margin_95"], 0.01)
add("CEAP A ceiling 64.52", 64.52, ca["ceiling_acc"])
add("CEAP A LOSO 64.35", 64.35, ca["loso"]["acc"])
add("CEAP A credit -0.17", -0.17, ca["credit_selected"]["mean"])
add("CEAP A equiv 1.28", 1.28, ca["credit_selected"]["equiv_margin_95"], 0.01)
add("CEAP A plug 61.98", 61.98, ca["sia_plug"]["acc"])
add("CEAP A majority 64.41", 64.41, ca["majority"]["acc"])
add("naive credit, arousal 2.37", 2.37,
    ca["loso"]["acc"] - ca["sia_plug"]["acc"], 0.01)

pv = dec("ceap", "Physio/V")
add("Physio V plug 50.36", 50.36, pv["sia_plug"]["acc"])
add("Physio V majority 62.33", 62.33, pv["majority"]["acc"])
add("Physio V LOSO 62.08", 62.08, pv["loso"]["acc"])
add("Physio V ceiling 62.23", 62.23, pv["ceiling_acc"])
add("Physio V credit -0.14", -0.14, pv["credit_selected"]["mean"])
add("naive credit, physio valence 11.72", 11.72,
    pv["loso"]["acc"] - pv["sia_plug"]["acc"], 0.01)

c5 = dec("ceap", "All/C5")
add("CEAP 5cls credit -4.47", -4.47, c5["credit_selected"]["mean"])
add("CEAP 5cls CI lo -6.7", -6.7, c5["credit_selected"]["ci95"][0], 0.05)
add("CEAP 5cls CI hi -2.2", -2.2, c5["credit_selected"]["ci95"][1], 0.08)

sv = dec("case", "All/V")
add("CASE V ceiling 71.16", 71.16, sv["ceiling_acc"])
add("CASE V LOSO 69.63", 69.63, sv["loso"]["acc"])
add("CASE V credit -1.53", -1.53, sv["credit_selected"]["mean"])
sa = dec("case", "All/A")
add("CASE A credit +0.77", 0.77, sa["credit_selected"]["mean"])

vv = dec("vreed", "All/V")
va = dec("vreed", "All/A")
add("VREED V ceiling 66.35", 66.35, vv["ceiling_acc"])
add("VREED V LOSO 71.79", 71.79, vv["loso"]["acc"])
add("VREED V credit +5.45", 5.45, vv["credit_selected"]["mean"])
add("VREED V CI lo +1.3", 1.3, vv["credit_selected"]["ci95"][0], 0.05)
add("VREED V CI hi +10.9", 10.9, vv["credit_selected"]["ci95"][1], 0.05)
add("VREED V vs q +1.92", 1.92, vv["loso"]["acc"] - vv["bn_q"]["acc"], 0.01)
add("VREED A ceiling 69.23", 69.23, va["ceiling_acc"])
add("VREED A credit +0.32", 0.32, va["credit_selected"]["mean"])
add("VREED clip-id 47.44", 47.44, vv["clip_id"]["acc"], 0.01)
add("VREED prior V 81.41", 81.41, vv["prior"]["acc"], 0.01)
add("VREED V vs prior -9.62", -9.62, vv["loso"]["acc"] - vv["prior"]["acc"], 0.01)

swap_v = bn["vreed"]["jng_bns_swap"]["All/V"]["credit_selected"]["mean"]
add("JNG/BNS swap V credit unchanged", 5.45, swap_v)

ddv = bn["ceap"]["doubly_disjoint"]["V"]
add("DD V acc 57.34", 57.34, ddv["acc"])
add("DD V delta -4.99", -4.99, ddv["delta"]["mean"])
add("DD V CI lo -6.74", -6.74, ddv["delta"]["ci95"][0], 0.05)
add("DD V CI hi -3.35", -3.35, ddv["delta"]["ci95"][1], 0.05)
dda = bn["ceap"]["doubly_disjoint"]["A"]
add("DD A acc 62.21", 62.21, dda["acc"])
add("DD A delta -2.20", -2.20, dda["delta"]["mean"])
add("DD C5 delta -9.40", -9.40,
    bn["ceap"]["doubly_disjoint"]["C5"]["delta"]["mean"], 0.01)
add("DD CASE V delta -2.30", -2.30,
    bn["case"]["doubly_disjoint"]["V"]["delta"]["mean"])
add("DD CASE A delta -5.42", -5.42,
    bn["case"]["doubly_disjoint"]["A"]["delta"]["mean"])

creds = [(f"{c}/{k}", d["credit_selected"]["mean"])
         for c in ("ceap", "case", "vreed")
         for k, d in bn[c]["decomposition"].items()]
n_neg = sum(1 for _, v in creds if v < 0)
add("negative in 9 of 13", 9, n_neg, 0)
add("13 rows total", 13, len(creds), 0)
add("four positive point estimates", 4, sum(1 for _, v in creds if v > 0), 0)

sig_pos = [d["credit_selected"] for c in ("ceap", "case", "vreed")
           for d in bn[c]["decomposition"].values()
           if d["credit_selected"]["mean"] > 0
           and d["credit_selected"]["ci95"][0] > 0]
add("one significantly positive row", 1, len(sig_pos), 0)
add("that row is VREED V", True,
    dec("vreed", "All/V")["credit_selected"]["ci95"][0] > 0, 0)

binary = [k for k in bn["ceap"]["decomposition"] if k.endswith(("/V", "/A"))]
worst = max(bn["ceap"]["decomposition"][k]["credit_selected"]["ci95"][1]
            for k in binary)
add("binary CEAP upper limit <= 1.3", True, worst <= 1.3, 0)
add("6 binary CEAP rows", 6, len(binary), 0)

margins = [bn["ceap"]["decomposition"][k]["credit_selected"]["equiv_margin_95"]
           for k in binary]
add("min equiv margin 0.45", 0.45, min(margins), 0.01)
add("max equiv margin 2.87", 2.87, max(margins), 0.01)

add("CEAP V credit_q +0.89", 0.89, cv["credit_q"]["mean"], 0.01)

fails = []
for label, claimed, actual, tol in CLAIMS:
    if isinstance(claimed, bool) or isinstance(actual, bool):
        ok = bool(claimed) == bool(actual)
    else:
        ok = abs(float(claimed) - float(actual)) <= tol
    if not ok:
        fails.append((label, claimed, actual, tol))

if fails:
    print(f"{len(fails)}/{len(CLAIMS)} FAILED")
    for label, claimed, actual, tol in fails:
        print(f"  {label}: claimed {claimed} actual {actual} tol {tol}")
    sys.exit(1)
print(f"{len(CLAIMS)}/{len(CLAIMS)} OK")
