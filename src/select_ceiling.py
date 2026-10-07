"""Report the train-selected stimulus-information ceiling.

The ceiling in eq. (SIA) is a supremum over a decoder class. The identified
estimator is nested: on each leave-one-subject-out fold, pick the class
member with the highest *training* accuracy (bottleneck with T, without T,
plug-in lookup, majority) and take that member's *test* accuracy. Selecting
from test-fold numbers, whether once from means or per fold, uses the same
labels as the credit.

run_bottleneck.pack() already writes ceiling_acc / credit_selected.
This script reprints them from the stored JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "results" / "bottleneck.json"


def main():
    d = json.loads(PATH.read_text(encoding="utf-8"))
    for corpus in ("ceap", "case", "vreed"):
        if corpus not in d:
            continue
        print("#####", corpus)
        for key, r in d[corpus]["decomposition"].items():
            inf = r["credit_selected"]
            print(
                f"  {key:12s} ceil={r['ceiling_acc']:6.2f} "
                f"({r.get('ceiling_source','?')}, {r.get('ceiling_win_counts', {})})  "
                f"LOSO {r['loso']['acc']:6.2f}  credit {inf['mean']:+5.2f} "
                f"CI [{inf['ci95'][0]:+5.2f},{inf['ci95'][1]:+5.2f}]  "
                f"p={inf['wilcoxon_p']:.3f}  equiv<{inf['equiv_margin_95']:.2f}"
            )
        swap = d[corpus].get("jng_bns_swap")
        if swap:
            print("#####", corpus, "JNG/BNS swap")
            for key, r in swap.items():
                inf = r["credit_selected"]
                print(
                    f"  {key:12s} ceil={r['ceiling_acc']:6.2f}  "
                    f"LOSO {r['loso']['acc']:6.2f}  credit {inf['mean']:+5.2f} "
                    f"CI [{inf['ci95'][0]:+5.2f},{inf['ci95'][1]:+5.2f}]"
                )


if __name__ == "__main__":
    main()
