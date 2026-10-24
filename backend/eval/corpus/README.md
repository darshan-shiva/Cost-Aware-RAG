# Evaluation corpus

All documents here are **synthetic**. Northwind Logistics, the Aurora X2 / X3
batteries and Greenfield Outfitters are invented for this evaluation; any
resemblance to real organisations or products is accidental. The texts are
short on purpose so every expected answer can be checked by hand.

| File | Format | Pages | Used for |
|---|---|---|---|
| `northwind-employee-handbook.pdf` | PDF | 4 | single-fact, paraphrase, multi-page questions |
| `aurora-battery-manual.pdf` | PDF | 3 | lookalike questions (X2 vs X3), numeric facts |
| `greenfield-refund-policy.txt` | TXT | 1 | document replacement (v1 → v2) and deletion |
| `greenfield-refund-policy-v2.txt` | TXT | 1 | replacement content for the change test |

The PDFs are generated from `corpus.json` by `python -m eval.build_corpus`
(run from `backend/`), so their text is reproducible and reviewable in the JSON.
