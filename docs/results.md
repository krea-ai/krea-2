# SFT versus DRaFT-K results

We evaluated four characters with eight held-out prompts per character. The
prompts were fixed before training and were excluded from the DRaFT-K training
prompt set. Both branches shared SFT steps 1–500; one branch then received 50
DRaFT-K updates while the other restored the exact optimizer, RNG, cache, and
sampler state and continued SFT through step 1000.

| Training branch | Mean identity similarity | Face detection |
| --- | ---: | ---: |
| SFT 500 + DRaFT-K 50 | 0.471 | 93.75% |
| SFT 1000 | 0.212 | 100.00% |

The hybrid branch produced 2.22× the mean identity similarity. It used 10.1%
more optimization time on average across the four runs; the per-character
overhead ranged from 6.7% to 13.7%. This supports a conclusion of substantially
better identity similarity at near-matched training time—not exactly equal
compute. Face detection was slightly lower, so similarity and detection should
both remain visible when assessing future variants.

The machine-readable aggregate and anonymous per-run measurements are in
[`results/draft_vs_sft.json`](results/draft_vs_sft.json). Evaluation images are
not distributed.
