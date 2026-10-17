Benchmark over 30 random missions per cell (new map, EVE, clues and decoys each seed).

| condition | agent | location correct | status correct | found by scan | median error (cells) | mean time used | mean battery used |
|---|---|---|---|---|---|---|---|
| clean | bayes | 100% | 100% | 97% | 0.0 | 60 | 64 |
| clean | greedy | 87% | 87% | 60% | 0.0 | 155 | 134 |
| decoys | bayes | 100% | 100% | 93% | 0.0 | 69 | 73 |
| decoys | greedy | 60% | 70% | 37% | 2.5 | 177 | 146 |
| hard | bayes | 97% | 100% | 73% | 0.0 | 88 | 91 |
| hard | greedy | 20% | 63% | 17% | 5.8 | 161 | 128 |
