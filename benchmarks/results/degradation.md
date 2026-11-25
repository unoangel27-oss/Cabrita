**Notes.** (1) Regenerated artifacts: files rebuilt from the GGUF carry a header that lists `p1_p`…`p12_p`, 22 bytes longer than the Phase 0 header (eight allowed densities). The same `p1_p` is 2,562,220,548 bytes in the Phase 0 artifact and 2,562,220,570 bytes when rebuilt; the payload and its SHA-256 are identical. (2) Resolution of the quality suite: perplexity over **160 tokens** (`benchmarks/corpus.txt`, 798 bytes) and QA over **10 questions** (`benchmarks/qa.jsonl`), so 80.0% = **8/10** and one answer moves QA by **10 pp**.

| Density | Subparameters | Perplexity | Δ perplexity | QA | Δ QA |
|---|---:|---:|---:|---:|---:|
| p1_p | 195 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p2_p | 390 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p3_p | 585 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p4_p | 780 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p5_p | 975 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p6_p | 1170 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p7_p | 1365 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p8_p | 1560 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p9_p | 1755 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p10_p | 1950 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p11_p | 2145 | 20.870 | +0.0% | 80.0% | +0.0 pp |
| p12_p | 2340 | 20.870 | +0.0% | 80.0% | +0.0 pp |
