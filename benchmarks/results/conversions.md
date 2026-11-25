**Notes.** (1) Regenerated artifacts: files rebuilt from the GGUF carry a header that lists `p1_p`…`p12_p`, 22 bytes longer than the Phase 0 header (eight allowed densities). The same `p1_p` is 2,562,220,548 bytes in the Phase 0 artifact and 2,562,220,570 bytes when rebuilt; the payload and its SHA-256 are identical. (2) Resolution of the quality suite: perplexity over **160 tokens** (`benchmarks/corpus.txt`, 798 bytes) and QA over **10 questions** (`benchmarks/qa.jsonl`), so 80.0% = **8/10** and one answer moves QA by **10 pp**.

| Density | File bytes | Payload bytes | Tensors | Payload SHA-256 | Verified | Seconds |
|---|---:|---:|---:|---|---:|---:|
| p1_p | 2562220570 | 2560980992 | 195 | same | yes | 16 |
| p2_p | 2562236261 | 2560980992 | 195 | same | yes | 13 |
| p3_p | 2562251808 | 2560980992 | 195 | same | yes | 11 |
| p4_p | 2562267542 | 2560980992 | 195 | same | yes | 11 |
| p5_p | 2562283024 | 2560980992 | 195 | same | yes | 10 |
| p6_p | 2562298476 | 2560980992 | 195 | same | yes | 10 |
| p7_p | 2562313731 | 2560980992 | 195 | same | yes | 10 |
| p8_p | 2562329323 | 2560980992 | 195 | same | yes | 11 |
