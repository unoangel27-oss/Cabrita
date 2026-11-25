**Notes.** (1) Regenerated artifacts: these `.ukj` files were rebuilt from the GGUF with a header listing `p1_p`…`p12_p`, 22 bytes longer than the Phase 0 header (eight allowed densities), which is why `p1_p` has 2,562,220,570 bytes here and 2,562,220,548 in `q_star.md`; the payload and its SHA-256 are identical. (2) Resolution of the quality suite: perplexity over **160 tokens** (`benchmarks/corpus.txt`, 798 bytes), QA over **10 questions** (80.0% = 8/10; one answer = 10 pp).

| Density | File bytes | Payload bytes | Tensors | Payload SHA-256 | Verified | Seconds |
|---|---:|---:|---:|---|---:|---:|
| p1_p | 2562220570 | 2560980992 | 195 | same | yes | 52 |
| p2_p | 2562236261 | 2560980992 | 195 | same | yes | 51 |
| p3_p | 2562251808 | 2560980992 | 195 | same | yes | 33 |
| p4_p | 2562267542 | 2560980992 | 195 | same | yes | 33 |
| p5_p | 2562283024 | 2560980992 | 195 | same | yes | 29 |
| p6_p | 2562298476 | 2560980992 | 195 | same | yes | 30 |
| p7_p | 2562313731 | 2560980992 | 195 | same | yes | 29 |
| p8_p | 2562329323 | 2560980992 | 195 | same | yes | 30 |
