**Notes.** (1) Regenerated artifacts: files rebuilt from the GGUF carry a header that lists `p1_p`…`p12_p`, 22 bytes longer than the Phase 0 header (eight allowed densities). The same `p1_p` is 2,562,220,548 bytes in the Phase 0 artifact and 2,562,220,570 bytes when rebuilt; the payload and its SHA-256 are identical. (2) Resolution of the quality suite: perplexity over **160 tokens** (`benchmarks/corpus.txt`, 798 bytes) and QA over **10 questions** (`benchmarks/qa.jsonl`), so 80.0% = **8/10** and one answer moves QA by **10 pp**.

| Model | q* | p | tok/s | TTFT ms | Latency ms | RSS MB | Perplexity | QA |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `olmoe-q2_k-p1_p.ukj` | q2_k | p1_p | 2.72 | 8301 | 14207 | 2124.3 | 20.870 | 80.0% |
| `olmoe-q2_k-p3_p.ukj` | q2_k | p3_p | 2.67 | 9900 | 15918 | 2124.9 | 20.870 | 80.0% |
| `olmoe-q2_k-p5_p.ukj` | q2_k | p5_p | 2.93 | 7704 | 13185 | 2123.8 | 20.870 | 80.0% |
| `olmoe-q2_k-p8_p.ukj` | q2_k | p8_p | 2.61 | 8238 | 14362 | 2124.4 | 20.870 | 80.0% |
