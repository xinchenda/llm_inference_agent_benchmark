# Evaluation Protocol

## Fixed conditions

Answer-quality runs use temperature 0, seed 42, a maximum of 180 output tokens, and the same 100 questions. RAG runs use the same corpus snapshot and chunking parameters. Retrieval Recall@5 is measured at document level.

## Human review

Twelve sampled answers are manually reviewed for factual support, citation correctness, and whether an unsupported claim appears. Model-based judging is optional and is never the only label source.

## Performance runs

Performance runs discard two warm-up requests before measurement. Prompt length and output length are fixed for each comparison. The local incremental service cost is recorded as zero US dollars because every component is open source and self-hosted; hardware purchase and electricity are outside that figure.

