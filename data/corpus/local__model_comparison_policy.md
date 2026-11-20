# Model Selection Policy

## Required evidence

A model is eligible for a pilot only after the team records its upstream repository, exact weight revision, license, SHA256, peak memory, p95 latency, answer quality, and failure cases on the fixed evaluation set.

## Decision rule

Prefer the smallest model that passes the task's quality floor. A larger or higher-precision model is justified only when the measured quality gain exceeds its additional memory or latency cost. No model is selected from vendor benchmark claims alone.

