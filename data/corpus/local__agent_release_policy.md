# Atlas Local Agent Release Policy

## Version 2.1 context limit

Release 2.1 of the Atlas Local Agent uses a default context window of 8192 tokens. This limit applies to retrieved evidence plus the user question. The system reserves 512 tokens for the answer and 256 tokens for tool metadata.

## Deployment scope

Release 2.1 is an evaluation release. It is approved only for technical documents owned by the engineering team and synthetic evaluation questions. It is not approved for customer personal data.

## Failure behavior

If a tool times out twice, the agent stops calling tools and returns a partial answer with an explicit failure message. Repeated tool requests with identical arguments are rejected after the first call.

