# Local Service Security Boundary

## Authentication

The reference deployment accepts a static bearer token from an environment variable. This is suitable for a local pilot only. Production deployment must integrate the company identity provider, rotate credentials, and avoid storing secrets in source control.

## Data retention

Request logs keep tool names, durations, status codes, and citation identifiers. Raw questions and answers are disabled by default in the production profile. Evaluation mode may retain them because the questions are synthetic.

## Tool access

Report generation writes only under the configured artifact directory. Shell execution, arbitrary URL fetching, and unrestricted filesystem access are not exposed as agent tools.

