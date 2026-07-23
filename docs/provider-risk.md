# Provider and Gateway Risk Policy

One Oxygen distinguishes an official provider route from a third-party gateway route. An
official route sends a request directly to the API host controlled by the provider named in the
run. A gateway route sends the request to an independent intermediary, even when its HTTP
protocol and model identifier resemble an official provider.

Api.Airforce is an experimental `gateway_direct` integration. It is operationally documented as
an OpenAI-compatible gateway, but One Oxygen has not verified the upstream model identity,
routing, retention behavior, enterprise controls, or security claims. Its documentation describes
automatic upstream routing and failover. Consequently, an Airforce response is always recorded
as `third_party_gateway_unverified`, its logical provider remains `airforce`, and it is never
relabelled as an official OpenAI, Anthropic, Google, xAI, or DeepSeek result.

## Data policy

Airforce may receive only data explicitly classified `synthetic` or `public`. Missing,
`internal`, `confidential`, and `restricted` classifications fail closed with
`data_policy_violation`. One Oxygen does not attempt to infer a safer classification from request
content and does not override the caller's classification.

Confidential financial, customer, employee, transaction, diligence, and grader data is prohibited.
Gateway prompts and requests may be logged by the gateway or its upstream routes; One Oxygen has
not received verified enterprise evidence establishing otherwise. Official-provider keys are
also prohibited: only `AIRFORCE_API_KEY` is read for this integration, and OpenAI, Anthropic,
Gemini, xAI, and DeepSeek keys must never be sent to Airforce.

Gateway results use a separate `gateway_unverified` experiment namespace and leaderboard track.
They cannot be combined silently with official-provider results.

## Key revocation

To revoke access, delete or disable the key in the Api.Airforce account/dashboard, remove
`AIRFORCE_API_KEY` from the host environment and secret manager, close active shells that may
retain it, and issue a replacement only after reviewing the affected run records. Never put the
key in task YAML, CLI arguments, batch JSONL, run state, a Docker environment, or source control.

## Production approval gate

The integration must remain experimental until all of the following are reviewed and accepted:

- an executed data-processing agreement (DPA);
- a precise retention period for prompts, responses, logs, and backups;
- enforceable deletion guarantees;
- a complete subprocessor list;
- verifiable model and upstream-route provenance;
- an independent security audit or current SOC 2 evidence;
- a documented incident-response and notification process; and
- contractual confidentiality terms suitable for financial due diligence.

This policy does not allege malicious or fraudulent behavior. It describes Api.Airforce as an
operational but unverified third-party gateway for confidential benchmarking.
