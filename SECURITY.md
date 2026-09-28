# Security Policy

## Reporting security issues

Please report security issues privately using [the vulnerability submission form](https://github.com/vllm-project/semantic-router/security/advisories/new).

## Issue triage

Reports will be triaged by the vLLM Semantic Router maintainer team, coordinating with the [vLLM vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html) when applicable.

## Threat model

vLLM Semantic Router operates as an Envoy ExtProc filter in the critical path of LLM inference requests. Key security considerations include:

- **Request classification**: Classification models (PII, jailbreak, complexity) process untrusted user input. Malicious inputs targeting these models could affect routing decisions.
- **Routing decisions**: Incorrect routing due to adversarial inputs or configuration errors could expose sensitive model endpoints or bypass safety guardrails.
- **External dependencies**: Integration with classification models, backend LLM services, and caching layers creates trust boundaries.
- **Configuration**: YAML configuration files control routing behavior. Misconfiguration could result in security policy bypass.

Please see the [vLLM Security Guide](https://docs.vllm.ai/en/latest/usage/security.html) for broader security assumptions and recommendations when deploying LLM inference infrastructure.

Please see [PyTorch's Security Policy](https://github.com/pytorch/pytorch/blob/main/SECURITY.md) for more information on how to securely interact with models.

## Issue severity

We will determine the risk of each issue, taking into account affected versions, common defaults, deployment configurations, and attack surface. We use the following severity categories:

### CRITICAL Severity

Vulnerabilities that allow remote attackers to execute arbitrary code, take full control of the router or backend systems, or significantly compromise confidentiality, integrity, or availability without any interaction or privileges needed. Examples include remote code execution via malicious request payloads or deserialization issues. Generally those issues which are rated as CVSS ≥ 9.0.

### HIGH Severity

Serious security flaws that allow elevated impact—like RCE in specific, limited contexts, significant data leakage, or complete bypass of routing safety guardrails—but require advanced conditions or some trust. Examples include authentication bypass in dashboard endpoints, or exploitation requiring privileged network access. These issues typically have CVSS scores between 7.0 and 8.9.

### MODERATE Severity

Vulnerabilities that cause denial of service, partial disruption, or limited information disclosure, but do not allow arbitrary code execution or major data breach. Examples include resource exhaustion via crafted requests or configuration parsing bugs. These issues have a CVSS rating between 4.0 and 6.9.

### LOW Severity

Minor issues such as informational disclosures, logging errors, non-exploitable flaws, or weaknesses that require local or high-privilege access and offer negligible impact. Examples include side channel attacks or hash collisions. These issues often have CVSS scores less than 4.0.

## Fix disclosure policy

When a security report is accepted, the fix process depends on the severity:

- **CRITICAL and HIGH severity**: Fixes are developed in a private security fork and coordinated with the prenotification group before public disclosure.
- **MODERATE and LOW severity**: Fixes are developed and submitted as public pull requests. These issues do not require embargo since they do not enable arbitrary code execution or significant data breach, and public visibility accelerates community review and adoption of the fix.

The vLLM Semantic Router maintainer team reserves the right to adjust the disclosure approach on a case-by-case basis, taking into account factors such as active exploitation, unusual attack surface, or coordination requirements with vLLM or downstream vendors.

## Prenotification policy

For certain security issues of CRITICAL, HIGH, or MODERATE severity level, we may prenotify the vLLM vulnerability management team and certain organizations or vendors that ship vLLM Semantic Router. The purpose of this prenotification is to allow for a coordinated release of fixes for severe issues.

- This prenotification will be in the form of a private email notification. It may also include adding security contacts to the GitHub security advisory, typically a few days before release.

- If you wish to be added to the prenotification group, please contact the vLLM Semantic Router maintainer team and copy the members of the [vLLM vulnerability management team](https://docs.vllm.ai/en/latest/contributing/vulnerability_management.html). Each vendor contact will be analyzed on a case-by-case basis.

- Organizations and vendors who either ship or use vLLM Semantic Router are eligible to join the prenotification group if they meet at least one of the following qualifications:
  - Substantial internal deployment leveraging the upstream vLLM Semantic Router project.
  - Established internal security teams and comprehensive compliance measures.
  - Active and consistent contributions to the upstream vLLM Semantic Router project.

- We may withdraw organizations from receiving future prenotifications if they release fixes or any other information about issues before they are public. Group membership may also change based on policy refinements for who may be included.
