# System One Auto — brand and ecosystem references

The cover is an ImageGen illustration based on the vLLM Semantic Router
[v0.4 Hermes release cover](../../vllm/2026-09-24-v0.4-hermes-release/hero.png).
Its primary mark is the project's existing `vllm-sr-logo.light.png`; the article's
responsive figures use that original asset directly. The design uses a white
canvas, ink-navy text, electric blue routing and amber escalation accents.

The cover focuses exclusively on decision models. A separate responsive
comparison inside the article shows LLM and System One interfaces as parallel
uses of the routing framework, not a serial LLM-to-decision pipeline.
The measured cascade is only Decision 2.0 Kai → Vega.
An external provider must satisfy the native API contract before it can serve
as a compatible backend; a logo does not establish adapter conformance.

## Provider identity sources

| Example | Primary model or API source | Logo reference |
| --- | --- | --- |
| Decision 2.0 | [vLLM-SR model collection](https://huggingface.co/collections/vllm-sr/decision-20) | Original vLLM Semantic Router project mark |
| Cloudflare Clef | [Open checkpoint](https://huggingface.co/Cloudflare/clef) | [Official Cloudflare organization](https://huggingface.co/Cloudflare) avatar |
| Perplexity Decider | [Open checkpoint](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) | [Official Perplexity organization](https://huggingface.co/perplexity-ai) avatar |
| Laya | [Project repository](https://github.com/NandhaKishorM/laya) | Original `assets/logo-lockup.png` in that repository |
| Fastino GLiDE | [Official launch](https://fastino.ai/blog/introducing-glide-the-first-thinking-decision-model) | [Official Fastino organization](https://huggingface.co/fastino) avatar |
| TypeSafe Jev | [Official API quickstart](https://docs.typesafe.ai/introduction/quickstart) | [Official TypeSafe organization](https://github.com/typesafe-ai) avatar |

GLiDE is shown as hosted: as checked on 2026-10-10, its public weights were
announced but not released. The no-thinking label comes from the
[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index);
it is not a verified public API selector. Clef can also be served through a
hosted API. Open weights and hosting are therefore independent properties.

The article comparison uses locally stored, unmodified provider marks in
`../providers/`. Qwen, DeepSeek, OpenAI and Anthropic SVGs were rendered from
the original components in [`@lobehub/icons` 5.16.0](https://github.com/lobehub/lobe-icons),
with their shapes unchanged. The accompanying [MIT license](../providers/LICENSE-lobehub.txt)
retains LobeHub’s copyright and permission notice. The five Decision provider
PNG assets come from the identity sources listed above. Decision 2.0 uses
the project’s existing full logo directly. No remote logo service is needed
when rendering the article. Third-party brands remain the property of their
respective owners.

## Illustration brief

Create a white, blue and amber research launch banner in the Hermes identity.
Place the complete vLLM Semantic Router mark and System One Auto title on the
left, with the slogan “Route language. Route decisions.” On the right, show
an intelligent Auto selection hub with six uncluttered provider destinations.
Highlight one selected path in amber, with quieter paths for other candidates;
keep LLMs and their protocols in the separate comparison inside the article.
Preserve recognizable provider identities, generous spacing and readable
labels. Do not add benchmark scores
or imply a provider partnership. The results and cascade illustrations in the
article use the same palette, original project mark and native text; their
numbers are computed from the unchanged frozen evidence.
