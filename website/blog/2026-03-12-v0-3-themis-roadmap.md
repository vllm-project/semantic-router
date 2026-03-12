---
slug: v0-3-themis-roadmap
title: "v0.3 Themis Roadmap: Stability at Scale"
authors: [Xunzhuo, rootfs]
tags: [roadmap, themis, v0.3, stability, semantic-router]
image: /img/blog/themis.png
---

v0.3, codename **Themis**, is our production-readiness release for Semantic Router. The theme is simple: **Stability at Scale**. After Athena expanded the system brain, Themis is the release where we make that intelligence dependable across real environments, clearer to operate, and safer to ship into production.

This roadmap is not just about adding more capability. It is about making the full system coherent: one stable contract across Docker and Kubernetes, one cleaner deployment path, one real version story for images and packages, stronger performance validation on both NVIDIA and AMD, and a research track that directly improves the product instead of sitting outside it.

![img](/img/blog/themis.png)

<!-- truncate -->

## Why Themis

Themis is the Greek figure of order, rules, and judgment. That fits this release better than a speed-oriented or purely routing-oriented codename. Themis is where Semantic Router starts acting less like a promising set of powerful building blocks and more like a platform with stable contracts, repeatable operations, and enforceable guardrails.

The current v0.3 milestone reflects that shift. It includes the new workstreams opened specifically for Themis, but it also folds in existing issues around protocol compatibility, session affinity, memory hardening, dashboard state, observability, security, and API standardization. This release is not a narrow feature sprint. It is a system-shaping release.

## 1. Stable API, config, and deployment contracts

The highest-priority theme in Themis is eliminating contract drift across environments. Today, router behavior, Helm-facing config, dashboard flows, and the Python CLI still expose differences that create friction for operators. Themis is where we narrow those seams.

![img](/img/blog/api.png)

At the center of that work is a canonical API and config contract across router, CLI, dashboard, and Kubernetes. The goal is simple: after this release, a user should not have to mentally maintain one configuration model for local Docker workflows and another for Kubernetes deployment. This is the core of [#1505](https://github.com/vllm-project/semantic-router/issues/1505).

That contract work also includes the deployment entry point itself. The `vllm-sr` CLI should become the normal path for standing up both Docker and Kubernetes environments, instead of being treated as a local-only helper while Helm and other deployment paths evolve separately. That is the focus of [#1507](https://github.com/vllm-project/semantic-router/issues/1507).

We also want the runtime topology to become easier to reason about. Themis moves toward a router-focused `vllm-sr` image, with external services such as dashboard, Envoy, and persistence components split out more cleanly. This keeps the main runtime narrower and makes upgrades, debugging, and composition less fragile. That work is tracked in [#1508](https://github.com/vllm-project/semantic-router/issues/1508).

This same contract cleanup extends to protocol compatibility. Themis already includes work to support first-class OpenAI and Anthropic API entry points, align API definitions with official SDKs, and reduce homegrown JSON struct drift across the codebase. Those concerns now live in [#1517](https://github.com/vllm-project/semantic-router/issues/1517), [#1404](https://github.com/vllm-project/semantic-router/issues/1404), and [#1217](https://github.com/vllm-project/semantic-router/issues/1217).

## 2. Stable versions, upgrades, and production operations

Themis is also the release where we stop treating `latest` as a deployment strategy. Production users need to know what they are running, how they upgrade, how they roll back, and what guarantees exist between images, packages, and charts. That operational maturity is the purpose of [#1506](https://github.com/vllm-project/semantic-router/issues/1506).

This means introducing explicit version channels such as nightly and tagged releases, carrying versioned images and packages through the stack, and documenting a full upgrade and rollback flow instead of assuming rebuild-and-redeploy. A stable version story is part of stability at scale, not an afterthought to it.

Operational stability also depends on where state lives. Dashboard behavior today still depends too heavily on in-memory state for workflows that should survive restarts, scale-outs, and multi-user operation. Themis moves those operationally important pieces into a database-backed control plane, tracked in [#1509](https://github.com/vllm-project/semantic-router/issues/1509).

As milestone triage has progressed, this operations theme has also pulled in related issues around docs and environment correctness, especially where deployment docs, API expectations, and runtime behavior need to converge before we can credibly call the surface stable.

## 3. Performance at scale on real hardware

Themis is not only about control-plane cleanup. It is also about making sure the router and its supporting model stack behave well under real load, across real backends, on real platforms. That is the purpose of [#1510](https://github.com/vllm-project/semantic-router/issues/1510).

We want broader large-scale regression coverage across Candle, ONNX, and related runtime paths, with repeatable performance baselines for both NVIDIA and AMD. This matters because Semantic Router is increasingly expected to sit in front of more heterogeneous workloads: more model families, more protocol paths, more multi-component deployments, and more memory-heavy workflows.

This performance theme is also tied to product credibility. If we claim the platform is ready for production routing, then we need more than point optimizations. We need performance tests that survive release-to-release, platform-to-platform, and topology-to-topology changes.

That same bar increasingly applies to higher-level agent surfaces such as ClawOS. If model routing, memory, and tool execution are going to be orchestrated in room-based agent workflows, then performance and runtime visibility have to scale there too.

![img](/img/blog/research.png)

## 4. Research that feeds the product

Themis still includes research-heavy work, and it should. But the research in this milestone is there because it improves the production system, not because we are parking speculative ideas in the roadmap.

The first track is **NL-to-DSL authoring** in the dashboard, tracked in [#1511](https://github.com/vllm-project/semantic-router/issues/1511). The goal is to let users express routing intent in natural language and generate a usable DSL scaffold instead of forcing every workflow through fully manual route authoring.

The second track is a **feedback loop for generated DSL**, tracked in [#1512](https://github.com/vllm-project/semantic-router/issues/1512). Generated routing logic becomes much more useful when it can learn from real request history, observed routing outcomes, and user feedback, instead of acting like a one-shot assistant.

The third track is **multi-turn session affinity**, tracked in [#1513](https://github.com/vllm-project/semantic-router/issues/1513) and reinforced by the older conversation-stability issue [#1439](https://github.com/vllm-project/semantic-router/issues/1439). This is one of the clearest examples of research feeding production directly: without stable session affinity, routed multi-turn conversations can bounce between models and degrade user experience even if each single-turn decision looks correct.

There is also research around **model legitimacy and selection quality**, including [#1422](https://github.com/vllm-project/semantic-router/issues/1422) and [#1514](https://github.com/vllm-project/semantic-router/issues/1514). This line of work matters because model selection is only useful in production when it is trustworthy, inspectable, and not dependent on fragile external-only components. Themis should move that work closer to something operators can actually rely on.

ClawOS does have a genuine research component here, but it is specifically the context question. [#1522](https://github.com/vllm-project/semantic-router/issues/1522) is about studying context-management patterns and OpenClaw best practices so long-running, tool-rich, room-based workflows have a clearer operating model.

In that sense, the research section of Themis is really about system intelligence: generating better routing logic, improving it continuously, keeping conversations stable across turns, and making model-selection decisions more defensible.

## 5. Hardening the current product surface

Themis also has a large body of work that is less glamorous than new intelligence features, but just as important for adoption.

Model selection needs to become more usable without external-service-only dependencies, which is the focus of [#1514](https://github.com/vllm-project/semantic-router/issues/1514). Eval workflows need to be revisited so system eval and signal eval are first-class and stable inside the dashboard, tracked in [#1515](https://github.com/vllm-project/semantic-router/issues/1515).

RAG and memory workflows also need to become more production-friendly. That includes the main hardening track in [#1516](https://github.com/vllm-project/semantic-router/issues/1516), plus milestone issues already folded in around memory evolution such as [#1293](https://github.com/vllm-project/semantic-router/issues/1293), [#1287](https://github.com/vllm-project/semantic-router/issues/1287), [#1289](https://github.com/vllm-project/semantic-router/issues/1289), [#1350](https://github.com/vllm-project/semantic-router/issues/1350), and [#1353](https://github.com/vllm-project/semantic-router/issues/1353).

ClawOS also belongs in this product-hardening bucket. [#1521](https://github.com/vllm-project/semantic-router/issues/1521) is not a research item; it is about making collaborative rooms work as a first-class product surface through Matrix-style full WebSocket communication between rooms and participants.

This is also where protocol polish and dashboard usability meet. The goal is not just to have more capability on paper, but to make those capabilities easier to operate in the dashboard, easier to expose consistently through APIs, and easier to validate end to end.

![img](/img/blog/clawos.png)

## 6. Security and quality closure

Themis is also where we close the operational gaps that would block serious production adoption. That starts with the main security and RBAC workstream in [#1518](https://github.com/vllm-project/semantic-router/issues/1518), but it is reinforced by several already-folded issues that expose concrete weaknesses in the current surface.

That includes security issues such as [#1443](https://github.com/vllm-project/semantic-router/issues/1443), [#1445](https://github.com/vllm-project/semantic-router/issues/1445), [#1447](https://github.com/vllm-project/semantic-router/issues/1447), [#1448](https://github.com/vllm-project/semantic-router/issues/1448), [#1452](https://github.com/vllm-project/semantic-router/issues/1452), [#1454](https://github.com/vllm-project/semantic-router/issues/1454), [#1456](https://github.com/vllm-project/semantic-router/issues/1456), and [#1458](https://github.com/vllm-project/semantic-router/issues/1458). These are exactly the kinds of issues that justify the Themis theme: if the platform is going to be production-ready, the security model has to be explicit and closed-loop.

Quality also means broader E2E coverage. The main expansion item is [#1519](https://github.com/vllm-project/semantic-router/issues/1519), but related milestone issues such as [#1295](https://github.com/vllm-project/semantic-router/issues/1295), [#1432](https://github.com/vllm-project/semantic-router/issues/1432), [#1501](https://github.com/vllm-project/semantic-router/issues/1501), and [#1083](https://github.com/vllm-project/semantic-router/issues/1083) show the same pattern: production hardening requires better system-level tests, better observability, and fewer hidden assumptions.

That broader observability push now also includes ClawOS-specific visibility into model and tool behavior through [#1523](https://github.com/vllm-project/semantic-router/issues/1523), so agentic workflows are not left outside the production-debugging story.

## What success looks like

If Themis is successful, Semantic Router should feel materially different to deploy and operate:

- API and config behavior should be much more consistent across Docker, Kubernetes, CLI, and dashboard workflows
- release channels, upgrades, and rollbacks should be explicit rather than implicit
- performance claims should be backed by repeatable NVIDIA and AMD validation
- research work should show up as product intelligence, especially in DSL generation, feedback loops, session affinity, ClawOS context management, and better model selection
- memory, eval, protocol compatibility, and dashboard state should look more like stable platform features than experimental edges
- security, RBAC, observability, and E2E coverage should be strong enough that production users can trust the platform boundary

Themis is therefore less about one headline feature and more about making the whole system hold together under real use.

For the active implementation tracker, see [v0.3 - Themis: Stability at Scale milestone](https://github.com/vllm-project/semantic-router/milestone/4) and [issue #1520](https://github.com/vllm-project/semantic-router/issues/1520).
