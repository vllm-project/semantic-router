# TD011: Website Information Architecture and Documentation Governance Have Drifted Apart

## Status

Open

## Scope

website navigation, doc discoverability, locale parity, and doc-site governance

## Summary

The website currently lacks one stable information architecture that consistently separates user journeys, navigation structure, locale structure, and documentation maintenance rules.

At the public site level, the homepage, intro page, and sidebar do not present the same mental model of the product surface. At the repository level, the docs tree already contains content that is not reachable from the main sidebar, locale branches have started to diverge structurally, and the site-maintainer docs describe outdated config filenames and weaker governance than the current repo expects.

## Evidence

- [website/src/pages/index.tsx](../../../website/src/pages/index.tsx)
- [website/docs/intro.md](../../../website/docs/intro.md)
- [website/sidebars.ts](../../../website/sidebars.ts)
- [website/docs/installation/k8s/operator.md](../../../website/docs/installation/k8s/operator.md)
- [website/docs/installation/docker-compose.md](../../../website/docs/installation/docker-compose.md)
- [website/docs/installation/milvus.md](../../../website/docs/installation/milvus.md)
- [website/docs/tutorials/response-api/redis-storage.md](../../../website/docs/tutorials/response-api/redis-storage.md)
- [website/docs/tutorials/response-api/redis-cluster-storage.md](../../../website/docs/tutorials/response-api/redis-cluster-storage.md)
- [website/docs/tutorials/performance-tuning/modernbert-32k-performance.md](../../../website/docs/tutorials/performance-tuning/modernbert-32k-performance.md)
- [website/docs/tutorials/intelligent-route/context-routing.md](../../../website/docs/tutorials/intelligent-route/context-routing.md)
- [website/docs/tutorials/intelligent-route/complexity-routing.md](../../../website/docs/tutorials/intelligent-route/complexity-routing.md)
- [website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cookbook/classifier-tuning.md](../../../website/i18n/zh-Hans/docusaurus-plugin-content-docs/current/cookbook/classifier-tuning.md)
- [website/scripts/check-translation-sync.sh](../../../website/scripts/check-translation-sync.sh)
- [website/docusaurus.config.ts](../../../website/docusaurus.config.ts)
- [website/README.md](../../../website/README.md)

## Why It Matters

- Users are not being cleanly routed by task or audience. Local CLI users, platform deployers, API consumers, researchers, and contributors currently meet overlapping top-level navigation instead of explicit paths.
- The sidebar is no longer a complete inventory of public content. Reachability drift means some pages exist in the repo but are effectively hidden from normal site navigation.
- Locale structure has started to diverge from English structure in ways that are bigger than translation lag alone, which makes parity harder to reason about and maintain.
- The website's own governance is weakly enforced. Broken markdown links are only warnings, translation sync checks focus on timestamps rather than navigation parity, and maintainer docs still describe outdated `.js` config filenames even though the site now uses `.ts`.
- Without a stronger IA and governance contract, capability-drift fixes like TD010 will keep regressing because there is still no single public-docs structure that defines which surfaces are canonical, legacy, advanced, or contributor-only.

## Desired End State

- A deliberate audience-oriented website IA that clearly separates getting started, local CLI usage, platform deployment, API/reference material, advanced capabilities, research/proposals, and contributor docs.
- One canonical navigation inventory where public docs are either intentionally linked from the main site structure or explicitly marked as internal, archived, or locale-specific exceptions.
- Locale and version branches follow documented parity rules so structural divergence is intentional and reviewable rather than accidental.
- Site-maintainer docs describe the current Docusaurus toolchain and governance accurately.
- The doc site includes mechanical checks for sidebar reachability, locale structure parity, and stronger link integrity expectations.

## Exit Criteria

- Homepage, intro, navbar, and sidebar present the same top-level product and documentation map.
- Public markdown docs in `website/docs/` are either reachable from canonical navigation or explicitly excluded by policy and validation.
- Locale-only sections or version-only sections are documented and mechanically distinguishable from accidental drift.
- Website maintainer docs match the current `.ts`-based Docusaurus config and sidebar files.
- CI or docs validation fails when public doc structure, navigation inventory, or link integrity drifts from the declared website IA policy.
