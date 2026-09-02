# Deployment assets

`deploy/` contains only assets that create or configure a deployment target:

- `helm/` and `operator/` for packaged Kubernetes installation;
- `kubernetes/`, `kserve/`, and `openshift/` for platform manifests;
- `local/` for the local Envoy deployment boundary.

Router configuration is intentionally separate:

- complete use cases: `config/recipes/`;
- reusable config fragments: `config/fragments/{signal,decision,algorithm,plugin}/`;
- runtime backend examples: `config/runtime/`;
- development utilities and auxiliary servers: `tools/`.

Do not add prose-only guides, benchmark output, helper programs, or standalone
router examples here. Public instructions belong in `website/` and repository
automation belongs in `tools/`.

The public
[Deployment and Hardware Support Matrix](../website/docs/installation/support-matrix.md)
is the canonical classification of these assets. Adding or removing a direct
child of `deploy/` or `deploy/kubernetes/` requires updating that matrix; the
repository's deployment-matrix check enforces complete coverage.
