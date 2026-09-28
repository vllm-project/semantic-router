# Provider mocker fixture

This fixture serves native Chat Completions, Responses, Anthropic Messages, and
image-generation protocols without downloading a model. It is for Router
contract tests and local development.

```bash
make docker-build-provider-mocker
kind load docker-image semantic-router-ci/provider-mocker:e2e-test
kubectl apply -k deploy/kubernetes/provider-mocker
kubectl rollout status deployment/provider-mocker
kubectl port-forward service/provider-mocker 8000:8000
```

The checked-in manifest uses a local image and `imagePullPolicy: Never`. For a
remote cluster, set the image to a published `provider-mocker@sha256:...` digest
and the pull policy to `IfNotPresent` before deployment. Use
`PROVIDER_MOCKER_MODEL` and `PROVIDER_MOCKER_SCENARIO` environment variables to
choose the fixture model and scenario. See the
[service contract](../../../tools/test/services/provider-mocker/README.md).

For actual small-model inference, use the optional
[Qwen3-0.6B runner](../../../tools/test/services/tiny-model/README.md).
