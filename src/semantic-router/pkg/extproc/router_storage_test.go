package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A configuration that names no Responses store still serves /v1/responses,
// the way the other stores default to one inside the Router.
func TestConfigWithoutAResponsesStoreServesTheResponsesAPI(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(`
version: v0.5
providers:
  models:
    - name: chat
      provider_model_id: chat
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
recipes:
  - name: default
`))
	if err != nil {
		t.Fatal(err)
	}
	filter := createResponseAPIFilter(cfg)
	if filter == nil {
		t.Fatalf("the Responses API has no store with backend %q", cfg.ResponseAPI.StoreBackend)
	}
	filter.store.Close()
}
