package extproc

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// sharedAliasConfig keeps vllm-sr/auto on the Chat default entrypoint and
// publishes it on System One too, as the escalate tutorial does.
const sharedAliasConfig = `
version: v0.5
listeners:
  - name: native
    port: 8801
    apis:
      chat_completions: {}
      systemone:
        schema: vllm-sr/4.0
        models: [vllm-sr/auto]
providers:
  defaults:
    model: local-model
  models:
    - name: local-model
      backend_refs:
        - provider: vllm
          endpoint: 127.0.0.1:1/v1
    - name: strong
      api_format: systemone
      provider_model_id: vllm-sr/Decision-2.0-Nox-4B
      backend_refs:
        - provider: systemone-compatible
          endpoint: http://127.0.0.1:1/v1
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-decisions
recipes:
  - name: default
    decisions:
      - name: default-decision
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: local-model}]
  - name: native-decisions
    decisions:
      - name: classify
        rules: {}
        modelRefs: [{model: strong}]
        algorithm:
          type: escalate
          budget: {deadline: 3s, max_calls: 1}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native:
                  - {question_type: choice, field: top_probability, gte: 0.9}
            stages:
              - {name: strong, model: strong}
`

// A Chat and a System One entrypoint may share an alias, since entrypoint
// names are unique per API, and the Router starts serving both.
func TestRouterStartsWithAnAliasOnChatAndSystemOne(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(sharedAliasConfig), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	server, err := NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg))
	if err != nil {
		t.Fatalf("NewServer() error = %v, want the Router to start", err)
	}
	t.Cleanup(func() { _ = server.service.Close() })

	resources := server.service.Snapshot().Resources()
	for _, name := range []string{"chat/vllm-sr/auto", "systemone/vllm-sr/auto"} {
		if _, ok := resources.Get(configsnapshot.Ref{Kind: configsnapshot.KindRoute, Name: name}); !ok {
			t.Fatalf("the active snapshot has no route %s", name)
		}
	}
	if entrypoint, ok := server.CurrentConfig().ResolveEntrypoint(config.SystemOneAPI, "vllm-sr/auto"); !ok || entrypoint.Recipe != "native-decisions" {
		t.Fatalf("System One vllm-sr/auto = %+v, %v; want native-decisions", entrypoint, ok)
	}
}
