package configsnapshot

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// sharedAliasDocument publishes vllm-sr/auto on both APIs: the Chat default
// entrypoint and a System One entrypoint with its own native recipe.
const sharedAliasDocument = `
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
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - api: systemone
    model_names: [vllm-sr/auto]
    recipe: native-decisions
recipes:
  - name: default
  - name: native-decisions
    decisions:
      - name: classify
        rules: {}
        modelRefs: [{model: kai}]
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
              - {name: fast, model: kai}
`

// Entrypoint names are unique per API, so a Chat and a System One entrypoint
// that share an alias compile to one route each.
func TestChatAndSystemOneEntrypointsMayShareAnAlias(t *testing.T) {
	resources := compileTest(t, parseTestConfig(t, sharedAliasDocument))

	for _, want := range []struct {
		name    string
		api     config.InferenceAPI
		path    string
		program string
	}{
		{"chat/vllm-sr/auto", config.ChatAPI, "entrypoints[0]", "default"},
		{"systemone/vllm-sr/auto", config.SystemOneAPI, "entrypoints[1]", "native-decisions"},
	} {
		route := mustGet(t, resources, KindRoute, want.name)
		if route.Path != want.path {
			t.Fatalf("%s path = %q, want %q", want.name, route.Path, want.path)
		}
		if got := route.Spec.(RouteSpec); got != (RouteSpec{API: want.api, Model: "vllm-sr/auto", Program: want.program}) {
			t.Fatalf("%s spec = %+v", want.name, got)
		}
		if got := refNames(route.Refs); !reflect.DeepEqual(got, []string{"program/" + want.program}) {
			t.Fatalf("%s refs = %v", want.name, got)
		}
	}
}
