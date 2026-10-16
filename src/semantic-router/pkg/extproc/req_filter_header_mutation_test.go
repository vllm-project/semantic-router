package extproc

import (
	"testing"

	corev3 "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestBuildHeaderMutationsUsesDistinctAddAndUpdateSemantics(t *testing.T) {
	payload, err := config.NewStructuredPayload(map[string]interface{}{
		"add": []map[string]string{
			{"name": "x-added", "value": "added"},
		},
		"update": []map[string]string{
			{"name": "x-updated", "value": "updated"},
		},
		"delete": []string{"x-deleted"},
	})
	if err != nil {
		t.Fatal(err)
	}
	decision := &config.Decision{
		Name: "header-mutation",
		Plugins: []config.DecisionPlugin{
			{Type: config.DecisionPluginHeaderMutation, Configuration: payload},
		},
	}

	setHeaders, removeHeaders := (&OpenAIRouter{}).buildHeaderMutations(decision)
	if len(setHeaders) != 2 {
		t.Fatalf("set headers = %d, want 2", len(setHeaders))
	}
	if got := setHeaders[0].GetAppendAction(); got != corev3.HeaderValueOption_APPEND_IF_EXISTS_OR_ADD {
		t.Fatalf("add append action = %v, want APPEND_IF_EXISTS_OR_ADD", got)
	}
	if got := setHeaders[1].GetAppendAction(); got != corev3.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
		t.Fatalf("update append action = %v, want OVERWRITE_IF_EXISTS_OR_ADD", got)
	}
	if len(removeHeaders) != 1 || removeHeaders[0] != "x-deleted" {
		t.Fatalf("remove headers = %#v, want [x-deleted]", removeHeaders)
	}
}
