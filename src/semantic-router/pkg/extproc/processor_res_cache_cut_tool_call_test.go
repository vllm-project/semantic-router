package extproc

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// A buffered answer the output limit cut inside its last tool call reaches the
// client but is never cached, since a hit would give another request the cut
// call; a whole call that ends at the same stop is still cached.
func TestUpdateResponseCacheSkipsAnAnswerCutInAToolCall(t *testing.T) {
	answer := func(arguments string) *llmprotocol.Response {
		return &llmprotocol.Response{
			StopReason: llmprotocol.StopMaxTokens,
			Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
				Kind:     llmprotocol.ContentToolCall,
				ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "lookup", Arguments: arguments},
			}}}},
		}
	}
	for name, tc := range map[string]struct {
		arguments string
		cached    bool
	}{
		"cut":   {arguments: `{"city":"Par`},
		"empty": {arguments: ``},
		"whole": {arguments: `{"city":"Paris"}`, cached: true},
	} {
		t.Run(name, func(t *testing.T) {
			mockCache, router, decision := statusCacheRouter()
			ctx := withSelectedDecision(&RequestContext{
				RequestID: "req-" + name, UpstreamStatusCode: 200,
				RequestModel: "test", RequestQuery: "weather",
				SemanticRequest:  testNeutralRequest("test", "weather"),
				SemanticResponse: answer(tc.arguments),
			}, decision)
			router.updateResponseCache(ctx, []byte(`{"choices":[]}`))
			if mockCache.addEntryCalled != tc.cached {
				t.Fatalf("cache written = %v, want %v", mockCache.addEntryCalled, tc.cached)
			}
		})
	}
}
