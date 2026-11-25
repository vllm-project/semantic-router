package extproc

import (
	"fmt"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

const blockedStoreResponseAPI = `
global:
  services:
    response_api:
      enabled: true
      store_backend: memory
      ttl_seconds: 3600
      max_responses: 100
`

// A streamed block ends the turn as a completed Responses object, so a
// client with store on continues from its id. The follow-up reaches the
// model with the block's message as the earlier answer, never the answer the
// guard refused.
func TestAFollowUpChainsToAStoredBlock(t *testing.T) {
	for _, tc := range []struct {
		name, config, prompt, next, message string
		upstream                            bool
	}{
		{
			name: "fast_response", config: fmt.Sprintf(blockHistoryConfig, "", "") + blockedStoreResponseAPI,
			prompt: blockHistoryPrompt, next: blockHistoryNext, message: blockHistoryMessage,
		},
		{
			name: "response_guard", config: guardCacheConfig, upstream: true,
			prompt: "__guarded__ __refuse__ how fast is light", next: "__guarded__ and sound?",
			message: "Response blocked by the response guard",
		},
	} {
		cfg, err := config.ParseYAMLBytes([]byte(tc.config))
		if err != nil {
			t.Fatalf("parse config: %v", err)
		}
		for mode, engineFor := range looperAdapters {
			t.Run(tc.name+"/"+mode, func(t *testing.T) {
				engine := engineFor(newParityRouter(t, cfg))
				blocked := planStoredTurn(t, engine, fmt.Sprintf(`"input":%q`, tc.prompt))
				out := blocked.Immediate
				if tc.upstream {
					if out, err = engine.Respond(t.Context(), blocked, chatUpstreamAnswer(benignAnswer, true)); err != nil {
						t.Fatalf("respond: %v", err)
					}
				}
				if out == nil {
					t.Fatal("the blocked turn has no answer")
				}
				answer, _, err := protocolcodec.NewBuiltinEngine().DecodeResponseStream(llmprotocol.OpenAIResponsesV1,
					[]byte(drainResponse(t, out)), llmprotocol.StreamContext{Context: t.Context()})
				if err != nil || answer.ID == "" {
					t.Fatalf("decode the block's answer: %v %+v", err, answer)
				}
				plan := planStoredTurn(t, engine, fmt.Sprintf(`"previous_response_id":%q,"input":%q`, answer.ID, tc.next))
				if plan.Call == nil {
					t.Fatalf("the follow-up was answered by the Router: %d %s", plan.Immediate.Status, plan.Immediate.Body)
				}
				sent := string(plan.Call.Request.Body)
				if !strings.Contains(sent, tc.message) || strings.Contains(sent, benignAnswer) {
					t.Fatalf("the model receives %s, want the block's message as the earlier answer", sent)
				}
			})
		}
	}
}

// planStoredTurn plans a streamed Responses turn with store on.
func planStoredTurn(t *testing.T, engine routing.Engine, fields string) *routing.Plan {
	t.Helper()
	plan, err := engine.Plan(t.Context(), globalPluginsRequest("/v1/responses",
		`{"model":"vllm-sr/auto","stream":true,"store":true,`+fields+`}`))
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Cleanup(func() { plan.Finish(nil) })
	return plan
}
