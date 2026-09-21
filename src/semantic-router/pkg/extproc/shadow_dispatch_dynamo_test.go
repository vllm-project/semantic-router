package extproc

import (
	"encoding/json"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func TestShadowDispatchSkipsDynamoState(t *testing.T) {
	for _, target := range []string{"vllm", "dynamo", "cross-format"} {
		for _, source := range []string{"nvext", "empty-nvext", "cache_salt", "header", "profile", "decision"} {
			t.Run(target+"/"+source, func(t *testing.T) {
				backend := newShadowTestBackend(t)
				router, primaryModel := newShadowTestRouter(t, backend)
				for i := range router.Config.VLLMEndpoints {
					endpoint := &router.Config.VLLMEndpoints[i]
					endpoint.Type = "dynamo"
					if endpoint.Name == "shadow-backend" && target == "vllm" {
						endpoint.Type = "vllm"
					}
				}
				if target == "cross-format" {
					model := router.Config.ModelConfig[shadowTestModel]
					model.APIFormat = config.APIFormatResponses
					router.Config.ModelConfig[shadowTestModel] = model
				}
				if source == "profile" {
					profile := router.Config.ProviderProfiles["provider"]
					profile.ExtraHeaders = map[string]string{headers.DynamoDPRank: "7"}
					router.Config.ProviderProfiles["provider"] = profile
				}
				before := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonDynamoState)
				run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), func(ctx *RequestContext) {
					ctx.ProtocolEnvelope.Format = llmprotocol.OpenAIChatV1
					switch source {
					case "nvext", "empty-nvext":
						ext := &llmprotocol.DynamoRequestNVExt{}
						if source == "nvext" {
							ext.UseRawPrompt = llmprotocol.Bool(true)
						}
						ctx.ProtocolEnvelope.Dynamo = &llmprotocol.DynamoEnvelope{RequestNVExt: ext}
					case "cache_salt":
						salt := "tenant-a"
						ctx.ProtocolEnvelope.Dynamo = &llmprotocol.DynamoEnvelope{RequestTopLevelCacheSalt: &salt}
					case "header":
						ctx.Headers[headers.DynamoDPRank] = "7"
					case "decision":
						ctx.VSRSelectedDecision.Plugins = []config.DecisionPlugin{{
							Type: config.DecisionPluginHeaderMutation,
							Configuration: config.MustStructuredPayload(map[string]interface{}{
								"update": []map[string]string{{"name": headers.DynamoDPRank, "value": "7"}},
							}),
						}}
					}
				})
				waitForShadow(t, router)
				if backend.requestCount() != 0 {
					t.Fatal("Dynamo state reached shadow backend")
				}
				if got := shadowCounter(shadowTestDecision, metrics.ShadowDispatchResultDropped, shadowReasonDynamoState) - before; got != 1 {
					t.Fatalf("skip counter delta = %v, want 1", got)
				}
				var primary map[string]json.RawMessage
				if err := json.Unmarshal(run.body, &primary); err != nil {
					t.Fatalf("primary dispatch failed: %v", err)
				}
				if len(primary["messages"]) == 0 {
					t.Fatal("primary request lost messages")
				}
				if source == "nvext" || source == "empty-nvext" {
					if len(primary["nvext"]) == 0 {
						t.Fatal("shadow skip removed primary nvext")
					}
				}
				if source == "cache_salt" && string(primary["cache_salt"]) != `"tenant-a"` {
					t.Fatal("shadow skip changed primary cache_salt")
				}
			})
		}
	}
}

func TestShadowDispatchAllowsOrdinaryRequestToDynamo(t *testing.T) {
	backend := newShadowTestBackend(t)
	router, primaryModel := newShadowTestRouter(t, backend)
	for i := range router.Config.VLLMEndpoints {
		router.Config.VLLMEndpoints[i].Type = "dynamo"
	}
	run := runShadowRequest(t, router, primaryModel, shadowTestPluginConfig(), nil)
	waitForShadow(t, router)
	if backend.requestCount() != 1 {
		t.Fatalf("requests = %d, want 1", backend.requestCount())
	}
	if outcome := singleShadowOutcome(t, run); outcome.Verdict != shadowVerdictCompleted {
		t.Fatalf("outcome = %+v", outcome)
	}
}
