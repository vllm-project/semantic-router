package extproc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func automaticFixture(t *testing.T, text string, handler http.HandlerFunc) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	r, ctx := overflowFixture(t, text)
	name := ctx.VSRSelectedDecision.ModelRefs[0].Model
	params := r.Config.ModelConfig[name]
	params.MaxOutputTokens = params.ContextWindowSize
	r.Config.ModelConfig[name] = params
	r.Config.ProviderProfiles["provider"] = config.ProviderProfile{Type: "vllm", BaseURL: server.URL + "/v1"}
	payload, err := config.NewStructuredPayload(map[string]any{"default_max_tokens": "auto"})
	require.NoError(t, err)
	ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: "request_params", Configuration: payload})
	ctx.SemanticRequest.Sampling.MaxOutputTokens = nil
	snapshotClientMaxOutputTokens(*ctx.SemanticRequest, ctx)
	return r, ctx
}

func renderMock(t *testing.T, calls *int, capReduction int, failStatus int) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		*calls++
		require.Equal(t, "/v1/chat/completions/render", req.URL.Path)
		var body map[string]any
		require.NoError(t, json.NewDecoder(req.Body).Decode(&body))
		if failStatus != 400 {
			require.Nil(t, body["max_tokens"])
		}
		require.Nil(t, body["max_completion_tokens"])
		if failStatus != 0 {
			w.WriteHeader(failStatus)
			fmt.Fprint(w, `{"error":{"param":"temperature"}}`)
			return
		}
		messages := body["messages"].([]any)
		count := 17
		for _, item := range messages {
			if value, ok := item.(map[string]any)["content"].(string); ok {
				count += len([]rune(value))
			}
		}
		capacity := 32768
		if body["model"] == "wide" {
			capacity = 65536
		}
		if count >= capacity {
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"param":"input_tokens"}}`)
			return
		}
		require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "token_ids": make([]int, count), "features": nil, "sampling_params": map[string]any{"max_tokens": capacity - count - capReduction}}))
	}
}

func TestAutomaticOutputUsesExactRenderedCandidateAndFinalBudgets(t *testing.T) {
	calls := 0
	text := "HEAD\n" + strings.Repeat("中文🙂", 9000) + "\nTAIL"
	r, ctx := automaticFixture(t, text, renderMock(t, &calls, 0, 0))
	raw, err := json.Marshal(map[string]any{"model": "auto", "messages": []map[string]any{{"role": "user", "content": text}}, "top_k": 20, "min_p": 0, "repetition_penalty": 1, "cache_salt": "arm_cache_namespace"})
	require.NoError(t, err)
	decoded, rejection := r.prepareProtocolRequest(raw, ctx)
	require.Nil(t, rejection)
	require.NotNil(t, decoded)
	d := ctx.VSRSelectedDecision
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	require.False(t, ctx.ContextCompressionApplied, "valid non-ASCII input must not be byte-truncated")
	refs, err := r.decisionEligibleModelRefs(d, ctx)
	require.NoError(t, err)
	require.Len(t, refs, 1)
	demand := ctx.AutomaticCandidateDemands[refs[0].Model]
	require.Equal(t, len([]rune(text))+17, demand.InputTokens)
	require.EqualValues(t, 32768-demand.InputTokens, *demand.MaxOutputTokens)
	require.Nil(t, ctx.SemanticRequest.Sampling.MaxOutputTokens, "selection must not mutate ingress")
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, refs[0].Model, d.Name, false, ctx)
	require.NoError(t, err)
	ctx.SemanticRequest.Messages[0].Content[0].Text += "more"
	ctx.SemanticRequest.Generation++
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	response, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	require.NoError(t, err)
	var wire map[string]any
	require.NoError(t, json.Unmarshal(response.GetRequestBody().Response.BodyMutation.GetBody(), &wire))
	wireMax := wire["max_tokens"]
	if wireMax == nil {
		wireMax = wire["max_completion_tokens"]
	}
	require.EqualValues(t, 32768-demand.InputTokens-4, wireMax)
	require.EqualValues(t, 20, wire["top_k"])
	require.Equal(t, "arm_cache_namespace", wire["cache_salt"])
	require.Equal(t, 3, calls)
	require.Equal(t, text+"more", ctx.SemanticRequest.Messages[0].Content[0].Text)
	preview := r.SelectModelForEval(services.EvalModelSelectionInput{Decision: d, Demand: selection.CandidateDemand{Known: true, AutomaticOutput: true}})
	require.Equal(t, services.EvalSelectionUnavailable, preview.Status)
}

func TestAutomaticOutputDifferentCandidateWindowsPreserveInput(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, strings.Repeat("a", 40000), renderMock(t, &calls, 0, 0))
	first := ctx.VSRSelectedDecision.ModelRefs[0].Model
	wide := r.Config.ModelConfig[first]
	wide.ContextWindowSize = 65536
	wide.MaxOutputTokens = 65536
	wide.ExternalModelIDs = nil
	r.Config.ModelConfig["wide"] = wide
	ctx.VSRSelectedDecision.ModelRefs = append(ctx.VSRSelectedDecision.ModelRefs, config.ModelRef{Model: "wide"})
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	refs, err := r.decisionEligibleModelRefs(ctx.VSRSelectedDecision, ctx)
	require.NoError(t, err)
	require.Equal(t, []config.ModelRef{{Model: "wide"}}, refs)
	require.False(t, ctx.ContextCompressionApplied)
	require.EqualValues(t, 65536-40017, *ctx.AutomaticCandidateDemands["wide"].MaxOutputTokens)
}

func TestAutomaticOutputConfirmedOverflowUsesConfiguredCompression(t *testing.T) {
	calls := 0
	original := "HEAD\n" + strings.Repeat("archive text ", 6000) + "\nTAIL"
	r, ctx := automaticFixture(t, original, renderMock(t, &calls, 0, 0))
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	require.Equal(t, 2, calls)
	require.True(t, ctx.ContextCompressionApplied)
	changed := ctx.SemanticRequest.Messages[0].Content[0].Text
	require.True(t, strings.HasPrefix(changed, "HEAD\n"))
	require.True(t, strings.HasSuffix(changed, "\nTAIL"))
	require.Contains(t, changed, "context omitted by route compression")
	for _, demand := range ctx.AutomaticCandidateDemands {
		require.EqualValues(t, 32768, int64(demand.InputTokens)+*demand.MaxOutputTokens)
	}
}

func TestAutomaticOutputRendererErrorsAndHiddenCapsNeverCompress(t *testing.T) {
	for _, tt := range []struct {
		name              string
		status, reduction int
	}{{"not-enabled", 404, 0}, {"invalid-parameter", 400, 0}, {"hidden-output-cap", 0, 100}} {
		t.Run(tt.name, func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, strings.Repeat("中文", 10000), renderMock(t, &calls, tt.reduction, tt.status))
			err := r.prepareDecisionContextOverflow(ctx, "auto")
			require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
			wantCalls := 1
			if tt.status == 400 {
				wantCalls = 2
			}
			require.Equal(t, wantCalls, calls)
			require.False(t, ctx.ContextCompressionApplied)
		})
	}
}

func TestAutomaticOutputExplicitCallerHasNoRenderCalls(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	snapshotClientMaxOutputTokens(*ctx.SemanticRequest, ctx)
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	_, err := r.decisionEligibleModelRefs(ctx.VSRSelectedDecision, ctx)
	require.NoError(t, err)
	name := ctx.VSRSelectedDecision.ModelRefs[0].Model
	dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, name, ctx.VSRSelectedDecision.Name, false, ctx)
	require.NoError(t, err)
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
	_, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
	require.NoError(t, err)
	require.Equal(t, 0, calls)
	require.EqualValues(t, 12, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
}

func TestAutomaticOutputBlockedCallerBudgetUsesRenderer(t *testing.T) {
	for _, field := range []string{"max_tokens", "max_completion_tokens", " max_output_tokens "} {
		t.Run(strings.TrimSpace(field), func(t *testing.T) {
			calls := 0
			r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
			ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
			snapshotClientMaxOutputTokens(*ctx.SemanticRequest, ctx)
			setAutomaticBlockedParams(t, ctx, []string{field})
			require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
			refs, err := r.decisionEligibleModelRefs(ctx.VSRSelectedDecision, ctx)
			require.NoError(t, err)
			require.Len(t, refs, 1)
			demand := ctx.AutomaticCandidateDemands[refs[0].Model]
			require.Equal(t, 22, demand.InputTokens)
			require.EqualValues(t, 32746, *demand.MaxOutputTokens)
			require.Equal(t, 1, calls)
			require.EqualValues(t, 12, *ctx.SemanticRequest.Sampling.MaxOutputTokens, "selection must not mutate ingress")
			require.False(t, ctx.SemanticRequest.Sampling.AutomaticOutput)
			dispatch, err := r.prepareProviderDispatch(ctx.SemanticRequest, refs[0].Model, ctx.VSRSelectedDecision.Name, false, ctx)
			require.NoError(t, err)
			response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{}}}}
			_, err = r.finalizeProviderDispatchResponse(dispatch, response, ctx)
			require.NoError(t, err)
			require.Equal(t, 3, calls)
			require.EqualValues(t, 32746, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
		})
	}
}

func TestAutomaticOutputBlockedCallerBudgetRejectsInvalidPolicyBeforeRender(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
	ctx.SemanticRequest.Sampling.MaxOutputTokens = llmprotocol.Int64(12)
	snapshotClientMaxOutputTokens(*ctx.SemanticRequest, ctx)
	setAutomaticBlockedParams(t, ctx, []string{"max_tokens", "messages"})
	err := r.prepareDecisionContextOverflow(ctx, "auto")
	require.ErrorContains(t, err, "required semantic field")
	require.Equal(t, 0, calls)
	require.Nil(t, ctx.AutomaticCandidateDemands)
	require.False(t, ctx.ContextCompressionApplied)
	require.EqualValues(t, 12, *ctx.SemanticRequest.Sampling.MaxOutputTokens)
}

func setAutomaticBlockedParams(t *testing.T, ctx *RequestContext, fields []string) {
	t.Helper()
	payload, err := config.NewStructuredPayload(map[string]any{"default_max_tokens": "auto", "blocked_params": fields})
	require.NoError(t, err)
	for i := range ctx.VSRSelectedDecision.Plugins {
		if ctx.VSRSelectedDecision.Plugins[i].Type == "request_params" {
			ctx.VSRSelectedDecision.Plugins[i].Configuration = payload
			return
		}
	}
	t.Fatal("automatic fixture requires request_params")
}

func TestAutomaticOutputCapabilityFilterAndFastResponse(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "use the calculator", renderMock(t, &calls, 0, 0))
	model := ctx.VSRSelectedDecision.ModelRefs[0].Model
	ctx.SemanticRequest.Tools = []llmprotocol.Tool{{Name: "calculator", InputSchema: json.RawMessage(`{"type":"object"}`)}}
	incompatible := r.Config.ModelConfig[model]
	incompatible.Capabilities = []string{"chat"}
	r.Config.ModelConfig["no-tools"] = incompatible
	ctx.VSRSelectedDecision.ModelRefs = append(ctx.VSRSelectedDecision.ModelRefs, config.ModelRef{Model: "no-tools"})
	refs, err := r.decisionEligibleModelRefs(ctx.VSRSelectedDecision, ctx)
	require.NoError(t, err)
	require.Equal(t, []config.ModelRef{{Model: model}}, refs)
	require.Equal(t, 1, calls)
	fast, err := config.NewStructuredPayload(map[string]any{"status_code": 200, "body": "ready"})
	require.NoError(t, err)
	ctx.VSRSelectedDecision.Plugins = append(ctx.VSRSelectedDecision.Plugins, config.DecisionPlugin{Type: "fast_response", Configuration: fast})
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	require.Equal(t, 1, calls)
}

func TestAutomaticOutputExactlyFullWindowUsesOneTokenProbe(t *testing.T) {
	calls := 0
	r, ctx := automaticFixture(t, "HEAD\n"+strings.Repeat("a", 32741)+"\nTAIL", func(w http.ResponseWriter, req *http.Request) {
		calls++
		var body map[string]any
		require.NoError(t, json.NewDecoder(req.Body).Decode(&body))
		switch calls {
		case 1:
			require.Nil(t, body["max_tokens"])
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"param":null}}`)
		case 2:
			require.EqualValues(t, 1, body["max_tokens"])
			require.Nil(t, body["max_completion_tokens"])
			w.WriteHeader(400)
			fmt.Fprint(w, `{"error":{"param":"input_tokens"}}`)
		case 3:
			require.Nil(t, body["max_tokens"])
			require.NoError(t, json.NewEncoder(w).Encode(map[string]any{"model": body["model"], "token_ids": make([]int, 31000), "sampling_params": map[string]any{"max_tokens": 1768}}))
		default:
			t.Fatal("unbounded render calls")
		}
	})
	require.NoError(t, r.prepareDecisionContextOverflow(ctx, "auto"))
	require.Equal(t, 3, calls)
	require.True(t, ctx.ContextCompressionApplied)
}

func TestAutomaticOutputUnsupportedProviderAndMultipleBackends(t *testing.T) {
	for _, multiple := range []bool{false, true} {
		calls := 0
		r, ctx := automaticFixture(t, "hello", renderMock(t, &calls, 0, 0))
		if multiple {
			name := ctx.VSRSelectedDecision.ModelRefs[0].Model
			params := r.Config.ModelConfig[name]
			params.PreferredEndpoints = append(params.PreferredEndpoints, "other")
			r.Config.ModelConfig[name] = params
		} else {
			profile := r.Config.ProviderProfiles["provider"]
			profile.Type = "openai"
			r.Config.ProviderProfiles["provider"] = profile
		}
		require.ErrorIs(t, r.prepareDecisionContextOverflow(ctx, "auto"), selection.ErrNoEligibleCandidates)
		require.Equal(t, 0, calls)
	}
}

func TestAutomaticOutputVLLMDefaultSamplingSerialization(t *testing.T) {
	for _, tt := range []struct {
		name, sampling string
		input          int
		ok             bool
	}{
		{"omitted-default", `{}`, 32752, true},
		{"explicit-default", `{"max_tokens":16}`, 32752, true},
		{"null-limit", `{"max_tokens":null}`, 32752, false},
		{"zero-limit", `{"max_tokens":0}`, 32752, false},
		{"negative-limit", `{"max_tokens":-1}`, 32752, false},
		{"null-object", `null`, 32752, false},
		{"missing-object", ``, 32752, false},
		{"default-does-not-hide-smaller-cap", `{}`, 100, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, ctx := automaticFixture(t, "hello", func(w http.ResponseWriter, req *http.Request) {
				var body map[string]any
				require.NoError(t, json.NewDecoder(req.Body).Decode(&body))
				output := map[string]any{"model": body["model"], "token_ids": make([]int, tt.input)}
				if tt.sampling != "" {
					output["sampling_params"] = json.RawMessage(tt.sampling)
				}
				require.NoError(t, json.NewEncoder(w).Encode(output))
			})
			err := r.prepareDecisionContextOverflow(ctx, "auto")
			if tt.ok {
				require.NoError(t, err)
				for _, demand := range ctx.AutomaticCandidateDemands {
					require.EqualValues(t, 16, *demand.MaxOutputTokens)
				}
			} else {
				require.ErrorIs(t, err, selection.ErrNoEligibleCandidates)
			}
		})
	}
}
