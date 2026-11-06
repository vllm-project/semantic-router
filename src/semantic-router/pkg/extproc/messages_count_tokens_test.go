package extproc

import (
	"encoding/json"
	"math/rand"
	"regexp"
	"strings"
	"testing"
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// Claude Code's Read tool sends one file as one user message, with no
// max_tokens, and compares the count with its read limit.
const countTokensTestBody = `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"package main\nfunc main() {}\n"}],"tools":[]}`

func sendCountTokens(t *testing.T, router *OpenAIRouter, session, body string) (*ext_proc.ProcessingResponse, *RequestContext) {
	t.Helper()
	ctx := routingTestContext(llmprotocol.AnthropicMessagesV1, nil)
	ctx.Headers[":path"] = messagesCountPath + "?beta=true"
	ctx.Headers["x-claude-code-session-id"] = session
	stream := NewMockStream(nil)
	require.NoError(t, router.processRequestBody(stream, &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true},
	}, ctx))
	require.Len(t, stream.Responses, 1)
	return stream.Responses[0], ctx
}

func countTokensEstimate(t *testing.T, response *ext_proc.ProcessingResponse) int {
	t.Helper()
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate, "the Router answers with its own estimate")
	require.EqualValues(t, 200, immediate.GetStatus().GetCode())
	var answer struct {
		InputTokens int `json:"input_tokens"`
	}
	require.NoError(t, json.Unmarshal(immediate.GetBody(), &answer))
	return answer.InputTokens
}

func TestCountTokensUsesTheSessionModelsMessagesBackend(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	router := routingTestRouter("test-model")
	// A decision on the shape of the conversation reads no content.
	router.Config.Decisions = []config.Decision{{
		Name: "escalate", ModelRefs: []config.ModelRef{{Model: "test-model"}},
		Rules: config.RuleNode{Type: config.SignalTypeConversation, Name: "tool_results_10"},
	}}
	sessiontelemetry.RecordLastModel(sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "count-session"), "test-model")

	response, ctx := sendCountTokens(t, router, "count-session", countTokensTestBody)

	common := response.GetRequestBody().GetResponse()
	require.NotNil(t, common, "the request goes to the backend")
	path := ""
	for _, header := range common.GetHeaderMutation().GetSetHeaders() {
		if header.GetHeader().GetKey() == ":path" {
			path = string(header.GetHeader().GetRawValue())
		}
	}
	require.Equal(t, messagesCountPath, path)
	var sent map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(common.GetBodyMutation().GetBody(), &sent))
	require.JSONEq(t, `"test-model"`, string(sent["model"]))
	for name := range sent {
		require.Contains(t, countTokensFields, name, "count_tokens accepts no generation field")
	}
	require.True(t, ctx.SkipProcessing, "the backend's count passes back unchanged")
	require.Empty(t, ctx.RouterReplayID, "counting is not a turn")
	model, _, _ := sessiontelemetry.PeekLastModelInfo(sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "count-session"), time.Time{})
	require.Equal(t, "test-model", model)
}

func TestCountTokensEstimatesWithoutAMessagesBackend(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	request := llmprotocol.Request{Messages: []llmprotocol.Message{{
		Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "package main\nfunc main() {}\n"}},
	}}}
	want := estimateCountTokens(&request)

	t.Run("no model served the session yet", func(t *testing.T) {
		response, _ := sendCountTokens(t, routingTestRouter("test-model"), "new-session", countTokensTestBody)
		require.Equal(t, want, countTokensEstimate(t, response))
	})
	t.Run("the model's backend speaks Chat", func(t *testing.T) {
		body := strings.Replace(countTokensTestBody, "vllm-sr/auto", "chat-model", 1)
		response, _ := sendCountTokens(t, routingTestRouter("chat-model"), "chat-session", body)
		require.Equal(t, want, countTokensEstimate(t, response))
	})
	masking, err := config.NewStructuredPayload(map[string]any{"enabled": true, "signal": "personal"})
	require.NoError(t, err)
	fastResponse, err := config.NewStructuredPayload(map[string]any{"message": "refused"})
	require.NoError(t, err)
	stripTools, err := config.NewStructuredPayload(map[string]any{"enabled": true, "policy": map[string]any{"mode": "none", "strip_tool_history": true}})
	require.NoError(t, err)
	for name, guard := range map[string]config.Decision{
		"a decision masks content":       {Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginMasking, Configuration: masking}}},
		"a decision reads content":       {Rules: config.RuleNode{Type: config.SignalTypeKeyword, Name: "confidential"}},
		"a decision answers by itself":   {Plugins: []config.DecisionPlugin{{Type: "fast_response", Configuration: fastResponse}}},
		"a decision strips tool history": {Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginTools, Configuration: stripTools}}},
	} {
		t.Run(name, func(t *testing.T) {
			router := routingTestRouter("test-model")
			guard.Name, guard.ModelRefs = "guard", []config.ModelRef{{Model: "test-model"}}
			router.Config.Decisions = []config.Decision{guard}
			sessiontelemetry.RecordLastModel(sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "guarded-session"), "test-model")
			response, _ := sendCountTokens(t, router, "guarded-session", countTokensTestBody)
			require.Equal(t, want, countTokensEstimate(t, response), "content a decision guards never reaches the backend")
		})
	}
}

func TestCountTokensKeepsRoutingAdmission(t *testing.T) {
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	t.Cleanup(sessiontelemetry.ResetRouterSessionMemoryForTesting)
	key := sessiontelemetry.RoutingSessionKey(config.DefaultRecipeName, "admission-session")

	t.Run("a rate limit denies the caller", func(t *testing.T) {
		router := routingTestRouter("test-model")
		router.Config.RateLimit.Providers = []config.RateLimitProviderConfig{{
			Type: "local-limiter",
			Rules: []config.RateLimitRule{{
				Name: "one-per-minute", Match: config.RateLimitMatch{Model: "test-model"}, RequestsPerUnit: 1, Unit: "minute",
			}},
		}}
		router.RateLimiter = buildRateLimitResolver(router.Config)
		sessiontelemetry.RecordLastModel(key, "test-model")
		_, first := sendCountTokens(t, router, "admission-session", countTokensTestBody)
		require.True(t, first.SkipProcessing, "the first count goes upstream")
		response, second := sendCountTokens(t, router, "admission-session", countTokensTestBody)
		require.EqualValues(t, 429, response.GetImmediateResponse().GetStatus().GetCode())
		require.False(t, second.SkipProcessing, "a denied count never goes upstream")
	})
	t.Run("a token budget is smaller than the input", func(t *testing.T) {
		router := routingTestRouter("test-model")
		router.Config.RateLimit.Providers = []config.RateLimitProviderConfig{{
			Type: "local-limiter",
			Rules: []config.RateLimitRule{{
				Name: "five-tokens", Match: config.RateLimitMatch{Model: "test-model"}, TokensPerUnit: 5, Unit: "minute",
			}},
		}}
		router.RateLimiter = buildRateLimitResolver(router.Config)
		sessiontelemetry.RecordLastModel(key, "test-model")
		response, ctx := sendCountTokens(t, router, "admission-session", countTokensTestBody)
		require.EqualValues(t, 429, response.GetImmediateResponse().GetStatus().GetCode())
		require.False(t, ctx.SkipProcessing, "an over-budget count never goes upstream")
	})
	t.Run("candidate requirements apply", func(t *testing.T) {
		router := routingTestRouter("test-model")
		router.Config.CandidateRequirements = &config.CandidateRequirements{Context: config.CandidateContextKnownLimits}
		sessiontelemetry.RecordLastModel(key, "test-model")
		response, _ := sendCountTokens(t, router, "admission-session", countTokensTestBody)
		require.Positive(t, countTokensEstimate(t, response), "the remembered model may not admit this input")
	})
}

func TestCountTokensBodyDecodesWithoutAnOutputLimit(t *testing.T) {
	limited := withCountTokensOutputLimit([]byte(`{"model":"m","messages":[],"thinking":{"type":"enabled","budget_tokens":2048}}`))
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(limited, &fields))
	require.JSONEq(t, `2049`, string(fields["max_tokens"]), "the placeholder stays above the thinking budget")
	explicit := []byte(`{"model":"m","messages":[],"max_tokens":7}`)
	require.Equal(t, explicit, withCountTokensOutputLimit(explicit))
	// The protocol decoder, not the placeholder, refuses a null body.
	require.Equal(t, []byte(`null`), withCountTokensOutputLimit([]byte(`null`)))
}

func TestCountTokensEstimateFollowsPreTokenizerPieces(t *testing.T) {
	text := func(s string) *llmprotocol.Request {
		return &llmprotocol.Request{Messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: s}}}}}
	}
	framing := estimateCountTokens(text(""))
	require.Equal(t, 2, estimateCountTokens(text("hello world"))-framing, "a word with its leading space is one piece")
	require.Equal(t, 3, estimateCountTokens(text("hello"+strings.Repeat(" ", 400)+"world"))-framing, "a whitespace run is one token")
	require.Equal(t, 5, estimateCountTokens(text(strings.Repeat("a", 30)))-framing, "a long piece counts one token per six bytes")
	art := text(strings.Repeat(strings.Repeat(" ", 60)+"#\n", 200))
	require.Less(t, estimateCountTokens(art), llmprotocol.EstimateInput(art).Tokens/4, "padding is not counted by the byte")
}

func TestCountTokensEstimateScansWithoutAllocating(t *testing.T) {
	digits := strings.Repeat("7", 1<<20)
	require.Equal(t, 1<<20, textPieceTokens(digits), "every digit is a piece")
	require.Zero(t, testing.AllocsPerRun(1, func() { _ = textPieceTokens(digits) }))
}

// The scanner splits exactly as the pattern it documents.
func TestCountTokensScannerMatchesThePattern(t *testing.T) {
	pattern := regexp.MustCompile(`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+`)
	byPattern := func(text string) int {
		tokens := 0
		for _, piece := range pattern.FindAllString(text, -1) {
			if strings.TrimSpace(piece) == "" {
				tokens++
			} else {
				tokens += (len(piece) + 5) / 6
			}
		}
		return tokens
	}
	alphabet := []string{"a", "Z", "é", "字", "7", "٣", " ", "  ", "\t", "\n", "\r\n", "\f", "\u00a0", "'", "'s", "'RE", "'ll", "-", "==", "{", "\"", "\xff", "ab_c", "\U0001F600"}
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 5000; i++ {
		var b strings.Builder
		for n := rng.Intn(40); n > 0; n-- {
			b.WriteString(alphabet[rng.Intn(len(alphabet))])
		}
		text := b.String()
		require.Equal(t, byPattern(text), textPieceTokens(text), "%q", text)
	}
}
