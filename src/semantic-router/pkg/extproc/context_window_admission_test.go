package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"go.uber.org/zap/zaptest/observer"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount"
)

// promptOfTokens is English prose that the proxy tokenizer, o200k_base,
// counts at exactly tokens: whole sentences, then " a", one token each.
func promptOfTokens(t *testing.T, tokens int) string {
	t.Helper()
	proxy, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	perSentence := proxy.Count(englishSentence+englishSentence) - proxy.Count(englishSentence)
	text := strings.TrimRight(strings.Repeat(englishSentence, tokens/perSentence-1), " ")
	text += strings.Repeat(" a", tokens-proxy.Count(text))
	require.Equal(t, tokens, proxy.Count(text))
	return text
}

// routerCountOf is the Router's count, with the proxy tokenizer, of a
// one-message prompt of text.
func routerCountOf(text string) int {
	proxy, err := tokencount.Builtin(tokencount.O200kBase)
	if err != nil {
		panic(err)
	}
	return countPromptTokens(proxy, testNeutralRequest("vllm-sr/auto", text))
}

func longWindowRouterFor(t *testing.T) *OpenAIRouter {
	t.Helper()
	return longWindowRouter(t, strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1))
}

func runLongWindowCaseAgainst(t *testing.T, router *OpenAIRouter, name, body string, upstream *parity.Upstream) *parity.Record {
	t.Helper()
	return parity.NewRecorder(router, routing.DefaultOptions).Run(context.Background(), parity.Case{
		Name: name,
		Request: parity.CaseRequest{
			Method: "POST", Path: "/v1/chat/completions",
			Headers: [][]string{{"content-type", "application/json"}}, Body: body,
		},
		Upstream: upstream,
	})
}

// withCounts gives ctx a request longer than the models' windows in bytes,
// with the Router's counts of it, so the window check rests on those counts.
// Models that share a tokenizer share a count.
func withCounts(router *OpenAIRouter, ctx *RequestContext, counts map[string]int) {
	if ctx.SemanticRequest == nil {
		ctx.SemanticRequest = testNeutralRequest("vllm-sr/auto", strings.Repeat(englishSentence, 5_000))
	}
	if ctx.contextCounts == nil {
		ctx.contextCounts = map[contextCountKey]contextCount{}
	}
	for model, tokens := range counts {
		key := contextCountKey{
			tokenizer:  router.promptTokenizer(model).Name(),
			shape:      promptShape(ctx.VSRSelectedDecision),
			generation: ctx.SemanticRequest.Generation,
		}
		ctx.contextCounts[key] = contextCount{tokens: tokens, ok: true}
	}
}

func observeRouterEvents(t *testing.T) *observer.ObservedLogs {
	t.Helper()
	core, logs := observer.New(zapcore.InfoLevel)
	t.Cleanup(zap.ReplaceGlobals(zap.New(core)))
	return logs
}

// A prompt of 1,040,000 tokens is in the band of a 1,048,576-token window:
// the Router admits it and the backend decides. The one count taken before
// the signals serves the window check after the decision too.
func TestAPromptInTheBandOfTheWindowReachesTheBackend(t *testing.T) {
	router := longWindowRouterFor(t)
	logs := observeRouterEvents(t)
	record := runLongWindowCase(t, router, "english-1040000", "/v1/chat/completions",
		chatRequestBody(t, promptOfTokens(t, 1_040_000)))
	requireServedByLongModel(t, record)

	counted := logs.FilterMessage("context_tokens_counted").All()
	require.Len(t, counted, 1)
	require.Equal(t, tokencount.O200kBase, counted[0].ContextMap()["tokenizer"])
}

// A prompt of 1,100,000 tokens is beyond the band: it is refused before any
// signal runs, so no guard reads a window of it, with 400
// context_length_exceeded in the client's own protocol, naming the window
// and the estimated count (#6320).
func TestAPromptBeyondTheBandIsRefusedBeforeTheSignals(t *testing.T) {
	text := promptOfTokens(t, 1_100_000)
	estimate := routerCountOf(text)
	for _, tc := range []struct {
		name, path string
		body       map[string]any
	}{
		{"chat", "/v1/chat/completions", map[string]any{"messages": []map[string]string{{"role": "user", "content": text}}}},
		{"responses", "/v1/responses", map[string]any{"input": text}},
		{"messages", "/v1/messages", map[string]any{"max_tokens": 16, "messages": []map[string]string{{"role": "user", "content": text}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := longWindowRouterFor(t)
			logs := observeRouterEvents(t)
			tc.body["model"] = "vllm-sr/auto"
			body, err := json.Marshal(tc.body)
			require.NoError(t, err)

			started := time.Now()
			record := runLongWindowCase(t, router, "over-window-"+tc.name, tc.path, string(body))
			elapsed := time.Since(started)
			t.Logf("refused in %v", elapsed)

			require.Empty(t, record.Error)
			require.Nil(t, record.Upstream, "the request reached a backend")
			require.NotNil(t, record.Response)
			require.Equal(t, http.StatusBadRequest, record.Response.Status, "%s", record.Response.Body)
			require.Empty(t, logs.FilterMessage("signal_phase_timing").All(), "a signal ran on a request that cannot run")
			require.Len(t, logs.FilterMessage("context_tokens_counted").All(), 1)
			require.Less(t, elapsed, 5*time.Second)

			decoded := decodeClientError(t, record.Response.Body)
			require.Equal(t, "invalid_request_error", decoded.Error.Type)
			if tc.path == "/v1/messages" {
				require.Equal(t, "error", decoded.Type)
				require.Equal(t, fmt.Sprintf("prompt is too long: %d tokens > 1048576 maximum", estimate), decoded.Error.Message)
				return
			}
			require.Equal(t, "context_length_exceeded", *decoded.Error.Code)
			require.Equal(t, "messages", *decoded.Error.Param)
			require.Equal(t, fmt.Sprintf("This model's maximum context length is 1048576 tokens. However, your request has "+
				"an estimated %d input tokens. Please reduce the length of the input.", estimate), decoded.Error.Message)
		})
	}
}

// A prompt in the band reaches the backend; when the backend refuses it for
// its length, the client gets that refusal as context_length_exceeded.
func TestABackendRefusalOfAnInBandPromptIsContextLengthExceeded(t *testing.T) {
	router := longWindowRouterFor(t)
	record := runLongWindowCaseAgainst(t, router, "in-band-refused", chatRequestBody(t, promptOfTokens(t, 1_060_000)),
		&parity.Upstream{Status: http.StatusBadRequest, Headers: [][]string{{"content-type", "application/json"}}, Body: vllmContextLengthError})
	require.Empty(t, record.Error)
	require.NotNil(t, record.Upstream, "the band leaves the prompt to the backend")
	require.NotNil(t, record.Response)
	requireContextLengthExceeded(t, record.Response.Status, record.Response.Body)
}

// The margin is configurable: with none, the same 1,060,000 tokens are
// refused before the backend.
func TestTheBandFollowsTheConfiguredMargin(t *testing.T) {
	yaml := strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1) +
		"  router:\n    context_window:\n      margin: 0\n"
	router := longWindowRouter(t, yaml)
	record := runLongWindowCase(t, router, "no-margin", "/v1/chat/completions", chatRequestBody(t, promptOfTokens(t, 1_060_000)))
	require.Nil(t, record.Upstream)
	require.NotNil(t, record.Response)
	require.Equal(t, http.StatusBadRequest, record.Response.Status)
}

// A Router that learned a dense ratio from Chinese prose, JSON and logs still
// admits English of 1,040,000 tokens and refuses English of 1,100,000: the
// learned ratio has no say in either.
func TestTheRoutersCountDoesNotDriftWithDenseTraffic(t *testing.T) {
	router := longWindowRouterFor(t)
	recipe, ok := router.Config.RecipeByName("default")
	require.True(t, ok)
	for range 40 {
		ctx := &RequestContext{VSRContextTextBytes: 290_000}
		ctx.Routing.SelectRecipe(recipe)
		router.calibrateTokenEstimator(ctx, 100_000)
	}

	served := runLongWindowCase(t, router, "dense-then-1040000", "/v1/chat/completions",
		chatRequestBody(t, promptOfTokens(t, 1_040_000)))
	requireServedByLongModel(t, served)

	text := promptOfTokens(t, 1_100_000)
	refused := runLongWindowCase(t, router, "dense-then-1100000", "/v1/chat/completions", chatRequestBody(t, text))
	require.NotNil(t, refused.Response)
	require.Equal(t, http.StatusBadRequest, refused.Response.Status)
	require.Contains(t, decodeClientError(t, refused.Response.Body).Error.Message,
		fmt.Sprintf("your request has an estimated %d input tokens", routerCountOf(text)))
}

// A request whose bytes fit the window is never counted.
func TestAdmissionCountsOnlyWhenTheBytesCouldExceedTheWindow(t *testing.T) {
	router := longWindowRouterFor(t)
	logs := observeRouterEvents(t)
	record := runLongWindowCase(t, router, "short", "/v1/chat/completions",
		chatRequestBody(t, englishPromptOfTokens(200_000)))
	requireServedByLongModel(t, record)
	require.Empty(t, logs.FilterMessage("context_tokens_counted").All())
}

// A byte-level BPE tokenizer.json whose merges spell "hello".
const helloTokenizer = `{
  "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true},
  "decoder": {"type": "ByteLevel"},
  "model": {"type": "BPE", "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6, "hello": 7},
    "merges": ["h e", "l l", "he ll", "hell o"]}
}`

// Each model counts with the tokenizer it names, a built-in encoding or a
// tokenizer.json; a model that names none counts with the proxy tokenizer,
// o200k_base unless global.router.context_window.proxy_tokenizer says
// otherwise.
func TestModelsCountWithTheirTokenizerOrTheProxy(t *testing.T) {
	file := filepath.Join(t.TempDir(), "tokenizer.json")
	require.NoError(t, os.WriteFile(file, []byte(helloTokenizer), 0o600))
	text := "中文的文本，包括标点符号。 hello hello hello"
	request := testNeutralRequest("vllm-sr/auto", text)
	counts := map[string]int{}
	for _, name := range []string{tokencount.O200kBase, tokencount.Cl100kBase, file} {
		encoding, err := tokencount.Default().Load(context.Background(), name)
		require.NoError(t, err)
		counts[name] = countPromptTokens(encoding, request)
	}
	require.NotEqual(t, counts[tokencount.O200kBase], counts[tokencount.Cl100kBase])

	for _, tc := range []struct {
		name  string
		proxy string
		want  map[string]string
	}{
		{"default proxy", "", map[string]string{"plain": tokencount.O200kBase, "cl100k": tokencount.Cl100kBase, "file": file}},
		{"configured proxy", tokencount.Cl100kBase, map[string]string{"plain": tokencount.Cl100kBase, "file": file}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
					"plain":  {ContextWindowSize: 1_000},
					"cl100k": {ContextWindowSize: 1_000, Tokenizer: tokencount.Cl100kBase},
					"file":   {ContextWindowSize: 1_000, Tokenizer: file},
				}},
			}}
			router.Config.ContextWindow.ProxyTokenizer = tc.proxy
			for model, tokenizer := range tc.want {
				ctx := &RequestContext{SemanticRequest: request}
				tokens, ok := router.contextTokens(ctx, nil, model)
				require.True(t, ok)
				require.Equal(t, counts[tokenizer], tokens, "%s counts with %s", model, tokenizer)
			}
		})
	}
}

// A model whose Hugging Face tokenizer is not in the cache counts with the
// proxy until it arrives; an offline Router keeps the proxy.
func TestAModelWhoseTokenizerIsMissingCountsWithTheProxy(t *testing.T) {
	router := &OpenAIRouter{
		Config: &config.RouterConfig{
			BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
				"remote": {ContextWindowSize: 1_000, Tokenizer: "missing-org/missing-model"},
			}},
		},
		tokenizers: tokencount.NewRegistry(tokencount.Hub{CacheDir: t.TempDir(), Offline: true}),
	}
	require.Equal(t, tokencount.O200kBase, router.promptTokenizer("remote").Name())
	_, err := router.tokenizers.Load(t.Context(), "missing-org/missing-model")
	require.Error(t, err)
	require.Equal(t, tokencount.O200kBase, router.promptTokenizer("remote").Name())
}

// A decision that may dispatch a smaller prompt than the client sent defers
// the refusal to the window check after the decision.
func TestDecisionsThatMayShrinkThePromptDeferTheRefusal(t *testing.T) {
	systemPrompt := func(mode string) config.DecisionPlugin {
		return config.DecisionPlugin{Type: config.DecisionPluginSystemPrompt, Configuration: config.MustStructuredPayload(
			config.SystemPromptPluginConfig{SystemPrompt: "Be brief.", Mode: mode})}
	}
	for _, tc := range []struct {
		name     string
		decision config.Decision
		shrinks  bool
	}{
		{"no plugin", config.Decision{Name: "plain"}, false},
		{"system prompt inserted", config.Decision{Name: "insert", Plugins: []config.DecisionPlugin{
			systemPrompt(config.SystemPromptModeInsert),
		}}, false},
		{"system prompt replacing the client's", config.Decision{Name: "replace", Plugins: []config.DecisionPlugin{
			systemPrompt(config.SystemPromptModeReplace),
		}}, true},
		{"tools dropped", config.Decision{Name: "no-tools", Plugins: []config.DecisionPlugin{{
			Type: config.DecisionPluginTools, Configuration: config.MustStructuredPayload(config.ToolsPluginConfig{
				Enabled: true, Policy: &config.ToolsPolicyConfig{Mode: config.ToolsPolicyModeNone},
			}),
		}}}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.shrinks, decisionMayShrinkPrompt(&tc.decision))
		})
	}
}

// A decision that only inserts a system prompt is judged from the count of
// the prompt as sent: no second count when the bounds decide.
func TestAnInsertedSystemPromptIsJudgedOnTheCountAsSent(t *testing.T) {
	decision := &config.Decision{Name: "brief", Plugins: []config.DecisionPlugin{{
		Type: config.DecisionPluginSystemPrompt, Configuration: config.MustStructuredPayload(
			config.SystemPromptPluginConfig{SystemPrompt: "Be brief.", Mode: config.SystemPromptModeInsert}),
	}}}
	for _, tc := range []struct {
		name    string
		sent    int
		verdict contextWindowVerdict
	}{
		{"fits with the prompt added", 40_000, contextFits},
		{"beyond the band as sent", 60_000, contextExceeds},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := routeActionRouter(map[string]int{"model": 50_000})
			ctx := &RequestContext{}
			withCounts(router, ctx, map[string]int{"model": tc.sent})
			ctx.VSRSelectedDecision = decision

			verdict, tokens := router.contextWindowVerdict(ctx, decision, "model", 0)
			require.Equal(t, tc.verdict, verdict)
			require.Equal(t, tc.sent, tokens)
			require.Len(t, ctx.contextCounts, 1, "the decision's prompt was counted again")
		})
	}
}

func TestBandVerdicts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	for tokens, want := range map[int]contextWindowVerdict{
		969_999: contextFits, 970_000: contextFits, 970_001: contextInBand,
		1_000_000: contextInBand, 1_029_999: contextInBand, 1_030_000: contextExceeds,
	} {
		require.Equal(t, want, router.bandVerdict(1_000_000, tokens), "%d tokens", tokens)
	}
}

// A recipe with a decision that replaces the client's system prompt still
// refuses up front a request that is beyond the band without that prompt.
func TestAdmissionRefusesThroughADecisionThatDropsTheSystemPrompt(t *testing.T) {
	yaml := strings.Replace(strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1), "      - name: default_route\n",
		"      - name: default_route\n        plugins:\n          - type: system_prompt\n            configuration:\n"+
			"              system_prompt: Answer briefly.\n              mode: replace\n", 1)
	router := longWindowRouter(t, yaml)
	body, err := json.Marshal(map[string]any{
		"model": "vllm-sr/auto",
		"messages": []map[string]string{
			{"role": "system", "content": "You are a careful reader."},
			{"role": "user", "content": promptOfTokens(t, 1_100_000)},
		},
	})
	require.NoError(t, err)
	logs := observeRouterEvents(t)
	record := runLongWindowCase(t, router, "replaced-system-prompt", "/v1/chat/completions", string(body))
	require.NotNil(t, record.Response)
	require.Equal(t, http.StatusBadRequest, record.Response.Status, "%s", record.Response.Body)
	require.Empty(t, logs.FilterMessage("signal_phase_timing").All())
	require.Len(t, logs.FilterMessage("context_tokens_counted").All(), 1)

	shrink := promptShrink{instructions: true, tools: true}
	request := testNeutralRequest("vllm-sr/auto", "hello")
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "You are a careful reader."}}}}
	request.Tools = []llmprotocol.Tool{{Name: "lookup", Description: "Look a word up.", InputSchema: []byte(`{"type":"object"}`)}}
	require.Equal(t, len("You are a careful reader.")+len("lookup")+len("Look a word up.")+len(`{"type":"object"}`), shrink.removableBytes(request))
}

// A prompt with media is not counted: no text tokenizer knows its media's
// tokens, so the window check never refuses it.
func TestPromptsWithMediaAreNotCounted(t *testing.T) {
	request := testNeutralRequest("vllm-sr/auto", strings.Repeat(englishSentence, 100))
	request.Messages[0].Content = append(request.Messages[0].Content, llmprotocol.Content{
		Kind: llmprotocol.ContentImage, URL: "https://example.com/picture.png",
	})
	require.True(t, requestCarriesMedia(request))
	router := routeActionRouter(map[string]int{"model": 100})
	ctx := &RequestContext{SemanticRequest: request}
	_, ok := router.contextTokens(ctx, nil, "model")
	require.False(t, ok)
	verdict, _ := router.contextWindowVerdict(ctx, nil, "model", 0)
	require.Equal(t, contextFits, verdict)
}

// What the Router counts of a request: text, tool calls and results, tool
// definitions and the output schema, plus the template's reserves.
func TestCountPromptTokensCoversWhatATemplateRenders(t *testing.T) {
	proxy, err := tokencount.Builtin(tokencount.O200kBase)
	require.NoError(t, err)
	request := testNeutralRequest("vllm-sr/auto", "What is the weather?")
	request.Instructions = []llmprotocol.InstructionBlock{{Role: llmprotocol.RoleSystem, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Be brief."}}}}
	request.Messages = append(request.Messages, llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{
		Kind: llmprotocol.ContentToolCall, ToolCall: &llmprotocol.ToolCall{Name: "weather", Arguments: `{"city":"Paris"}`},
	}}})
	request.Tools = []llmprotocol.Tool{{Name: "weather", Description: "Look up the weather.", InputSchema: []byte(`{"type":"object"}`)}}

	text := "Be brief.\nWhat is the weather?\nweather\n{\"city\":\"Paris\"}\nweather\nLook up the weather.\n{\"type\":\"object\"}\n"
	reserve := contextTemplateReserve + 3*contextMessageReserve + contextToolCallReserve + contextToolDefinitionReserve
	require.Equal(t, proxy.Count(text)+reserve, countPromptTokens(proxy, request))
}

// A fallback candidate whose window the Router's count exceeds beyond the
// band is never called (#6322).
func TestFallbackSkipsACandidateTheRoutersCountRulesOut(t *testing.T) {
	ruleOut := func(router *OpenAIRouter, ctx *RequestContext) {
		params := router.Config.ModelConfig["model-fallback-1"]
		params.ContextWindowSize = 1_048_576
		router.Config.ModelConfig["model-fallback-1"] = params
		withCounts(router, ctx, map[string]int{"model-fallback-1": 1_213_758})
	}
	eligible := []string{"model-primary", "model-fallback-1", "model-fallback-2"}

	t.Run("native", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		ruleOut(session.router, session.ctx)
		step, err := session.Fallback().Next(t.Context(), failedWith(http.StatusServiceUnavailable, "unavailable"))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		require.Equal(t, "model-fallback-2", step.Call.Route)
		requireOneSkippedCandidate(t, logs, "model-fallback-1")
	})

	t.Run("response phase", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		ruleOut(router, ctx)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		var called []string
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			called = append(called, model)
			return []byte(anthropicFallbackSuccess), http.StatusOK, nil
		}
		resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.Equal(t, []string{"model-fallback-2"}, called)
		requireOneSkippedCandidate(t, logs, "model-fallback-1")
	})
}

const anthropicFallbackSuccess = `{"id":"msg_1","type":"message","role":"assistant","model":"model-fallback-2",` +
	`"content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`

// windowedFallbackRouter gives the primary a 131,072-token window, the first
// candidate a narrower one and the second a wider one.
func windowedFallbackRouter(router *OpenAIRouter) {
	for model, window := range map[string]int{"model-primary": 131_072, "model-fallback-1": 65_536, "model-fallback-2": 1_048_576} {
		params := router.Config.ModelConfig[model]
		params.ContextWindowSize = window
		router.Config.ModelConfig[model] = params
	}
}

// After a backend refuses the prompt for its length, fallback goes on, but
// only to candidates with wider windows: the narrower one is skipped.
func TestFallbackAfterALengthRefusalGoesOnlyToWiderWindows(t *testing.T) {
	eligible := []string{"model-primary", "model-fallback-1", "model-fallback-2"}

	t.Run("native, from the refusal's body", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), eligible...)
		windowedFallbackRouter(session.router)
		step, err := session.Fallback().Next(t.Context(), failedWith(http.StatusBadRequest, vllmContextLengthError))
		require.NoError(t, err)
		require.NotNil(t, step.Call)
		require.Equal(t, "model-fallback-2", step.Call.Route)
		requireOneSkippedCandidate(t, logs, "model-fallback-1")
		require.Equal(t, 131_072, session.ctx.fallbackMinWindow)
	})

	t.Run("response phase, a 400 to a prompt in the band", func(t *testing.T) {
		logs := observeSkippedCandidates(t)
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		windowedFallbackRouter(router)
		ctx := testFallbackRequestContext("model-primary", eligible)
		markContextInBand(ctx, "model-primary")
		ctx.UpstreamStatusCode = http.StatusBadRequest
		var called []string
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			called = append(called, model)
			return []byte(anthropicFallbackSuccess), http.StatusOK, nil
		}
		resp := router.maybeExecuteFallback(nil, ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.Equal(t, []string{"model-fallback-2"}, called)
		requireOneSkippedCandidate(t, logs, "model-fallback-1")
	})

	t.Run("response phase, a candidate's refusal moves on to a wider window", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		windowedFallbackRouter(router)
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		var called []string
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			called = append(called, model)
			if model == "model-fallback-1" {
				return []byte(vllmContextLengthError), http.StatusBadRequest, nil
			}
			return []byte(anthropicFallbackSuccess), http.StatusOK, nil
		}
		resp := router.maybeExecuteFallback(nil, ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.Equal(t, []string{"model-fallback-1", "model-fallback-2"}, called)
		require.Equal(t, 65_536, ctx.fallbackMinWindow)
	})

	t.Run("a 400 that is not about length does not fall back", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		windowedFallbackRouter(router)
		ctx := testFallbackRequestContext("model-primary", eligible)
		ctx.UpstreamStatusCode = http.StatusBadRequest
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			t.Fatalf("called %s", model)
			return nil, 0, nil
		}
		require.Nil(t, router.maybeExecuteFallback(nil, ctx))
		require.Zero(t, ctx.fallbackMinWindow)
	})

	t.Run("no wider candidate leaves the backend's refusal", func(t *testing.T) {
		session := nativeFallbackSession(t, fallback.DefaultEnabledPolicy(), "model-primary", "model-fallback-1")
		windowedFallbackRouter(session.router)
		step, err := session.Fallback().Next(t.Context(), failedWith(http.StatusBadRequest, vllmContextLengthError))
		require.NoError(t, err)
		require.Nil(t, step.Call)
		require.Nil(t, step.Immediate, "the primary's own refusal goes to the client")
	})
}
