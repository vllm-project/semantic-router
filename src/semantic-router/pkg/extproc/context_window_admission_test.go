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

// longWindowRouterFor serves longWindowConfig, whose model names no
// tokenizer: its prompts count with the proxy tokenizer and its wide band.
func longWindowRouterFor(t *testing.T) *OpenAIRouter {
	t.Helper()
	return longWindowRouter(t, strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1))
}

// namedTokenizerConfig is longWindowConfig with the model naming its
// tokenizer, o200k_base, so that its prompts count exactly, with the narrow
// band.
func namedTokenizerConfig() string {
	return strings.Replace(strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1),
		"      context_window_size: 1048576\n", "      context_window_size: 1048576\n      tokenizer: o200k_base\n", 1)
}

func namedTokenizerRouter(t *testing.T) *OpenAIRouter {
	t.Helper()
	return longWindowRouter(t, namedTokenizerConfig())
}

// requireWindowChecked asserts the one context_window_checked event of a
// request: the kind of count, its band and the verdict.
func requireWindowChecked(t *testing.T, logs *observer.ObservedLogs, count string, margin float64, verdict string) {
	t.Helper()
	checked := logs.FilterMessage("context_window_checked").All()
	require.NotEmpty(t, checked)
	fields := checked[0].ContextMap()
	require.Equal(t, count, fields["count"])
	require.InDelta(t, margin, fields["margin"], 1e-9)
	require.Equal(t, verdict, fields["verdict"])
}

func refusalCount(t *testing.T, count string) float64 {
	t.Helper()
	return counterValue(t, "llm_context_window_refusals_total", map[string]string{"count": count})
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
			tokenizer:  router.promptTokenizer(model).encoding.Name(),
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

// A prompt of 1,040,000 tokens is in the band of a 1,048,576-token window,
// counted exactly or with the proxy: the Router admits it and the backend
// decides. The one count taken before the signals serves the window check
// after the decision too, and the log says which count and band decided.
func TestAPromptInTheBandOfTheWindowReachesTheBackend(t *testing.T) {
	for _, tc := range []struct {
		name   string
		router func(*testing.T) *OpenAIRouter
		count  string
		margin float64
	}{
		{"named tokenizer", namedTokenizerRouter, "exact", config.DefaultContextWindowMargin},
		{"proxy", longWindowRouterFor, "proxy", config.DefaultContextWindowProxyMargin},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := tc.router(t)
			logs := observeRouterEvents(t)
			record := runLongWindowCase(t, router, "english-1040000", "/v1/chat/completions",
				chatRequestBody(t, promptOfTokens(t, 1_040_000)))
			requireServedByLongModel(t, record)

			counted := logs.FilterMessage("context_tokens_counted").All()
			require.Len(t, counted, 1)
			require.Equal(t, tokencount.O200kBase, counted[0].ContextMap()["tokenizer"])
			requireWindowChecked(t, logs, tc.count, tc.margin, "in_band")
		})
	}
}

// A model that names its tokenizer refuses a prompt of 1,100,000 tokens,
// beyond W x 1.03: before any signal runs, so no guard reads a window of it,
// with 400 context_length_exceeded in the client's own protocol, naming the
// window and the estimated count (#6320).
func TestANamedTokenizerRefusesBeyondTheNarrowBandBeforeTheSignals(t *testing.T) {
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
			router := namedTokenizerRouter(t)
			logs := observeRouterEvents(t)
			refusals := refusalCount(t, "exact")
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
			requireWindowChecked(t, logs, "exact", config.DefaultContextWindowMargin, "exceeds")
			require.InDelta(t, refusals+1, refusalCount(t, "exact"), 1e-9)
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

// A model without a tokenizer of its own counts with the proxy, which can
// overcount its real tokens by a quarter (Chinese text), so it refuses before
// the signals only beyond W x 1.3: 1,100,000 tokens reach the backend, and
// 1,400,000 are refused at once.
func TestAProxyCountRefusesOnlyBeyondTheWideBand(t *testing.T) {
	router := longWindowRouterFor(t)
	logs := observeRouterEvents(t)
	served := runLongWindowCase(t, router, "proxy-1100000", "/v1/chat/completions", chatRequestBody(t, promptOfTokens(t, 1_100_000)))
	requireServedByLongModel(t, served)
	requireWindowChecked(t, logs, "proxy", config.DefaultContextWindowProxyMargin, "in_band")

	logs = observeRouterEvents(t)
	refusals := refusalCount(t, "proxy")
	text := promptOfTokens(t, 1_400_000)
	refused := runLongWindowCase(t, router, "proxy-1400000", "/v1/chat/completions", chatRequestBody(t, text))
	require.Nil(t, refused.Upstream, "the request reached a backend")
	require.NotNil(t, refused.Response)
	require.Equal(t, http.StatusBadRequest, refused.Response.Status, "%s", refused.Response.Body)
	require.Empty(t, logs.FilterMessage("signal_phase_timing").All())
	require.Contains(t, decodeClientError(t, refused.Response.Body).Error.Message,
		fmt.Sprintf("your request has an estimated %d input tokens", routerCountOf(text)))
	requireWindowChecked(t, logs, "proxy", config.DefaultContextWindowProxyMargin, "exceeds")
	require.InDelta(t, refusals+1, refusalCount(t, "proxy"), 1e-9)
}

// A prompt in the band reaches the backend; when the backend refuses it for
// its length, the client gets that refusal as context_length_exceeded.
func TestABackendRefusalOfAnInBandPromptIsContextLengthExceeded(t *testing.T) {
	for _, tc := range []struct {
		name   string
		router func(*testing.T) *OpenAIRouter
		tokens int
	}{
		{"named tokenizer", namedTokenizerRouter, 1_060_000},
		{"proxy", longWindowRouterFor, 1_200_000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			record := runLongWindowCaseAgainst(t, tc.router(t), "in-band-refused", chatRequestBody(t, promptOfTokens(t, tc.tokens)),
				&parity.Upstream{Status: http.StatusBadRequest, Headers: [][]string{{"content-type", "application/json"}}, Body: vllmContextLengthError})
			require.Empty(t, record.Error)
			require.NotNil(t, record.Upstream, "the band leaves the prompt to the backend")
			require.NotNil(t, record.Response)
			requireContextLengthExceeded(t, record.Response.Status, record.Response.Body)
		})
	}
}

// Both bands are configurable: with margin 0 a named tokenizer refuses
// 1,060,000 tokens before the backend, and with proxy_margin 0.05 the proxy
// refuses 1,150,000, which the default proxy band leaves to the backend.
func TestTheBandsFollowTheConfiguredMargins(t *testing.T) {
	for _, tc := range []struct {
		name   string
		yaml   string
		tokens int
	}{
		{"margin 0", namedTokenizerConfig() + "  router:\n    context_window:\n      margin: 0\n", 1_060_000},
		{"proxy_margin 0.05", strings.Replace(longWindowConfig, "BACKEND", "127.0.0.1:18100", 1) +
			"  router:\n    context_window:\n      proxy_margin: 0.05\n", 1_150_000},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := longWindowRouter(t, tc.yaml)
			record := runLongWindowCase(t, router, "configured-band", "/v1/chat/completions", chatRequestBody(t, promptOfTokens(t, tc.tokens)))
			require.Nil(t, record.Upstream)
			require.NotNil(t, record.Response)
			require.Equal(t, http.StatusBadRequest, record.Response.Status)
		})
	}
}

// A Router that learned a dense ratio from Chinese prose, JSON and logs still
// admits English of 1,040,000 tokens and refuses English of 1,100,000: the
// learned ratio has no say in either.
func TestTheRoutersCountDoesNotDriftWithDenseTraffic(t *testing.T) {
	router := namedTokenizerRouter(t)
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

// Each model counts exactly with the tokenizer it names: a built-in encoding,
// a tokenizer.json, or a directory that holds one, as a mounted volume or an
// image directory does. A model that names none counts with the proxy
// tokenizer, o200k_base unless global.router.context_window.proxy_tokenizer
// says otherwise.
func TestModelsCountWithTheirTokenizerOrTheProxy(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "hello-model")
	require.NoError(t, os.MkdirAll(dir, 0o750))
	file := filepath.Join(dir, "tokenizer.json")
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

	type counted struct {
		tokenizer string
		kind      contextCountKind
	}
	for _, tc := range []struct {
		name  string
		proxy string
		want  map[string]counted
	}{
		{"default proxy", "", map[string]counted{
			"plain":  {tokencount.O200kBase, contextCountProxy},
			"cl100k": {tokencount.Cl100kBase, contextCountExact},
			"file":   {file, contextCountExact},
			"dir":    {file, contextCountExact},
		}},
		{"configured proxy", tokencount.Cl100kBase, map[string]counted{
			"plain": {tokencount.Cl100kBase, contextCountProxy},
			"dir":   {file, contextCountExact},
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{
				BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
					"plain":  {ContextWindowSize: 1_000},
					"cl100k": {ContextWindowSize: 1_000, Tokenizer: tokencount.Cl100kBase},
					"file":   {ContextWindowSize: 1_000, Tokenizer: file},
					"dir":    {ContextWindowSize: 1_000, Tokenizer: dir},
				}},
			}}
			router.Config.ContextWindow.ProxyTokenizer = tc.proxy
			for model, want := range tc.want {
				ctx := &RequestContext{SemanticRequest: request}
				tokens, kind, ok := router.contextTokens(ctx, nil, model)
				require.True(t, ok)
				require.Equal(t, counts[want.tokenizer], tokens, "%s counts with %s", model, want.tokenizer)
				require.Equal(t, want.kind, kind, "%s", model)
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
	for range 2 {
		tokenizer := router.promptTokenizer("remote")
		require.Equal(t, tokencount.O200kBase, tokenizer.encoding.Name())
		require.Equal(t, contextCountProxy, tokenizer.kind)
		_, err := router.tokenizers.Load(t.Context(), "missing-org/missing-model")
		require.Error(t, err)
	}
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
			params := router.Config.ModelConfig["model"]
			params.Tokenizer = tokencount.O200kBase
			router.Config.ModelConfig["model"] = params
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

// The band around a window is 3% for a count with the model's own tokenizer
// and 30% for a proxy count.
func TestBandVerdicts(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{}}
	for _, tc := range []struct {
		kind contextCountKind
		want map[int]contextWindowVerdict
	}{
		{contextCountExact, map[int]contextWindowVerdict{
			969_999: contextFits, 970_000: contextFits, 970_001: contextInBand,
			1_000_000: contextInBand, 1_029_999: contextInBand, 1_030_000: contextExceeds,
		}},
		{contextCountProxy, map[int]contextWindowVerdict{
			699_999: contextFits, 700_000: contextFits, 700_001: contextInBand, 1_030_000: contextInBand,
			1_299_999: contextInBand, 1_300_000: contextExceeds,
		}},
	} {
		for tokens, want := range tc.want {
			require.Equal(t, want, router.bandVerdict(1_000_000, tokens, tc.kind), "%s count of %d tokens", tc.kind, tokens)
		}
	}
}

// Each model's window check is logged and counted once per request and
// prompt shape, with the kind of count and the band that decided it.
func TestTheWindowCheckIsVisibleInMetricsAndLogs(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{
		BackendModels: config.BackendModels{ModelConfig: map[string]config.ModelParams{
			"named": {ContextWindowSize: 1_000, Tokenizer: tokencount.O200kBase},
			"proxy": {ContextWindowSize: 1_000},
		}},
	}}
	text := promptOfTokens(t, 1_100)
	ctx := &RequestContext{RequestID: "req-window", SemanticRequest: testNeutralRequest("vllm-sr/auto", text)}
	checks := func(count, verdict string) float64 {
		return counterValue(t, "llm_context_window_checks_total", map[string]string{"count": count, "verdict": verdict})
	}
	exactExceeds, proxyInBand := checks("exact", "exceeds"), checks("proxy", "in_band")
	logs := observeRouterEvents(t)

	for range 2 {
		verdict, tokens := router.contextWindowVerdict(ctx, nil, "named", 0)
		require.Equal(t, contextExceeds, verdict)
		require.Equal(t, routerCountOf(text), tokens)
		verdict, _ = router.contextWindowVerdict(ctx, nil, "proxy", 0)
		require.Equal(t, contextInBand, verdict)
	}

	require.InDelta(t, exactExceeds+1, checks("exact", "exceeds"), 1e-9)
	require.InDelta(t, proxyInBand+1, checks("proxy", "in_band"), 1e-9)
	events := logs.FilterMessage("context_window_checked").All()
	require.Len(t, events, 2)
	byModel := map[string]map[string]interface{}{}
	for _, event := range events {
		fields := event.ContextMap()
		byModel[fields["model"].(string)] = fields
	}
	require.Equal(t, "exact", byModel["named"]["count"])
	require.InDelta(t, 0.03, byModel["named"]["margin"], 1e-9)
	require.Equal(t, "exceeds", byModel["named"]["verdict"])
	require.Equal(t, "proxy", byModel["proxy"]["count"])
	require.InDelta(t, 0.30, byModel["proxy"]["margin"], 1e-9)
	require.Equal(t, "in_band", byModel["proxy"]["verdict"])
}

// A recipe with a decision that replaces the client's system prompt still
// refuses up front a request that is beyond the band without that prompt.
func TestAdmissionRefusesThroughADecisionThatDropsTheSystemPrompt(t *testing.T) {
	yaml := strings.Replace(namedTokenizerConfig(), "      - name: default_route\n",
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
	_, _, ok := router.contextTokens(ctx, nil, "model")
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
// band is never called (#6322). The candidate names its tokenizer, so the
// narrow band applies; on a proxy count the same 1.16 x W is in its band.
func TestFallbackSkipsACandidateTheRoutersCountRulesOut(t *testing.T) {
	ruleOut := func(router *OpenAIRouter, ctx *RequestContext) {
		params := router.Config.ModelConfig["model-fallback-1"]
		params.ContextWindowSize = 1_048_576
		params.Tokenizer = tokencount.O200kBase
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

	t.Run("a proxy count in its band is no reason to skip", func(t *testing.T) {
		router, _ := setupFallbackTestRouter(t, fallback.DefaultEnabledPolicy())
		ctx := testFallbackRequestContext("model-primary", eligible)
		params := router.Config.ModelConfig["model-fallback-1"]
		params.ContextWindowSize = 1_048_576
		router.Config.ModelConfig["model-fallback-1"] = params
		withCounts(router, ctx, map[string]int{"model-fallback-1": 1_213_758})
		ctx.UpstreamStatusCode = http.StatusServiceUnavailable
		var called []string
		router.fallbackCaller = func(_ context.Context, model string, _ []byte, _ map[string]string) ([]byte, int, error) {
			called = append(called, model)
			return []byte(strings.Replace(servedChatCompletion, "long-model", model, 1)), http.StatusOK, nil
		}
		resp := router.handleUpstreamTransportError([]byte("upstream 503"), ctx)
		require.NotNil(t, resp.GetImmediateResponse())
		require.Equal(t, []string{"model-fallback-1"}, called)
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
