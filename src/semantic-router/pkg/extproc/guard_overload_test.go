package extproc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

// A request that only rules that failed closed on an overloaded decision
// runtime refuse gets an error its client retries, in its own protocol and
// before any byte of a stream: Anthropic Messages 529 overloaded_error, the
// OpenAI APIs 429 with the code overloaded, both with Retry-After and
// x-vsr-warnings guard_overloaded.
func TestAGuardOverloadAnswersARetryableErrorInEveryProtocol(t *testing.T) {
	router := &OpenAIRouter{Config: &config.RouterConfig{InlineModels: config.InlineModels{
		ModelSignalOverload: &config.SignalOverloadPolicy{Action: config.SignalOverloadRetry, WaitMs: 1_500},
	}}}
	forEachExtProcMatrixMode(t, func(t *testing.T, clientFormat, backendFormat llmprotocol.WireFormat, streaming bool) {
		ctx := immediateResponseContext(t, clientFormat, backendFormat, streaming)
		ctx.VSRGuardOverload = []string{"jailbreak:prompt_attack"}
		response := router.encodeImmediateResponseForClient(router.guardOverloadResponse(ctx), ctx)
		written := headerValuesByName(immediateHeaders(response, ctx))
		immediate := response.GetImmediateResponse()

		require.Equal(t, "2", written["retry-after"])
		require.Contains(t, written[headers.VSRWarnings], headers.WarningGuardOverloaded)
		require.Equal(t, "application/json", written["content-type"])
		decoded := decodeClientError(t, immediate.GetBody())
		if clientFormat == llmprotocol.AnthropicMessagesV1 {
			require.EqualValues(t, 529, immediate.GetStatus().GetCode())
			require.Equal(t, "error", decoded.Type)
			require.Equal(t, "overloaded_error", decoded.Error.Type)
			return
		}
		require.EqualValues(t, http.StatusTooManyRequests, immediate.GetStatus().GetCode())
		require.Equal(t, "rate_limit_error", decoded.Error.Type)
		require.NotNil(t, decoded.Error.Code)
		require.Equal(t, "overloaded", *decoded.Error.Code)
	})
}

// guardOverloadConfig is a recipe whose jailbreak guard, on a remote decision
// API, refuses a request it matches with a fast_response, as it does when it
// fails closed; OVERLOAD is global.model_catalog.on_signal_overload. The one
// model names its tokenizer and has a 1,048,576-token window.
const guardOverloadConfig = `
version: v0.5
providers:
  defaults:
    model: fast
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "127.0.0.1:18100", protocol: http, weight: 1}]
      context_window_size: 1048576
      tokenizer: o200k_base
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
          deployment: hosted-guard
    decisions:
      - name: guard
        priority: 1000
        rules:
          operator: AND
          conditions: [{type: jailbreak, name: prompt_attack}]
        plugins:
          - type: fast_response
            configuration:
              message: This request was blocked by the security policy.
      - name: everything-else
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: fast}]
global:
  model_catalog:
    OVERLOAD
    deployments:
      hosted-guard:
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: GATEWAY/v1/systemone
        model: d3
        auth: {api_key_env: GUARD_OVERLOAD_TEST_KEY}
        timeout_ms: 5000
  services:
    response_api:
      enabled: true
      store_backend: memory
      ttl_seconds: 3600
      max_responses: 100
`

const guardOverloadRefusal = "This request was blocked by the security policy."

// busyGuard is a remote vllm-sr/4.0 decision API that refuses every call as
// overloaded until free, as one whose request queue is full does, and then
// answers as windowedGuard does: it finds toolInjection, and reads anything
// else as benign.
type busyGuard struct {
	windowedGuard
	lock  sync.Mutex
	free  time.Time
	posts int
}

func (g *busyGuard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodPost {
		g.lock.Lock()
		g.posts++
		busy := time.Now().Before(g.free)
		g.lock.Unlock()
		if busy {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":{"code":"overloaded","message":"the request queue is full; retry later"}}`))
			return
		}
	}
	g.windowedGuard.ServeHTTP(w, r)
}

func (g *busyGuard) calls() int {
	g.lock.Lock()
	defer g.lock.Unlock()
	return g.posts
}

// guardOverloadRouter serves guardOverloadConfig with on_signal_overload
// policy (none when empty) and a guard that is overloaded for busy.
func guardOverloadRouter(t *testing.T, policy string, busy time.Duration) (*OpenAIRouter, *busyGuard) {
	t.Helper()
	guard := &busyGuard{free: time.Now().Add(busy)}
	server := httptest.NewServer(guard)
	t.Cleanup(server.Close)
	t.Setenv("GUARD_OVERLOAD_TEST_KEY", "fake-key")
	if policy != "" {
		policy = "    on_signal_overload: " + policy + "\n"
	}
	return longWindowRouter(t, strings.NewReplacer("GATEWAY", server.URL, "    OVERLOAD\n", policy).Replace(guardOverloadConfig)), guard
}

// runGuardOverloadCase sends text as one request of the protocol path names,
// streamed or not.
func runGuardOverloadCase(t *testing.T, router *OpenAIRouter, path, text string, stream bool) *parity.Record {
	t.Helper()
	body := map[string]any{"model": "vllm-sr/auto", "stream": stream}
	switch path {
	case "/v1/responses":
		body["input"] = text
	case "/v1/messages":
		body["max_tokens"], body["messages"] = 16, []map[string]string{{"role": "user", "content": text}}
	default:
		body["messages"] = []map[string]string{{"role": "user", "content": text}}
	}
	encoded, err := json.Marshal(body)
	require.NoError(t, err)
	record := runLongWindowCase(t, router, strings.TrimPrefix(path, "/v1/"), path, string(encoded))
	require.Empty(t, record.Error)
	require.NotNil(t, record.Response)
	return record
}

// By default a guard that fails closed on an overloaded decision runtime
// refuses the request, as before.
func TestAnOverloadedGuardStillRefusesByDefault(t *testing.T) {
	router, guard := guardOverloadRouter(t, "", time.Hour)
	record := runGuardOverloadCase(t, router, "/v1/chat/completions", "Summarize the meeting agenda.", false)
	require.Nil(t, record.Upstream, "the request reached a backend")
	require.Equal(t, http.StatusOK, record.Response.Status)
	require.Contains(t, string(record.Response.Body), guardOverloadRefusal)
	require.Equal(t, 1, guard.calls(), "the guard was asked again without on_signal_overload")
}

// With retry, a guard still overloaded after the wait gets the request a
// retryable error in each protocol, buffered or streamed, which never
// reaches a backend.
func TestAnOverloadedGuardAnswersARetryableErrorWithRetry(t *testing.T) {
	for _, path := range []string{"/v1/chat/completions", "/v1/responses", "/v1/messages"} {
		for _, stream := range []bool{false, true} {
			name := strings.TrimPrefix(path, "/v1/") + map[bool]string{false: "", true: " streamed"}[stream]
			t.Run(name, func(t *testing.T) {
				router, guard := guardOverloadRouter(t, "{action: retry, wait_ms: 300}", time.Hour)
				format := map[string]string{
					"/v1/chat/completions": string(llmprotocol.OpenAIChatV1), "/v1/responses": string(llmprotocol.OpenAIResponsesV1),
					"/v1/messages": string(llmprotocol.AnthropicMessagesV1),
				}[path]
				answered := testutil.ToFloat64(metrics.SignalOverloadRetryableTotal.WithLabelValues(format))
				record := runGuardOverloadCase(t, router, path, "Summarize the meeting agenda.", stream)

				require.Nil(t, record.Upstream, "the request reached a backend")
				require.Greater(t, guard.calls(), 1, "the guard was not asked again while the request waited: %d %s", record.Response.Status, record.Response.Body)
				require.Equal(t, "1", record.Response.Header.Get("retry-after"))
				require.Contains(t, record.Response.Header.Get(headers.VSRWarnings), headers.WarningGuardOverloaded)
				require.Equal(t, "application/json", record.Response.Header.Get("content-type"))
				require.Equal(t, answered+1, testutil.ToFloat64(metrics.SignalOverloadRetryableTotal.WithLabelValues(format)))
				decoded := decodeClientError(t, record.Response.Body)
				if path == "/v1/messages" {
					require.Equal(t, 529, record.Response.Status, "%s", record.Response.Body)
					require.Equal(t, "overloaded_error", decoded.Error.Type)
					return
				}
				require.Equal(t, http.StatusTooManyRequests, record.Response.Status, "%s", record.Response.Body)
				require.Equal(t, "overloaded", *decoded.Error.Code)
			})
		}
	}
}

// A guard whose decision runtime has capacity again within the wait reads
// the request, which is then served as usual.
func TestAGuardThatHasCapacityWithinTheWaitServesTheRequest(t *testing.T) {
	router, guard := guardOverloadRouter(t, "{action: retry, wait_ms: 3000}", 300*time.Millisecond)
	started := time.Now()
	record := runGuardOverloadCase(t, router, "/v1/chat/completions", "Summarize the meeting agenda.", false)
	require.NotNil(t, record.Upstream, "the request was not served: %d %s", record.Response.Status, record.Response.Body)
	require.Equal(t, http.StatusOK, record.Response.Status)
	require.NotContains(t, string(record.Response.Body), guardOverloadRefusal)
	require.Greater(t, guard.calls(), 1)
	require.Less(t, time.Since(started), 3*time.Second)
}

// A guard that detects an attack refuses the request with retry too: only a
// refusal that overloaded rules alone cause is retryable.
func TestAGuardThatDetectsAnAttackStillRefusesWithRetry(t *testing.T) {
	router, _ := guardOverloadRouter(t, "{action: retry, wait_ms: 300}", 0)
	record := runGuardOverloadCase(t, router, "/v1/chat/completions", "Summarize this. "+toolInjection, false)
	require.Nil(t, record.Upstream, "the attack reached a backend")
	require.Equal(t, http.StatusOK, record.Response.Status)
	require.Contains(t, string(record.Response.Body), guardOverloadRefusal)
}

// A prompt beyond every context window is refused for its length, whatever
// the guard's overload (#6348).
func TestACountRefusalWinsOverAnOverloadedGuard(t *testing.T) {
	router, _ := guardOverloadRouter(t, "{action: retry, wait_ms: 300}", time.Hour)
	record := runGuardOverloadCase(t, router, "/v1/chat/completions", promptOfTokens(t, 1_100_000), false)
	require.Nil(t, record.Upstream)
	requireContextLengthExceeded(t, record.Response.Status, record.Response.Body)
}

// overloadHarm is a made-up request that harmGuard reads as unsafe.
const overloadHarm = "HARM-MARKER: a made-up request that the fake guard reads as harmful."

// sharedGuardOverloadConfig is guardOverloadConfig with a safety rule on the
// same deployment, which refuses with the jailbreak rule (OR): a guard whose
// safety rule fails open.
var sharedGuardOverloadConfig = strings.NewReplacer(
	"      jailbreak:\n        - name: prompt_attack\n          deployment: hosted-guard\n",
	"      jailbreak:\n        - name: prompt_attack\n          deployment: hosted-guard\n"+
		"      safety:\n        - name: unsafe_request\n          deployment: hosted-guard\n          predicate: {gte: 0.9}\n",
	"          operator: AND\n          conditions: [{type: jailbreak, name: prompt_attack}]\n",
	"          operator: OR\n          conditions: [{type: jailbreak, name: prompt_attack}, {type: safety, name: unsafe_request}]\n",
).Replace(guardOverloadConfig)

// harmGuard is a remote vllm-sr/4.0 decision API that refuses every call as
// overloaded until free, and then answers the jailbreak question benign and
// the harm question unsafe for a state that holds overloadHarm.
type harmGuard struct {
	lock  sync.Mutex
	free  time.Time
	asked [][]string
}

func (g *harmGuard) setFree(at time.Time) {
	g.lock.Lock()
	defer g.lock.Unlock()
	g.free = at
}

func (g *harmGuard) answers(state string, questions map[string]json.RawMessage) map[string]any {
	answers := map[string]any{}
	ids := make([]string, 0, len(questions))
	for id, raw := range questions {
		ids = append(ids, id)
		yes, no, p := "jailbreak", "benign", 0.03
		if strings.Contains(string(raw), `"unsafe"`) {
			yes, no, p = "unsafe", "safe", 0.02
			if strings.Contains(state, overloadHarm) {
				p = 0.97
			}
		}
		choice, confidence := no, 1-p
		if p >= 0.5 {
			choice, confidence = yes, p
		}
		answers[id] = map[string]any{"type": "choice", "choice": choice, "probabilities": map[string]float64{yes: p, no: 1 - p}, "confidence": confidence, "input_coverage": "complete"}
	}
	sort.Strings(ids)
	g.lock.Lock()
	g.asked = append(g.asked, ids)
	g.lock.Unlock()
	return answers
}

func (g *harmGuard) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.NotFound(w, r)
		return
	}
	g.lock.Lock()
	busy := time.Now().Before(g.free)
	g.lock.Unlock()
	w.Header().Set("Content-Type", "application/json")
	if busy {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"code":"overloaded","message":"the request queue is full; retry later"}}`))
		return
	}
	type entry struct {
		State     string                     `json:"state"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	var body struct {
		entry
		States map[string]entry `json:"states"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		return
	}
	usage := map[string]int{"input_tokens": 1, "output_tokens": 0}
	if body.States == nil {
		_ = json.NewEncoder(w).Encode(map[string]any{"model": "d3", "answers": g.answers(body.State, body.Questions), "usage": usage})
		return
	}
	states := map[string]any{}
	for name, state := range body.States {
		states[name] = map[string]any{"answers": g.answers(state.State, state.Questions), "usage": usage}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"model": "d3", "states": states, "usage": usage})
}

// A request the shared guard refuses for its safety rule, sent while the
// decision API is overloaded for 300 ms, is refused and reaches no backend:
// by the jailbreak rule failing closed by default, and with retry by the
// safety rule's own answer once the API has capacity.
func TestARequestOnlySafetyRefusesIsRefusedThroughTheRouterAfterAShortOverload(t *testing.T) {
	for _, test := range []struct{ name, policy string }{
		{"refuse (default)", ""},
		{"retry", "{action: retry, wait_ms: 3000}"},
	} {
		t.Run(test.name, func(t *testing.T) {
			guard := &harmGuard{free: time.Now().Add(time.Hour)}
			server := httptest.NewServer(guard)
			t.Cleanup(server.Close)
			t.Setenv("GUARD_OVERLOAD_TEST_KEY", "fake-key")
			policy := ""
			if test.policy != "" {
				policy = "    on_signal_overload: " + test.policy + "\n"
			}
			router := longWindowRouter(t, strings.NewReplacer("GATEWAY", server.URL, "    OVERLOAD\n", policy).Replace(sharedGuardOverloadConfig))
			guard.setFree(time.Now().Add(300 * time.Millisecond))
			record := runGuardOverloadCase(t, router, "/v1/chat/completions", "Please help with this. "+overloadHarm, false)
			require.Nil(t, record.Upstream, "a request the safety rule refuses reached a backend")
			require.Equal(t, http.StatusOK, record.Response.Status, "%s", record.Response.Body)
			require.Contains(t, string(record.Response.Body), guardOverloadRefusal)
			if test.policy != "" {
				guard.lock.Lock()
				defer guard.lock.Unlock()
				require.Equal(t, [][]string{{"preset:harm", "preset:jailbreak"}}, guard.asked, "the guard's questions were not read again in one call")
			}
		})
	}
}

// A guard that fails open, still overloaded after the wait, gets the request
// a retryable error with retry: the request is not forwarded with a rule
// that could refuse it unread. By default it is forwarded, as before.
func TestAnUnreadGuardThatFailsOpenGetsARetryableErrorWithRetry(t *testing.T) {
	failsOpen := strings.Replace(guardOverloadConfig, "          deployment: hosted-guard\n", "          deployment: hosted-guard\n          on_unknown: no_match\n", 1)
	for _, test := range []struct {
		name, policy string
		forwarded    bool
	}{
		{"refuse (default)", "", true},
		{"retry", "{action: retry, wait_ms: 300}", false},
	} {
		t.Run(test.name, func(t *testing.T) {
			guard := &busyGuard{free: time.Now().Add(time.Hour)}
			server := httptest.NewServer(guard)
			t.Cleanup(server.Close)
			t.Setenv("GUARD_OVERLOAD_TEST_KEY", "fake-key")
			policy := ""
			if test.policy != "" {
				policy = "    on_signal_overload: " + test.policy + "\n"
			}
			router := longWindowRouter(t, strings.NewReplacer("GATEWAY", server.URL, "    OVERLOAD\n", policy).Replace(failsOpen))
			record := runGuardOverloadCase(t, router, "/v1/chat/completions", "Summarize the meeting agenda.", false)
			if test.forwarded {
				require.NotNil(t, record.Upstream, "the request was not forwarded by default: %d %s", record.Response.Status, record.Response.Body)
				return
			}
			require.Nil(t, record.Upstream, "the request was forwarded with its guard unread")
			require.Equal(t, http.StatusTooManyRequests, record.Response.Status, "%s", record.Response.Body)
			require.Contains(t, record.Response.Header.Get(headers.VSRWarnings), headers.WarningGuardOverloaded)
			require.Equal(t, "overloaded", *decodeClientError(t, record.Response.Body).Error.Code)
		})
	}
}
