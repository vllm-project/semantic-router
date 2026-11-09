package extproc

import (
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// guardCacheConfig puts an exact response cache on a decision whose guard
// blocks the answer to a __refuse__ prompt and writes a note into the answer
// to a __note__ prompt. The rules read request-stage keywords, so no decision
// model is needed to decide them.
const guardCacheConfig = `
version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s
providers:
  defaults:
    model: guarded-model
  models:
    - name: guarded-model
      provider_model_id: guarded-model-instruct
      api_format: openai
      backend_refs:
        - name: guarded-primary
          endpoint: 127.0.0.1:18000
          protocol: http
          provider: vllm
recipes:
  - name: default
    signals:
      keywords:
        - name: guarded_route
          operator: OR
          keywords: ["__guarded__"]
        - name: refuse_answer
          operator: OR
          keywords: ["__refuse__"]
        - name: note_answer
          operator: OR
          keywords: ["__note__"]
    decisions:
      - name: guarded
        priority: 10
        rules:
          operator: AND
          conditions:
            - type: keyword
              name: guarded_route
        modelRefs:
          - model: guarded-model
            use_reasoning: false
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              mode: exact
              scope: global
          - type: response_guard
            configuration:
              rules:
                - when: {type: keyword, name: refuse_answer}
                  action: block
                - when: {type: keyword, name: note_answer}
                  action: annotate
                  details: true
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
global:
  services:
    response_api:
      enabled: true
      store_backend: memory
      ttl_seconds: 3600
      max_responses: 100
  stores:
    response_cache:
      backend_type: memory
      max_entries: 100
      ttl_seconds: 3600
`

const responseGuardNote = "[Response Warning] A response check flagged this response."

// A cache hit runs no guard, so the cache keeps only what the guard let a
// client receive: a refused answer stays refused when the request repeats,
// a passed one is served from the cache, and a noted one is served with its
// note. Each protocol, streamed or not, through both gateway modes.
func TestTheResponseCacheKeepsOnlyWhatTheGuardDelivered(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte(guardCacheConfig))
	if err != nil {
		t.Fatalf("parse config: %v", err)
	}
	for mode, engineFor := range looperAdapters {
		engine := engineFor(newParityRouter(t, cfg))
		for _, format := range guardProtocols {
			for _, stream := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/stream=%t", mode, format, stream), func(t *testing.T) {
					question := fmt.Sprintf("how fast is light (%s %s %t)", mode, format, stream)
					refused := guardCacheRequest(format, "__guarded__ __refuse__ "+question, stream)
					for attempt := 1; attempt <= 2; attempt++ {
						out := answerGuardedRequest(t, engine, refused, stream, false)
						assertGuardRefused(t, format, stream, out)
					}

					passed := guardCacheRequest(format, "__guarded__ "+question, stream)
					assertGuardDelivered(t, answerGuardedRequest(t, engine, passed, stream, false), benignAnswer)
					assertGuardDelivered(t, answerGuardedRequest(t, engine, passed, stream, true), benignAnswer)

					noted := guardCacheRequest(format, "__guarded__ __note__ "+question, stream)
					assertGuardDelivered(t, answerGuardedRequest(t, engine, noted, stream, false), responseGuardNote, benignAnswer)
					assertGuardDelivered(t, answerGuardedRequest(t, engine, noted, stream, true), responseGuardNote, benignAnswer)
				})
			}
		}
	}
}

// answerGuardedRequest plans request and returns the client's response: the
// cache's answer when fromCache, else the guarded decision's model's answer.
func answerGuardedRequest(t *testing.T, engine routing.Engine, request func() *routing.Request, stream, fromCache bool) *routing.Response {
	t.Helper()
	plan, err := engine.Plan(t.Context(), request())
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	t.Cleanup(func() { plan.Finish(nil) })
	if fromCache {
		if plan.Immediate == nil || plan.Immediate.Header.Get(headers.VSRResponsePath) != headers.ResponsePathCache {
			t.Fatalf("the request was not answered from the cache: %+v", plan.Immediate)
		}
		return plan.Immediate
	}
	if plan.Immediate != nil {
		t.Fatalf("the Router answered the request itself (%s %d %s), want the model's answer",
			plan.Immediate.Header.Get(headers.VSRResponsePath), plan.Immediate.Status, plan.Immediate.Body)
	}
	out, err := engine.Respond(t.Context(), plan, chatUpstreamAnswer(benignAnswer, stream))
	if err != nil {
		t.Fatalf("respond: %v", err)
	}
	return out
}

var guardCacheRequestIDs atomic.Int64

// guardCacheRequest builds one request of format for prompt, streamed or
// not; every request it builds has its own request ID.
func guardCacheRequest(format llmprotocol.WireFormat, prompt string, stream bool) func() *routing.Request {
	path, body := "/v1/chat/completions", fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"messages":[{"role":"user","content":%q}]}`, stream, prompt)
	var extra routing.Header
	switch format {
	case llmprotocol.OpenAIResponsesV1:
		path, body = "/v1/responses", fmt.Sprintf(`{"model":"vllm-sr/auto","stream":%t,"store":false,"input":%q}`, stream, prompt)
	case llmprotocol.AnthropicMessagesV1:
		path = "/v1/messages"
		body = fmt.Sprintf(`{"model":"vllm-sr/auto","max_tokens":64,"stream":%t,"messages":[{"role":"user","content":%q}]}`, stream, prompt)
		extra = routing.Header{{Name: "anthropic-version", Value: "2023-06-01"}}
	}
	return func() *routing.Request {
		header := append(routing.Header{
			{Name: ":method", Value: "POST"},
			{Name: ":path", Value: path},
			{Name: ":authority", Value: "router"},
			{Name: ":scheme", Value: "http"},
			{Name: "content-type", Value: "application/json"},
			{Name: "x-request-id", Value: fmt.Sprintf("guard-cache-%d", guardCacheRequestIDs.Add(1))},
		}, extra...)
		return &routing.Request{Header: header, Body: []byte(body)}
	}
}

// chatUpstreamAnswer is the Chat backend's answer, as one body or a stream.
func chatUpstreamAnswer(answer string, stream bool) *routing.UpstreamResponse {
	if stream {
		var body strings.Builder
		for _, part := range streamOf(llmprotocol.OpenAIChatV1, answer) {
			body.Write(part)
		}
		return &routing.UpstreamResponse{
			Status: 200, Header: routing.Header{{Name: "content-type", Value: "text/event-stream"}}, Body: strings.NewReader(body.String()),
		}
	}
	return &routing.UpstreamResponse{
		Status: 200, Header: routing.Header{{Name: "content-type", Value: "application/json"}},
		Body: strings.NewReader(strings.ReplaceAll(string(extProcResponseFixture(llmprotocol.OpenAIChatV1)), "hello", answer)),
	}
}

func assertGuardRefused(t *testing.T, format llmprotocol.WireFormat, stream bool, out *routing.Response) {
	t.Helper()
	body := drainResponse(t, out)
	if stream {
		if out.Status != 200 {
			t.Fatalf("status = %d, want the refusal as a 200 event stream", out.Status)
		}
		assertRefusedStream(t, format, body, "Response blocked by the response guard", benignAnswer)
		return
	}
	if out.Status != 403 || !strings.Contains(body, "Response blocked by the response guard") || strings.Contains(body, benignAnswer) {
		t.Fatalf("reply = %d %s, want the guard's 403 and none of the answer", out.Status, body)
	}
}

func assertGuardDelivered(t *testing.T, out *routing.Response, texts ...string) {
	t.Helper()
	body := drainResponse(t, out)
	if out.Status != 200 {
		t.Fatalf("reply = %d %s, want the answer", out.Status, body)
	}
	for _, text := range texts {
		if !strings.Contains(body, text) {
			t.Fatalf("reply %s does not carry %q", body, text)
		}
	}
}

// An entry is served only under the guard that passed it: the partition
// changes when the decision gains a guard or its guard changes, and stays
// the same for the same guard and for a decision with none.
func TestResponseCachePartitionFollowsTheDecisionsGuard(t *testing.T) {
	partition := func(plugins ...config.DecisionPlugin) string {
		decision := config.Decision{Name: "guarded", Plugins: plugins}
		ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIChatV1, VSRSelectedDecision: &decision}
		return responseCacheIdentity(ctx, "vllm-sr/auto").Partition.Key()
	}
	unguarded := partition()
	annotating := partition(guardPlugin(jailbreakGuard(config.ResponseGuardAnnotate, "")))
	blocking := partition(guardPlugin(jailbreakGuard(config.ResponseGuardBlock, "")))
	if unguarded == annotating || annotating == blocking || unguarded == blocking {
		t.Fatalf("partitions = %q, %q, %q; want one per guard", unguarded, annotating, blocking)
	}
	if again := partition(guardPlugin(jailbreakGuard(config.ResponseGuardBlock, ""))); again != blocking {
		t.Fatalf("the same guard gave partition %q, then %q", blocking, again)
	}
	if !strings.HasSuffix(unguarded, ":") {
		t.Fatalf("partition %q of a decision without a guard carries an epoch", unguarded)
	}
}

// The cache keeps an answer the guard passed or noted in full, and never one
// it blocked or flagged only in x-vsr-warnings, which a hit does not carry.
func TestResponseGuardCacheSkipReason(t *testing.T) {
	for _, tc := range []struct {
		name      string
		jailbreak float32
		action    string
		want      string
	}{
		{"passed", 0.01, config.ResponseGuardBlock, ""},
		{"blocked", 0.95, config.ResponseGuardBlock, "response_guard_blocked"},
		{"flagged in the headers", 0.95, config.ResponseGuardAnnotate, "response_guard_annotated"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router, ctx := newResponseStageRouter(t, newJailbreakDecider(tc.jailbreak), "", tc.action)
			guardAnswer(t, router, ctx, unsafeAnswer)
			if got := ctx.ResponseGuard.cacheSkipReason(); got != tc.want {
				t.Fatalf("cacheSkipReason() = %q, want %q", got, tc.want)
			}
		})
	}
	t.Run("noted", func(t *testing.T) {
		server, _ := newHallucinationDecider([]string{"450 meters"}, false)
		router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, true, ""))
		guardResponse(t, router, ctx, hallucinationAssistantResponse(hallucinationAnswer))
		if got := ctx.ResponseGuard.cacheSkipReason(); got != "" {
			t.Fatalf("cacheSkipReason() = %q for an answer whose warning is its note", got)
		}
	})
	if got := (*responseGuardRun)(nil).cacheSkipReason(); got != "" {
		t.Fatalf("cacheSkipReason() = %q without a guard", got)
	}
}
