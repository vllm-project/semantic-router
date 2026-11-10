package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
)

const benignAnswer = "The speed of light is about 299,792 km/s."

var guardProtocols = []llmprotocol.WireFormat{
	llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, llmprotocol.AnthropicMessagesV1,
}

// streamOf is format's upstream stream of answer, one SSE event per part.
func streamOf(format llmprotocol.WireFormat, answer string) [][]byte {
	stream := strings.ReplaceAll(string(extProcStreamFixture(format)), "hello", answer)
	var parts [][]byte
	for _, event := range strings.SplitAfter(stream, "\n\n") {
		if event != "" {
			parts = append(parts, []byte(event))
		}
	}
	return parts
}

// streamingClient puts ctx in the state of a streaming request of format that
// reached a backend of the same format; a Chat client asks for the usage chunk.
func streamingClient(ctx *RequestContext, format llmprotocol.WireFormat) {
	ctx.SourceFormat, ctx.TargetFormat = format, format
	ctx.RequestModel = "model-a"
	if ctx.SemanticRequest == nil {
		ctx.SemanticRequest = &llmprotocol.Request{Generation: 1, Model: "model-a"}
	}
	ctx.SemanticRequest.Stream = true
	ctx.ExpectStreamingResponse = true
	if format == llmprotocol.OpenAIChatV1 {
		includeUsage := true
		ctx.SemanticRequest.StreamOptions.IncludeUsage = &includeUsage
	}
}

// answerHeaders runs the response headers phase for a 200 of contentType.
func answerHeaders(t *testing.T, router *OpenAIRouter, ctx *RequestContext, contentType string) *ext_proc.ProcessingResponse {
	t.Helper()
	response, err := router.handleResponseHeaders(&ext_proc.ProcessingRequest_ResponseHeaders{ResponseHeaders: &ext_proc.HttpHeaders{
		Headers: &core.HeaderMap{Headers: []*core.HeaderValue{{Key: ":status", Value: "200"}, {Key: "content-type", Value: contentType}}},
	}}, ctx)
	if err != nil {
		t.Fatalf("response headers: %v", err)
	}
	return response
}

// answerBody delivers parts as the gateway's response body, the last one
// ending the stream, and returns what the client receives and every reply.
func answerBody(t *testing.T, router *OpenAIRouter, ctx *RequestContext, parts [][]byte) (string, []*ext_proc.ProcessingResponse) {
	t.Helper()
	var client strings.Builder
	var replies []*ext_proc.ProcessingResponse
	for i, part := range parts {
		reply, err := router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
			ResponseBody: &ext_proc.HttpBody{Body: part, EndOfStream: i == len(parts)-1},
		}, ctx)
		if err != nil {
			t.Fatalf("response body part %d: %v", i, err)
		}
		if immediate := reply.GetImmediateResponse(); immediate != nil {
			client.Write(immediate.GetBody())
		} else {
			client.Write(clientStreamBytes(reply, part))
		}
		replies = append(replies, reply)
	}
	return client.String(), replies
}

// whole joins parts into the one body a buffering gateway delivers.
func whole(parts [][]byte) [][]byte {
	var body []byte
	for _, part := range parts {
		body = append(body, part...)
	}
	return [][]byte{body}
}

// streamedThroughUnguarded is what the client of format receives for parts
// through a decision without a guard: the stream as the Router streams it.
func streamedThroughUnguarded(t *testing.T, format llmprotocol.WireFormat, parts [][]byte) string {
	t.Helper()
	router := &OpenAIRouter{}
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: t.Context()}
	streamingClient(ctx, format)
	answerHeaders(t, router, ctx, "text/event-stream")
	client, _ := answerBody(t, router, ctx, parts)
	return client
}

// guardedStreamingRouter is the response-stage router whose decision's guard
// takes action on a jailbreak in the answer, for a streaming client of format.
func guardedStreamingRouter(t *testing.T, jailbreak float32, action string, format llmprotocol.WireFormat) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	router, ctx := newResponseStageRouter(t, newJailbreakDecider(jailbreak), "", action)
	streamingClient(ctx, format)
	return router, ctx
}

func bufferedCount(ctx *RequestContext, outcome string) float64 {
	return testutil.ToFloat64(metrics.ResponseGuardBufferedTotal.WithLabelValues(requestDecisionStateKey(ctx), outcome))
}

// assertRefusedStream checks that client is format's stream failing with
// message, carrying nothing of the answer.
func assertRefusedStream(t *testing.T, format llmprotocol.WireFormat, client, message, answer string) {
	t.Helper()
	opening := map[llmprotocol.WireFormat]string{
		llmprotocol.OpenAIChatV1:        `data: {"error":`,
		llmprotocol.OpenAIResponsesV1:   "event: response.failed",
		llmprotocol.AnthropicMessagesV1: "event: error",
	}[format]
	if !strings.Contains(client, opening) || !strings.Contains(client, message) {
		t.Fatalf("client stream = %s, want %q carrying %q", client, opening, message)
	}
	if strings.Contains(client, answer) {
		t.Fatalf("client stream = %s, yet the refused answer must not reach the client", client)
	}
}

// guardOutcomes are the response_guard rule outcomes of the request's record.
func guardOutcomes(t *testing.T, recorder *routerreplay.Recorder, ctx *RequestContext) []routerreplay.Outcome {
	t.Helper()
	var found []routerreplay.Outcome
	for _, outcome := range replayOutcomes(t, recorder, ctx.RouterReplayID) {
		if outcome.Target == config.DecisionPluginResponseGuard {
			found = append(found, outcome)
		}
	}
	if len(found) == 0 {
		t.Fatal("the record has no response_guard outcome")
	}
	return found
}

func TestOnlyAnEnforcingGuardBuffersAStreamedAnswer(t *testing.T) {
	jailbreak := config.RuleNode{Type: config.SignalTypeJailbreak, Name: responseStageRuleName}
	for _, tc := range []struct {
		name   string
		guard  *config.ResponseGuardPluginConfig
		buffer bool
	}{
		{"block", &config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: jailbreak, Action: config.ResponseGuardBlock}}}, true},
		{"annotate with details", &config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: jailbreak, Action: config.ResponseGuardAnnotate, Details: true}}}, true},
		{"on_unknown block", &config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: jailbreak, Action: config.ResponseGuardNone}}, OnUnknown: config.ResponseGuardBlock}, true},
		{"annotate", &config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: jailbreak, Action: config.ResponseGuardAnnotate}}}, false},
		{"none", &config.ResponseGuardPluginConfig{Rules: []config.ResponseGuardRule{{When: jailbreak, Action: config.ResponseGuardNone}}, OnUnknown: config.ResponseGuardNone}, false},
		{"no guard", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &RequestContext{VSRSelectedDecision: &config.Decision{Name: "test_decision"}}
			if tc.guard != nil {
				ctx.VSRSelectedDecision = responseGuardDecision(*tc.guard)
			}
			response := answerHeaders(t, &OpenAIRouter{}, ctx, "text/event-stream")
			want := http_ext.ProcessingMode_STREAMED
			if tc.buffer {
				want = http_ext.ProcessingMode_BUFFERED
			}
			if got := response.GetModeOverride().GetResponseBodyMode(); got != want || (ctx.ResponseGuardHold != nil) != tc.buffer {
				t.Fatalf("response body mode = %v, held = %v; want %v, held = %v", got, ctx.ResponseGuardHold != nil, want, tc.buffer)
			}
		})
	}
}

// A buffered answer needs no hold: the gateway holds every body that is not
// streamed.
func TestAnEnforcingGuardLeavesABufferedAnswerAlone(t *testing.T) {
	_, ctx := guardedStreamingRouter(t, 0.95, config.ResponseGuardBlock, llmprotocol.OpenAIChatV1)
	ctx.SemanticRequest.Stream, ctx.ExpectStreamingResponse = false, false
	response := answerHeaders(t, &OpenAIRouter{}, ctx, "application/json")
	if response.GetModeOverride() != nil || ctx.ResponseGuardHold != nil {
		t.Fatalf("mode override = %v, held = %v for a buffered answer", response.GetModeOverride(), ctx.ResponseGuardHold != nil)
	}
}

// The guard of a held stream reads the whole answer before the client receives
// any of it. An answer it passes reaches the client as the stream the Router
// streams without a guard, usage and terminal event included.
func TestHeldStreamReleasesAPassedAnswerAsTheStream(t *testing.T) {
	for _, format := range guardProtocols {
		t.Run(string(format), func(t *testing.T) {
			router, ctx := guardedStreamingRouter(t, 0.01, config.ResponseGuardBlock, format)
			recorder := startResponseStageReplay(t, router, ctx)
			streamingClient(ctx, format)
			before := bufferedCount(ctx, "passed")
			parts := streamOf(format, benignAnswer)

			if mode := answerHeaders(t, router, ctx, "text/event-stream").GetModeOverride().GetResponseBodyMode(); mode != http_ext.ProcessingMode_BUFFERED {
				t.Fatalf("response body mode = %v, want BUFFERED", mode)
			}
			client, _ := answerBody(t, router, ctx, whole(parts))

			if want := streamedThroughUnguarded(t, format, parts); client != want {
				t.Fatalf("client stream =\n%s\nwant the stream as it streams unguarded:\n%s", client, want)
			}
			if got := bufferedCount(ctx, "passed"); got != before+1 {
				t.Fatalf("%s{outcome=passed} = %v, want %v", "llm_response_guard_buffered_total", got, before+1)
			}
			for _, outcome := range guardOutcomes(t, recorder, ctx) {
				if outcome.Metadata["stream"] != responseGuardStreamBuffered || outcome.Verdict != "not_matched" {
					t.Fatalf("guard outcome = %+v, want a rule not matched on a buffered stream", outcome)
				}
			}
		})
	}
}

// An answer the guard blocks never reaches the client: it receives the
// refusal a buffered answer's 403 carries, in its protocol's streaming form.
func TestHeldStreamRefusesABlockedAnswerInStreamForm(t *testing.T) {
	for _, format := range guardProtocols {
		t.Run(string(format), func(t *testing.T) {
			router, ctx := guardedStreamingRouter(t, 0.95, config.ResponseGuardBlock, format)
			recorder := startResponseStageReplay(t, router, ctx)
			streamingClient(ctx, format)
			ctx.Headers[headers.VSRDebug] = "true"
			before := bufferedCount(ctx, "blocked")

			answerHeaders(t, router, ctx, "text/event-stream")
			client, replies := answerBody(t, router, ctx, whole(streamOf(format, unsafeAnswer)))

			assertRefusedStream(t, format, client, "Response blocked: jailbreak content detected in LLM output", unsafeAnswer)
			if trace := bodyPhaseHeader(replies[0], headers.VSRTrace); !strings.Contains(trace, "response_guard=blocked;buffered") {
				t.Fatalf("%s = %q, want the blocked outcome of a buffered stream", headers.VSRTrace, trace)
			}
			if got := bufferedCount(ctx, "blocked"); got != before+1 {
				t.Fatalf("llm_response_guard_buffered_total{outcome=blocked} = %v, want %v", got, before+1)
			}
			rule := outcomeFor(t, guardOutcomes(t, recorder, ctx), config.DecisionPluginResponseGuard, "rules[0]")
			if rule.Verdict != "matched" || rule.Metadata["action"] != config.ResponseGuardBlock || rule.Metadata["stream"] != responseGuardStreamBuffered {
				t.Fatalf("rule outcome = %+v, want matched and blocked on a buffered stream", rule)
			}
		})
	}
}

// The same guard on the same answers, not streamed: 403 for the blocked one,
// the answer for the passed one.
func TestBufferedAnswerIsBlockedOrPassedInEveryProtocol(t *testing.T) {
	for _, format := range guardProtocols {
		for _, tc := range []struct {
			name      string
			jailbreak float32
			answer    string
			blocked   bool
		}{{"blocked", 0.95, unsafeAnswer, true}, {"passed", 0.01, benignAnswer, false}} {
			t.Run(string(format)+"/"+tc.name, func(t *testing.T) {
				router, ctx := guardedStreamingRouter(t, tc.jailbreak, config.ResponseGuardBlock, format)
				ctx.SemanticRequest.Stream, ctx.ExpectStreamingResponse = false, false
				body := []byte(strings.ReplaceAll(string(extProcResponseFixture(format)), "hello", tc.answer))

				answerHeaders(t, router, ctx, "application/json")
				client, replies := answerBody(t, router, ctx, [][]byte{body})

				immediate := replies[0].GetImmediateResponse()
				if tc.blocked != (immediate != nil && immediate.GetStatus().GetCode() == 403) {
					t.Fatalf("reply = %v, blocked = %v", replies[0], tc.blocked)
				}
				if tc.blocked && (!strings.Contains(client, "jailbreak content detected") || strings.Contains(client, tc.answer)) {
					t.Fatalf("403 body = %s, want the block message and none of the answer", client)
				}
				if !tc.blocked && !strings.Contains(client, tc.answer) {
					t.Fatalf("client body = %s, want the answer", client)
				}
			})
		}
	}
}

// A gateway that ignores the Router's BUFFERED mode streams the body anyway;
// the Router holds every part itself, so the client still receives nothing
// before the guard has read the whole answer.
func TestHeldStreamSendsNothingBeforeTheGuardRead(t *testing.T) {
	for _, tc := range []struct {
		name      string
		jailbreak float32
		answer    string
	}{{"blocked", 0.95, unsafeAnswer}, {"passed", 0.01, benignAnswer}} {
		t.Run(tc.name, func(t *testing.T) {
			router, ctx := guardedStreamingRouter(t, tc.jailbreak, config.ResponseGuardBlock, llmprotocol.OpenAIChatV1)
			parts := streamOf(llmprotocol.OpenAIChatV1, tc.answer)

			answerHeaders(t, router, ctx, "text/event-stream")
			client, replies := answerBody(t, router, ctx, parts)

			for i, reply := range replies[:len(replies)-1] {
				if sent := clientStreamBytes(reply, parts[i]); len(sent) != 0 {
					t.Fatalf("part %d sent %q to the client before the guard read the answer", i, sent)
				}
			}
			if tc.jailbreak > 0.5 {
				assertRefusedStream(t, llmprotocol.OpenAIChatV1, client, "jailbreak content detected", tc.answer)
			} else if want := streamedThroughUnguarded(t, llmprotocol.OpenAIChatV1, parts); client != want {
				t.Fatalf("client stream =\n%s\nwant\n%s", client, want)
			}
		})
	}
}

// A held stream larger than the hold's bound is refused unread: the guard
// cannot read what the Router does not hold, so nothing of it leaves.
func TestHeldStreamOverTheLimitIsRefusedUnread(t *testing.T) {
	limit := responseGuardHoldLimit
	t.Cleanup(func() { responseGuardHoldLimit = limit })
	for _, tc := range []struct {
		name  string
		parts func([][]byte) [][]byte
	}{
		{"one buffered body", whole},
		{"streamed parts", func(parts [][]byte) [][]byte { return parts }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			decider, calls := newCountingJailbreakDecider(0.01)
			router, ctx := newResponseStageRouter(t, decider, "", config.ResponseGuardBlock)
			recorder := startResponseStageReplay(t, router, ctx)
			streamingClient(ctx, llmprotocol.OpenAIChatV1)
			parts := streamOf(llmprotocol.OpenAIChatV1, benignAnswer)
			responseGuardHoldLimit = len(parts[0]) + 1
			before := bufferedCount(ctx, responseGuardOverLimit)

			answerHeaders(t, router, ctx, "text/event-stream")
			client, replies := answerBody(t, router, ctx, tc.parts(parts))

			assertRefusedStream(t, llmprotocol.OpenAIChatV1, client, "larger than the response guard can check", benignAnswer)
			blocked := false
			for _, reply := range replies {
				blocked = blocked || bodyPhaseHeader(reply, headers.VSRResponsePath) == headers.ResponsePathBlocked
			}
			if !blocked {
				t.Fatalf("no reply set %s to %q", headers.VSRResponsePath, headers.ResponsePathBlocked)
			}
			if calls.Load() != 0 {
				t.Fatalf("the guard asked the decision model %d time(s) about an answer it did not hold", calls.Load())
			}
			if got := bufferedCount(ctx, responseGuardOverLimit); got != before+1 {
				t.Fatalf("llm_response_guard_buffered_total{outcome=over_limit} = %v, want %v", got, before+1)
			}
			unread := outcomeFor(t, guardOutcomes(t, recorder, ctx), config.DecisionPluginResponseGuard, "")
			if unread.Verdict != "unread" || unread.Reason != responseGuardOverLimit || unread.Metadata["action"] != config.ResponseGuardBlock {
				t.Fatalf("guard outcome = %+v, want the answer refused unread over the limit", unread)
			}
		})
	}
}

// A held stream that ends without an answer has nothing for the guard to
// read: the client receives the stream's failure and none of what came first.
func TestIncompleteHeldStreamSendsOnlyItsFailure(t *testing.T) {
	decider, calls := newCountingJailbreakDecider(0.01)
	router, ctx := newResponseStageRouter(t, decider, "", config.ResponseGuardBlock)
	streamingClient(ctx, llmprotocol.OpenAIChatV1)
	before := bufferedCount(ctx, responseGuardIncomplete)
	parts := [][]byte{
		[]byte(`data: {"id":"response_1","model":"model-a","choices":[{"index":0,"delta":{"role":"assistant","content":"` + unsafeAnswer + `"},"finish_reason":null}]}` + "\n\n"),
		[]byte(`data: {"error":{"message":"upstream overloaded","type":"server_error","code":"overloaded"}}` + "\n\n"),
	}

	answerHeaders(t, router, ctx, "text/event-stream")
	client, _ := answerBody(t, router, ctx, whole(parts))

	assertRefusedStream(t, llmprotocol.OpenAIChatV1, client, "upstream overloaded", unsafeAnswer)
	if calls.Load() != 0 || ctx.ResponseGuard != nil {
		t.Fatalf("the guard read a stream that ended without an answer (%d decision model calls)", calls.Load())
	}
	if got := bufferedCount(ctx, responseGuardIncomplete); got != before+1 {
		t.Fatalf("llm_response_guard_buffered_total{outcome=incomplete} = %v, want %v", got, before+1)
	}
}

// A rule that writes a note is enforcing too: the held answer reaches the
// client with the note at its head, its warning in x-vsr-warnings.
func TestHeldStreamCarriesTheGuardsNote(t *testing.T) {
	server, _ := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, true, ""))
	streamingClient(ctx, llmprotocol.OpenAIChatV1)

	answerHeaders(t, router, ctx, "text/event-stream")
	client, replies := answerBody(t, router, ctx, whole(streamOf(llmprotocol.OpenAIChatV1, hallucinationAnswer)))

	content := chatStreamContent(client)
	if !strings.HasPrefix(content, "[Hallucination Warning]") || !strings.HasSuffix(content, hallucinationAnswer) {
		t.Fatalf("streamed answer = %q, want the note at the head of the answer", content)
	}
	if !strings.Contains(client, `"usage":`) || !strings.HasSuffix(client, "data: [DONE]\n\n") {
		t.Fatalf("client stream = %s, want the usage chunk and the terminal event", client)
	}
	if warnings := bodyPhaseHeader(replies[0], headers.VSRWarnings); !strings.Contains(warnings, headers.WarningHallucination) {
		t.Fatalf("%s = %q, want %q", headers.VSRWarnings, warnings, headers.WarningHallucination)
	}
}

// A guard that neither blocks nor writes notes keeps true streaming: every
// part reaches the client as it arrives, and the guard reads the answer once
// the stream ended and records what it found.
func TestNonEnforcingGuardKeepsTrueStreaming(t *testing.T) {
	router, ctx := guardedStreamingRouter(t, 0.95, config.ResponseGuardAnnotate, llmprotocol.OpenAIChatV1)
	recorder := startResponseStageReplay(t, router, ctx)
	streamingClient(ctx, llmprotocol.OpenAIChatV1)
	parts := streamOf(llmprotocol.OpenAIChatV1, unsafeAnswer)

	if mode := answerHeaders(t, router, ctx, "text/event-stream").GetModeOverride().GetResponseBodyMode(); mode != http_ext.ProcessingMode_STREAMED {
		t.Fatalf("response body mode = %v, want STREAMED", mode)
	}
	_, replies := answerBody(t, router, ctx, parts)

	if first := string(clientStreamBytes(replies[0], parts[0])); !strings.Contains(first, unsafeAnswer) {
		t.Fatalf("first part sent %q, want the answer as it arrived", first)
	}
	if matched, _ := ctx.ResponseGuard.jailbreakFinding(); !matched {
		t.Fatal("the guard did not read the streamed answer")
	}
	outcomes := replayOutcomes(t, recorder, ctx.RouterReplayID)
	rule := outcomeFor(t, outcomes, config.DecisionPluginResponseGuard, "rules[0]")
	if rule.Verdict != "matched" || rule.Metadata["action"] != config.ResponseGuardAnnotate || rule.Metadata["stream"] != responseGuardStreamUnbuffered {
		t.Fatalf("rule outcome = %+v, want matched and annotated on an unbuffered stream", rule)
	}
	sum := sha256.Sum256([]byte(unsafeAnswer))
	for _, outcome := range outcomes {
		if outcome.Source == primaryResponseOutcomeSource && outcome.Metadata["response_sha256"] == hex.EncodeToString(sum[:]) {
			return
		}
	}
	t.Fatalf("outcomes = %+v, want the primary digest of the streamed answer", outcomes)
}

// The gateway sends a buffered body as one ExtProc message; one larger than
// the server admits ends the stream before the Router reads it, and the hold
// counts it over the limit.
func TestHeldStreamOverTheMessageLimitIsCounted(t *testing.T) {
	router, ctx := guardedStreamingRouter(t, 0.01, config.ResponseGuardBlock, llmprotocol.OpenAIChatV1)
	before := bufferedCount(ctx, responseGuardOverLimit)
	answerHeaders(t, router, ctx, "text/event-stream")

	err := router.handleProcessReceiveError(ctx, status.Error(codes.ResourceExhausted, "grpc: received message larger than max"))

	if err == nil || ctx.ResponseGuardHold.outcome != responseGuardOverLimit {
		t.Fatalf("receive error = %v, hold outcome = %q, want the stream failed over the limit", err, ctx.ResponseGuardHold.outcome)
	}
	if got := bufferedCount(ctx, responseGuardOverLimit); got != before+1 {
		t.Fatalf("llm_response_guard_buffered_total{outcome=over_limit} = %v, want %v", got, before+1)
	}
}

// A stream that never reaches a terminal answer has nothing to check. The
// detector must not be asked for a partial one.
func TestAbortedStreamIsNotReadByTheGuard(t *testing.T) {
	server, calls := newHallucinationDecider([]string{"450 meters"}, false)
	router, ctx := newHallucinationSignalRouter(t, server, groundedGuard(config.ResponseGuardAnnotate, false, ""))
	startResponseStageReplay(t, router, ctx)
	streamingClient(ctx, llmprotocol.OpenAIChatV1)
	ctx.IsStreamingResponse = true

	router.handleSemanticStreamingResponseBody([]byte("data: {\"id\":\"response_1\"\n\n"), true, ctx)

	if !ctx.StreamingAborted {
		t.Fatal("the malformed stream was not treated as aborted")
	}
	if calls.Load() != 0 || ctx.ResponseGuard != nil {
		t.Fatalf("an aborted stream has no terminal answer, yet the guard read it (%d detector calls)", calls.Load())
	}
}

// chatStreamContent joins the content deltas of a Chat stream.
func chatStreamContent(stream string) string {
	var content strings.Builder
	for _, event := range strings.Split(stream, "\n\n") {
		data, ok := strings.CutPrefix(event, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(data), &chunk) != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			content.WriteString(choice.Delta.Content)
		}
	}
	return content.String()
}
