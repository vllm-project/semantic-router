package protocolcodec

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// conformancePolicy is the test-only contract that rejects every upstream
// member the wire structs do not declare, so fixtures cannot drift from them.
func conformancePolicy() llmprotocol.Policy {
	policy := llmprotocol.DefaultPolicy()
	policy.RejectUnknownUpstreamFields = true
	return policy
}

// conformanceEngine decodes the golden fixtures, so a fixture that carries a
// member the wire structs do not declare fails instead of being dropped.
func conformanceEngine() *Engine {
	engine, err := NewEngine(NewBuiltinRegistry(), conformancePolicy())
	if err != nil {
		panic(err)
	}
	return engine
}

// decoratedChatCompletion carries the decorations from issue #3496: fields the
// Chat contract does not declare at the top level, in a choice and in usage.
const decoratedChatCompletion = `{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "created": 1,
  "model": "gpt-4.1-mini",
  "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
  "routing": {"region": "eastus"},
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": "hi"},
    "content_filter_results": {"hate": {"filtered": false, "severity": "safe"}}
  }],
  "usage": {
    "prompt_tokens": 1,
    "completion_tokens": 1,
    "total_tokens": 2,
    "latency_checkpoint": {"time_to_first_token_ms": 12}
  }
}`

// chatTarget is the canonical subset used by the stripping tests.
type chatTarget struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		FinishReason string `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

func TestStripUnknownUpstreamFieldsRemovesUndeclaredMembers(t *testing.T) {
	stripped, dropped := stripUnknownUpstreamFields([]byte(decoratedChatCompletion), reflect.TypeOf(chatTarget{}))

	want := []string{
		"choices[].content_filter_results",
		"prompt_filter_results",
		"routing",
		"usage.latency_checkpoint",
	}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}

	for _, field := range []string{"prompt_filter_results", "\"routing\"", "content_filter_results", "latency_checkpoint"} {
		if strings.Contains(string(stripped), field) {
			t.Errorf("stripped body still contains %s", field)
		}
	}
	for _, kept := range []string{`"id"`, `"model"`, `"finish_reason"`, `"total_tokens"`} {
		if !strings.Contains(string(stripped), kept) {
			t.Errorf("stripped body dropped canonical field %s", kept)
		}
	}
}

func TestStripUnknownUpstreamFieldsLeavesCanonicalBodyUntouched(t *testing.T) {
	canonical := `{"id":"chatcmpl-1","model":"gpt-4.1-mini","choices":[{"index":0}]}`

	stripped, dropped := stripUnknownUpstreamFields([]byte(canonical), reflect.TypeOf(chatTarget{}))

	if dropped != nil {
		t.Errorf("dropped = %v, want nil", dropped)
	}
	if string(stripped) != canonical {
		t.Errorf("stripped = %s, want the original body unchanged", stripped)
	}
}

func TestStripUnknownUpstreamFieldsDropsNestedMembers(t *testing.T) {
	body := `{"id":"a","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi","gateway_note":"x"}}]}`

	stripped, dropped := stripUnknownUpstreamFields([]byte(body), reflect.TypeOf(chatTarget{}))

	want := []string{"choices[].message.gateway_note"}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}
	if strings.Contains(string(stripped), "gateway_note") {
		t.Error("stripped body still contains the nested member")
	}
}

// A path is built from upstream member names, so it is bounded before it
// reaches the warning header and the debug log.
func TestUnknownFieldPathIsBounded(t *testing.T) {
	path := unknownFieldPath("usage", strings.Repeat("x", 4*maxUnknownFieldPathBytes))
	if len(path) > maxUnknownFieldPathBytes+len("...") || !strings.HasPrefix(path, "usage.x") {
		t.Fatalf("path = %d bytes, want at most %d", len(path), maxUnknownFieldPathBytes+len("..."))
	}
}

// Tolerance is the contract of every backend, whatever its failure-envelope
// vendor; only the conformance policy rejects unknown members.
func TestUnknownUpstreamFieldsDropForEveryBackend(t *testing.T) {
	for _, vendor := range []llmprotocol.ResponseVendor{"", llmprotocol.ResponseVendorCloudflare, llmprotocol.ResponseVendorSnowflake} {
		policy := llmprotocol.DefaultPolicy()
		policy.ResponseVendor = vendor
		if !dropUnknownUpstreamFields(policy) {
			t.Errorf("dropUnknownUpstreamFields(vendor=%q) = false, want true", vendor)
		}
	}
	if dropUnknownUpstreamFields(conformancePolicy()) {
		t.Error("dropUnknownUpstreamFields(conformance) = true, want false")
	}
}

func TestDecodeProviderWireAcceptsUnanticipatedFields(t *testing.T) {
	var target chatTarget
	body := `{"id":"a","object":"chat.completion","created":1,"model":"m",` +
		`"a_field_a_gateway_ships_next_year":{"nested":true},` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}]}`

	_, dropped, err := decodeProviderWireTolerant([]byte(body), &target, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decodeProviderWireTolerant() error = %v, want nil", err)
	}
	if !reflect.DeepEqual(dropped, []string{"a_field_a_gateway_ships_next_year"}) {
		t.Errorf("dropped = %v, want the unanticipated field reported", dropped)
	}
	if target.Model != "m" {
		t.Errorf("Model = %q, want m", target.Model)
	}
}

// Known members keep their types: tolerance covers names, not values.
func TestDecodeProviderWireStillRejectsAKnownFieldOfTheWrongType(t *testing.T) {
	var target chatTarget
	err := decodeProviderWire([]byte(`{"id":"a","model":"m","created":"yesterday","choices":[]}`), &target, llmprotocol.DefaultPolicy())
	if err == nil || !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Fatalf("decodeProviderWire() error = %v, want invalid_upstream_json", err)
	}
}

func TestConformanceDecodeRejectsUnknownFields(t *testing.T) {
	var target struct {
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(`{"model":"m","surprise_field":1}`), &target, conformancePolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want a rejection")
	}
	if !strings.Contains(err.Error(), "invalid_upstream_json") {
		t.Errorf("error = %v, want invalid_upstream_json", err)
	}
}

// Field details belong in the operator-visible cause, not the client message.
func TestConformanceDecodeKeepsTheOffendingFieldOutOfTheClientMessage(t *testing.T) {
	var target struct {
		Model string `json:"model"`
	}

	err := decodeProviderWire([]byte(`{"model":"m","surprise_field":1}`), &target, conformancePolicy())
	if err == nil {
		t.Fatal("decodeProviderWire() error = nil, want a rejection")
	}
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) {
		t.Fatalf("error = %v, want a ProtocolError", err)
	}
	if strings.Contains(protocolError.Message, "surprise_field") {
		t.Errorf("Message = %q, must not expose the upstream field name to the caller", protocolError.Message)
	}
	cause := errors.Unwrap(protocolError)
	if cause == nil || !strings.Contains(cause.Error(), "surprise_field") {
		t.Errorf("cause = %v, want it to name surprise_field for the operator log", cause)
	}
}

// Every dropped member must produce a dropped-field diagnostic.
func TestDecodeResponseReportsUnknownFieldsAsDiagnostics(t *testing.T) {
	_, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse([]byte(decoratedChatCompletion), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want nil", err)
	}

	dropped := map[string]bool{}
	for _, diagnostic := range diagnostics {
		if diagnostic.Action == llmprotocol.DiagnosticDropped && diagnostic.Reason == unknownUpstreamFieldReason {
			dropped[diagnostic.Field] = true
		}
	}
	for _, field := range []string{
		"prompt_filter_results",
		"routing",
		"choices[].content_filter_results",
		"usage.latency_checkpoint",
	} {
		if !dropped[field] {
			t.Errorf("no dropped diagnostic for %q; diagnostics = %+v", field, diagnostics)
		}
	}
}

// Error envelopes follow the same contract as successful responses.
func TestDecodeTransportErrorAcceptsDecoratedErrorEnvelope(t *testing.T) {
	body := `{"error":{"type":"invalid_request_error","code":"content_filter",` +
		`"message":"blocked","param":null,"innererror":{"content_filter_result":{"hate":{"filtered":true}}}}}`

	engine := NewBuiltinEngine()
	for _, format := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		t.Run(string(format), func(t *testing.T) {
			transportError, diagnostics, err := engine.DecodeTransportError(format, []byte(body))
			if err != nil {
				t.Fatalf("DecodeTransportError() error = %v, want nil", err)
			}
			if transportError.Error == nil || transportError.Error.Code != "content_filter" {
				t.Errorf("Error = %+v, want the upstream content_filter code preserved", transportError.Error)
			}
			if len(diagnostics) != 1 || diagnostics[0].Field != "error.innererror" ||
				diagnostics[0].Action != llmprotocol.DiagnosticDropped || diagnostics[0].Source != format {
				t.Errorf("diagnostics = %+v, want one dropped error.innererror field from %s", diagnostics, format)
			}
		})
	}

	_, _, conformanceErr := OpenAIChatCodec{}.DecodeTransportError([]byte(body), conformancePolicy())
	if conformanceErr == nil {
		t.Error("conformance DecodeTransportError() error = nil, want rejection")
	}
}

// An error envelope keeps its message and code when the upstream adds its own
// metadata beside them, as OpenRouter does (issue #5584).
func TestDecodeTransportErrorKeepsTheMessageBesideUpstreamMetadata(t *testing.T) {
	body := `{"error":{"type":"rate_limit_error","message":"Provider returned error","code":429,` +
		`"metadata":{"raw":"{}","provider_name":"Provider"}},"user_id":"user_1"}`

	transportError, diagnostics, err := NewBuiltinEngine().DecodeTransportError(llmprotocol.OpenAIChatV1, []byte(body))
	if err != nil {
		t.Fatalf("DecodeTransportError() error = %v, want nil", err)
	}
	if transportError.Error == nil || transportError.Error.Message != "Provider returned error" ||
		transportError.Error.Category != llmprotocol.ErrorRateLimited {
		t.Fatalf("Error = %+v, want the upstream message and the rate-limit category", transportError.Error)
	}
	dropped := map[string]bool{}
	for _, diagnostic := range diagnostics {
		dropped[diagnostic.Field] = diagnostic.Reason == unknownUpstreamFieldReason
	}
	if !dropped["error.metadata"] || !dropped["user_id"] {
		t.Fatalf("diagnostics = %+v, want error.metadata and user_id reported", diagnostics)
	}
}

func TestChatStreamErrorReportsUnknownFields(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	frame := []byte("data: {\"error\":{\"type\":\"server_error\",\"code\":\"blocked\",\"message\":\"blocked\"},\"gateway_trace\":{}}\n\n")
	_, events, diagnostics, err := stream.Push(frame)
	if err != nil {
		t.Fatal(err)
	}
	if len(events) != 1 || events[0].Error == nil || events[0].Error.Code != "blocked" {
		t.Fatalf("events = %+v, want the provider error", events)
	}
	if len(diagnostics) != 1 || diagnostics[0].Field != "gateway_trace" ||
		diagnostics[0].Action != llmprotocol.DiagnosticDropped {
		t.Errorf("diagnostics = %+v, want one dropped gateway_trace field", diagnostics)
	}
}

// A field that rides every frame is reported once per stream, so it cannot
// crowd later losses out of the bounded diagnostics list.
func TestChatStreamReportsAnUnknownFieldOncePerStream(t *testing.T) {
	stream, err := NewBuiltinEngine().NewStream(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIChatV1, llmprotocol.StreamContext{
		Context: context.Background(), PublicModel: "public-model",
	})
	if err != nil {
		t.Fatal(err)
	}
	body := []byte(
		`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","relay_trace":"t1","choices":[{"index":0,"delta":{"role":"assistant","content":"he"},"finish_reason":null}]}` + "\n\n" +
			`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","relay_trace":"t1","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}` + "\n\n" +
			`data: {"id":"c1","object":"chat.completion.chunk","created":1,"model":"m","relay_trace":"t1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}` + "\n\n" +
			"data: [DONE]\n\n")
	_, _, diagnostics, err := stream.Push(body)
	if err != nil {
		t.Fatalf("Push() error = %v", err)
	}
	if _, _, _, err := stream.Finalize(nil); err != nil {
		t.Fatalf("Finalize() error = %v", err)
	}
	count := 0
	for _, diagnostic := range diagnostics {
		if diagnostic.Field == "relay_trace" && diagnostic.Reason == unknownUpstreamFieldReason {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("relay_trace reported %d times, want once: %+v", count, diagnostics)
	}
}

// A caller that persists a response needs to know the bytes it holds are not
// canonical. This predicate is that signal.
func TestDiagnosticsDroppedUnknownFields(t *testing.T) {
	_, _, decorated, err := OpenAIChatCodec{}.DecodeResponse([]byte(decoratedChatCompletion), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want nil", err)
	}
	if !DiagnosticsDroppedUnknownFields(decorated) {
		t.Errorf("DiagnosticsDroppedUnknownFields() = false for %+v, want true", decorated)
	}

	if DiagnosticsDroppedUnknownFields(nil) {
		t.Error("DiagnosticsDroppedUnknownFields(nil) = true, want false")
	}
	unrelated := llmprotocol.Diagnostics{{
		Action: llmprotocol.DiagnosticDropped,
		Field:  "metadata",
		Reason: "response request metadata is not model output",
	}}
	if DiagnosticsDroppedUnknownFields(unrelated) {
		t.Error("DiagnosticsDroppedUnknownFields() = true for an unrelated drop, want false")
	}
}

// Issue #5046: a relay fronting Anthropic channels reports Chat usage with
// Anthropic-style counter names. The Chat codec maps them onto the Chat names
// before the decode, so the totals check sees a consistent usage object and
// the aliases surface as dropped members.
const relayChatCompletion = `{
  "id": "chatcmpl-gw-1",
  "object": "chat.completion",
  "created": 1,
  "model": "claude-channel-x",
  "choices": [{
    "index": 0,
    "finish_reason": "stop",
    "message": {"role": "assistant", "content": "hi"}
  }],
  "usage": {
    "input_tokens": 12,
    "output_tokens": 178,
    "total_tokens": 190
  }
}`

func TestDecodeChatProviderWireMapsAnthropicStyleUsageCounters(t *testing.T) {
	var target chatTarget
	canonical, dropped, err := decodeChatProviderWire([]byte(relayChatCompletion), &target, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decodeChatProviderWire() error = %v, want nil", err)
	}
	if target.Usage.PromptTokens != 12 || target.Usage.CompletionTokens != 178 || target.Usage.TotalTokens != 190 {
		t.Errorf("usage = %d/%d/%d, want 12/178/190",
			target.Usage.PromptTokens, target.Usage.CompletionTokens, target.Usage.TotalTokens)
	}
	want := []string{"usage.input_tokens", "usage.output_tokens"}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want %v", dropped, want)
	}
	if strings.Contains(string(canonical), "input_tokens") ||
		strings.Contains(string(canonical), "output_tokens") {
		t.Error("canonical body still carries the counter aliases")
	}
}

func TestDecodeChatProviderWireKeepsChatCounterPrecedence(t *testing.T) {
	var target chatTarget
	body := `{"id":"chatcmpl-gw-2","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":5,"input_tokens":12,"completion_tokens":6,` +
		`"output_tokens":178,"total_tokens":11}}`

	_, dropped, err := decodeChatProviderWire([]byte(body), &target, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decodeChatProviderWire() error = %v, want nil", err)
	}
	if target.Usage.PromptTokens != 5 || target.Usage.CompletionTokens != 6 || target.Usage.TotalTokens != 11 {
		t.Errorf("usage = %d/%d/%d, want the Chat fields to win (5/6/11)",
			target.Usage.PromptTokens, target.Usage.CompletionTokens, target.Usage.TotalTokens)
	}
	want := []string{"usage.input_tokens", "usage.output_tokens"}
	if !reflect.DeepEqual(dropped, want) {
		t.Errorf("dropped = %v, want the aliases reported %v", dropped, want)
	}
}

// The rewrite runs after the duplicate-key check on the bytes as received, so
// it cannot merge a repeated member away.
func TestDecodeChatProviderWireRejectsDuplicateKeysBeforeTheAlias(t *testing.T) {
	var target chatTarget
	body := `{"id":"a","model":"m","choices":[],"usage":{"input_tokens":1,"input_tokens":2,"total_tokens":1}}`
	_, _, err := decodeChatProviderWire([]byte(body), &target, llmprotocol.DefaultPolicy())
	if err == nil || !strings.Contains(err.Error(), "upstream_duplicate_json_field") {
		t.Fatalf("decodeChatProviderWire() error = %v, want upstream_duplicate_json_field", err)
	}
}

// Without the alias this payload dropped input_tokens/output_tokens, leaving
// total_tokens=190 against zero input/output, and the usage totals check
// rejected the response with usage_total_mismatch.
func TestChatDecodeResponseKeepsRelayUsageTotals(t *testing.T) {
	response, _, _, err := OpenAIChatCodec{}.DecodeResponse([]byte(relayChatCompletion), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want nil", err)
	}
	if response.Usage.InputTotal.Value == nil || *response.Usage.InputTotal.Value != 12 ||
		response.Usage.OutputTotal.Value == nil || *response.Usage.OutputTotal.Value != 178 {
		t.Fatalf("usage = %+v, want 12/178", response.Usage)
	}
}

// A same-format client receives the upstream bytes unchanged, unknown members
// included; the canonical encode that a persisted copy uses drops them and
// satisfies the conformance contract.
func TestSameFormatReplayKeepsUnknownFieldsAndCanonicalEncodeDropsThem(t *testing.T) {
	policy := llmprotocol.DefaultPolicy()
	response, envelope, _, err := OpenAIChatCodec{}.DecodeResponse([]byte(decoratedChatCompletion), policy)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v, want nil", err)
	}

	replayed, _, err := OpenAIChatCodec{}.EncodeResponse(response, envelope, policy)
	if err != nil {
		t.Fatalf("EncodeResponse(replay) error = %v, want nil", err)
	}
	if string(replayed) != decoratedChatCompletion {
		t.Fatalf("same-format replay changed the upstream bytes: %s", replayed)
	}

	envelope.ResponseReencodeRequired = true
	canonical, _, err := OpenAIChatCodec{}.EncodeResponse(response, envelope, policy)
	if err != nil {
		t.Fatalf("EncodeResponse(canonical) error = %v, want nil", err)
	}
	if strings.Contains(string(canonical), "routing") || strings.Contains(string(canonical), "latency_checkpoint") {
		t.Fatalf("canonical encode kept unknown members: %s", canonical)
	}
	if _, _, _, conformanceErr := (OpenAIChatCodec{}).DecodeResponse(canonical, conformancePolicy()); conformanceErr != nil {
		t.Errorf("conformance DecodeResponse(canonical) error = %v, want nil", conformanceErr)
	}
	if _, _, _, originalErr := (OpenAIChatCodec{}).DecodeResponse([]byte(decoratedChatCompletion), conformancePolicy()); originalErr == nil {
		t.Error("conformance DecodeResponse(original) error = nil, want rejection")
	}
}

// encoding/json matches member names case-insensitively, so a case-folded
// spelling is dropped before the decode instead of setting the field, and a
// second spelling beside the real one still fails as a duplicate.
func TestCaseFoldedUpstreamNameIsDroppedNotDecoded(t *testing.T) {
	var target struct {
		ID    string `json:"id"`
		Model string `json:"model"`
	}
	_, dropped, err := decodeProviderWireTolerant([]byte(`{"id":"a","Model":"spoofed"}`), &target, llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("decodeProviderWireTolerant() error = %v", err)
	}
	if target.Model != "" || !reflect.DeepEqual(dropped, []string{"Model"}) {
		t.Fatalf("model = %q, dropped = %v; want the case-folded spelling dropped", target.Model, dropped)
	}
	err = decodeProviderWire([]byte(`{"id":"a","model":"m","Model":"spoofed"}`), &target, llmprotocol.DefaultPolicy())
	if err == nil || !strings.Contains(err.Error(), "upstream_duplicate_json_field") {
		t.Fatalf("decodeProviderWire() error = %v, want upstream_duplicate_json_field", err)
	}
}

// An undeclared usage member is dropped before accounting: the decoded usage
// is the same with or without it.
func TestUnknownUsageMembersNeverReachAccounting(t *testing.T) {
	plain := `{"id":"c","object":"chat.completion","created":1,"model":"m",` +
		`"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi"}}],` +
		`"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}`
	decorated := strings.Replace(plain, `"prompt_tokens_details":{"cached_tokens":4}`,
		`"prompt_tokens_details":{"cached_tokens":4,"gateway_cached_tokens":9},"total_cost":0.25,"credits_used":7`, 1)
	want, _, _, err := OpenAIChatCodec{}.DecodeResponse([]byte(plain), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatal(err)
	}
	got, _, diagnostics, err := OpenAIChatCodec{}.DecodeResponse([]byte(decorated), llmprotocol.DefaultPolicy())
	if err != nil {
		t.Fatalf("DecodeResponse(decorated) error = %v", err)
	}
	if !reflect.DeepEqual(got.Usage, want.Usage) {
		t.Fatalf("usage = %+v, want %+v", got.Usage, want.Usage)
	}
	for _, field := range []string{"usage.credits_used", "usage.prompt_tokens_details.gateway_cached_tokens", "usage.total_cost"} {
		if !hasDiagnostic(diagnostics, field, llmprotocol.DiagnosticDropped) {
			t.Errorf("no dropped diagnostic for %s: %+v", field, diagnostics)
		}
	}
}
