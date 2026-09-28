package protocolcodec

import (
	"encoding/json"
	"errors"
	"reflect"
	"sort"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestOfficialAnthropicTerminalReasonInventoryIsClosed(t *testing.T) {
	engine := NewBuiltinEngine()
	tests := []anthropicTerminalReasonCase{
		{reason: "end_turn", neutral: llmprotocol.StopEndTurn},
		{reason: "max_tokens", neutral: llmprotocol.StopMaxTokens},
		{reason: "stop_sequence", neutral: llmprotocol.StopSequence, openAIClosed: true},
		{reason: "tool_use", neutral: llmprotocol.StopToolCall},
		{reason: "pause_turn", neutral: llmprotocol.StopPaused, openAIClosed: true},
		{reason: "refusal", neutral: llmprotocol.StopContentFilter},
		{reason: "model_context_window_exceeded", neutral: llmprotocol.StopContextWindow, openAIClosed: true},
	}
	for _, test := range tests {
		t.Run(test.reason, func(t *testing.T) { assertAnthropicTerminalReason(t, engine, test) })
	}
}

func TestOfficialAnthropicWhitespaceStopSequenceIsPreserved(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"done"}],"stop_reason":"stop_sequence","stop_sequence":" ","usage":{"input_tokens":1,"output_tokens":1}}`)
	response, _, _, err := NewBuiltinEngine().DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if response.StopReason != llmprotocol.StopSequence || response.MatchedStopSequence != " " {
		t.Fatalf("decoded terminal = %q matched=%q", response.StopReason, response.MatchedStopSequence)
	}
}

type anthropicTerminalReasonCase struct {
	reason       string
	neutral      llmprotocol.StopReason
	openAIClosed bool
}

func assertAnthropicTerminalReason(t *testing.T, engine *Engine, test anthropicTerminalReasonCase) {
	t.Helper()
	stopSequence := "null"
	if test.reason == "stop_sequence" {
		stopSequence = `"END"`
	}
	body := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"done"}],"stop_reason":"` + test.reason + `","stop_sequence":` + stopSequence + `,"usage":{"input_tokens":1,"output_tokens":1}}`)
	decoded, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatal(err)
	}
	if decoded.StopReason != test.neutral {
		t.Fatalf("decoded stop reason = %q, want %q", decoded.StopReason, test.neutral)
	}
	if test.reason == "stop_sequence" && decoded.MatchedStopSequence != "END" {
		t.Fatalf("matched stop sequence = %q, want END", decoded.MatchedStopSequence)
	}
	assertAnthropicTerminalRoundTrip(t, engine, test.reason, body)
	if test.openAIClosed {
		assertAnthropicTerminalClosedForOpenAI(t, engine, test.reason, body)
	}
}

func assertAnthropicTerminalRoundTrip(t *testing.T, engine *Engine, reason string, body []byte) {
	t.Helper()
	roundTrip, err := engine.TranslateResponse(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.AnthropicMessagesV1,
		body,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var wire anthropicResponseWire
	if err := json.Unmarshal(roundTrip.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.StopReason == nil || *wire.StopReason != reason {
		t.Fatalf("round-trip stop reason = %v, body=%s", wire.StopReason, roundTrip.Body)
	}
	if reason == "stop_sequence" && (wire.StopSequence == nil || *wire.StopSequence != "END") {
		t.Fatalf("round-trip matched stop sequence = %v, body=%s", wire.StopSequence, roundTrip.Body)
	}
}

func assertAnthropicTerminalClosedForOpenAI(t *testing.T, engine *Engine, reason string, body []byte) {
	t.Helper()
	for _, target := range []llmprotocol.WireFormat{llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1} {
		if _, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, target, body, nil); err == nil {
			t.Fatalf("%s silently accepted unrepresentable stop reason %q", target, reason)
		}
	}
}

func TestOfficialResponsesContentFilterIncompleteRoundTrip(t *testing.T) {
	engine := NewBuiltinEngine()
	body := []byte(`{"id":"resp_1","object":"response","model":"m","status":"incomplete","output":[],"incomplete_details":{"reason":"content_filter"},"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}`)
	translated, err := engine.TranslateResponse(
		llmprotocol.OpenAIResponsesV1,
		llmprotocol.OpenAIResponsesV1,
		body,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	if translated.Response.StopReason != llmprotocol.StopContentFilter {
		t.Fatalf("decoded stop reason = %q", translated.Response.StopReason)
	}
	var wire responsesResponseWire
	if err := json.Unmarshal(translated.Body, &wire); err != nil {
		t.Fatal(err)
	}
	if wire.Status != "incomplete" || wire.IncompleteDetails == nil || wire.IncompleteDetails.Reason != "content_filter" {
		t.Fatalf("content-filter terminal state changed: %s", translated.Body)
	}
}

func TestOfficialEncodedResponseResourcesContainRequiredFields(t *testing.T) {
	engine := NewBuiltinEngine()
	chat := []byte(`{"id":"response_1","object":"chat.completion","created":100,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`)
	assertEncodedResponsesResource(t, engine, chat)
	assertEncodedChatResource(t, engine)
	assertEncodedAnthropicResource(t, engine, chat)
}

func assertEncodedResponsesResource(t *testing.T, engine *Engine, chat []byte) {
	t.Helper()
	responses, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.OpenAIResponsesV1,
		chat,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var response map[string]json.RawMessage
	if err := json.Unmarshal(responses.Body, &response); err != nil {
		t.Fatal(err)
	}
	assertJSONFields(t, response, []string{
		"created_at", "error", "id", "incomplete_details", "instructions", "metadata", "model",
		"object", "output", "parallel_tool_calls", "temperature", "tool_choice", "tools", "top_p",
	}, "official Responses resource", responses.Body)
	var output []map[string]json.RawMessage
	if err := json.Unmarshal(response["output"], &output); err != nil || len(output) != 1 {
		t.Fatalf("official Responses output is invalid: %v body=%s", err, responses.Body)
	}
	assertJSONFields(
		t, output[0], []string{"content", "id", "role", "status", "type"},
		"official Responses output message", responses.Body,
	)
	assertResponsesUsageFields(t, response["usage"])
}

func assertResponsesUsageFields(t *testing.T, body json.RawMessage) {
	t.Helper()
	var usage struct {
		InputDetails  map[string]json.RawMessage `json:"input_tokens_details"`
		OutputDetails map[string]json.RawMessage `json:"output_tokens_details"`
	}
	if err := json.Unmarshal(body, &usage); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{"cached_tokens", "cache_write_tokens"} {
		if _, found := usage.InputDetails[field]; !found {
			t.Errorf("Responses input usage omitted required field %q", field)
		}
	}
	if _, found := usage.OutputDetails["reasoning_tokens"]; !found {
		t.Error("Responses output usage omitted required reasoning_tokens")
	}
}

func assertJSONFields(
	t *testing.T,
	object map[string]json.RawMessage,
	fields []string,
	label string,
	body []byte,
) {
	t.Helper()
	for _, field := range fields {
		if _, found := object[field]; !found {
			t.Errorf("%s omitted required field %q: %s", label, field, body)
		}
	}
}

func assertEncodedChatResource(t *testing.T, engine *Engine) {
	t.Helper()
	translatedChat, err := engine.TranslateResponse(
		llmprotocol.AnthropicMessagesV1,
		llmprotocol.OpenAIChatV1,
		[]byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn","usage":{"input_tokens":2,"output_tokens":1}}`),
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var chatResource map[string]json.RawMessage
	if err := json.Unmarshal(translatedChat.Body, &chatResource); err != nil {
		t.Fatal(err)
	}
	if _, found := chatResource["created"]; !found {
		t.Fatalf("official Chat resource omitted required created field: %s", translatedChat.Body)
	}
}

func assertEncodedAnthropicResource(t *testing.T, engine *Engine, chat []byte) {
	t.Helper()
	translatedAnthropic, err := engine.TranslateResponse(
		llmprotocol.OpenAIChatV1,
		llmprotocol.AnthropicMessagesV1,
		chat,
		func(*llmprotocol.Response) error { return nil },
	)
	if err != nil {
		t.Fatal(err)
	}
	var anthropicResource map[string]json.RawMessage
	if err := json.Unmarshal(translatedAnthropic.Body, &anthropicResource); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{
		"container", "content", "id", "model", "role", "stop_details", "stop_reason",
		"stop_sequence", "type", "usage",
	} {
		if _, found := anthropicResource[field]; !found {
			t.Errorf("official Anthropic resource omitted required field %q: %s", field, translatedAnthropic.Body)
		}
	}
	var anthropicUsage map[string]json.RawMessage
	if err := json.Unmarshal(anthropicResource["usage"], &anthropicUsage); err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{
		"cache_creation", "cache_creation_input_tokens", "cache_read_input_tokens", "inference_geo",
		"input_tokens", "output_tokens", "output_tokens_details", "server_tool_use", "service_tier",
	} {
		if _, found := anthropicUsage[field]; !found {
			t.Errorf("official Anthropic usage omitted required field %q: %s", field, translatedAnthropic.Body)
		}
	}
}

func jsonSemanticallyEqual(left, right []byte) bool {
	var leftValue, rightValue any
	return json.Unmarshal(left, &leftValue) == nil && json.Unmarshal(right, &rightValue) == nil &&
		reflect.DeepEqual(leftValue, rightValue)
}

func assertOfficialChatIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	chatBody := []byte(`{"model":"m","messages":[{"role":"user","content":"hello"}],"store":true,"user":"user-1"}`)
	chat, _, _, err := engine.DecodeRequest(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatal(err)
	}
	if chat.Store == nil || !*chat.Store || chat.EndUserID != "user-1" {
		t.Fatalf("Chat identity or store semantics were lost: %+v", chat)
	}
	translated, err := engine.TranslateRequest(llmprotocol.OpenAIChatV1, llmprotocol.OpenAIResponsesV1, chatBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	var responses map[string]any
	if json.Unmarshal(translated.Body, &responses) != nil || responses["store"] != true || responses["user"] != "user-1" {
		t.Fatalf("Chat to Responses identity/state translation = %s", translated.Body)
	}
}

func assertOfficialResponsesIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	responsesBody := []byte(`{"model":"m","input":"hello","conversation":{"id":"conv-1"},"reasoning":{"effort":"high"},"truncation":"auto","user":"user-2"}`)
	request, envelope, _, err := engine.DecodeRequest(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	if request.ConversationID != "conv-1" || request.ReasoningEffort != "high" || request.Truncation != "auto" || request.EndUserID != "user-2" {
		t.Fatalf("Responses semantic fields were lost: %+v", request)
	}
	request.Generation++
	encoded, err := engine.EncodeRequest(llmprotocol.OpenAIResponsesV1, request, envelope)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip map[string]any
	if json.Unmarshal(encoded.Body, &roundTrip) != nil || roundTrip["conversation"] != "conv-1" || roundTrip["truncation"] != "auto" || roundTrip["user"] != "user-2" {
		t.Fatalf("Responses semantic re-encoding = %s", encoded.Body)
	}
}

func assertOfficialAnthropicIdentityState(t *testing.T, engine *Engine) {
	t.Helper()
	anthropicBody := []byte(`{"model":"m","max_tokens":16,"metadata":{"user_id":"user-3"},"messages":[{"role":"user","content":"hello"}]}`)
	toChat, err := engine.TranslateRequest(llmprotocol.AnthropicMessagesV1, llmprotocol.OpenAIChatV1, anthropicBody, nil)
	if err != nil {
		t.Fatal(err)
	}
	var chatWire map[string]any
	if json.Unmarshal(toChat.Body, &chatWire) != nil || chatWire["user"] != "user-3" {
		t.Fatalf("Messages to Chat end-user translation = %s", toChat.Body)
	}
}

func TestOfficialProviderResponseShapesDecodeWithoutSilentLoss(t *testing.T) {
	engine := NewBuiltinEngine()
	assertOfficialResponsesProviderShape(t, engine)
	assertOfficialAnthropicProviderShape(t, engine)
}

func assertOfficialResponsesProviderShape(t *testing.T, engine *Engine) {
	t.Helper()
	responsesBody := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"completed_at":2,"status":"completed",
		"error":null,"incomplete_details":null,"instructions":"answer","max_output_tokens":64,
		"model":"m","output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"done","annotations":[]}]}],
		"parallel_tool_calls":true,"previous_response_id":"resp_0","reasoning":{"effort":"high"},
		"store":true,"temperature":0.2,"text":{"format":{"type":"text"}},"tool_choice":"auto",
		"tools":[],"top_p":0.9,"truncation":"disabled","usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8},
		"user":"user-1","metadata":{"trace":"1"}
	}`)
	response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	if len(response.Output) != 1 || response.Output[0].Content[0].Text != "done" || len(diagnostics) == 0 {
		t.Fatalf("Responses output or explicit omission diagnostics missing: response=%+v diagnostics=%+v", response, diagnostics)
	}
}

func assertOfficialAnthropicProviderShape(t *testing.T, engine *Engine) {
	t.Helper()
	anthropicBody := []byte(`{
		"id":"msg_1","type":"message","role":"assistant","model":"claude-test","container":{},
		"content":[{"type":"text","text":"done"}],"stop_details":null,"stop_reason":"end_turn","stop_sequence":null,
		"usage":{"cache_creation":{"ephemeral_1h_input_tokens":1,"ephemeral_5m_input_tokens":2},
		"cache_creation_input_tokens":3,"cache_read_input_tokens":4,"inference_geo":"us","input_tokens":10,
		"output_tokens":6,"output_tokens_details":{"thinking_tokens":2},"server_tool_use":{"web_search_requests":1},
		"service_tier":"standard"}
	}`)
	response, _, diagnostics, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, anthropicBody)
	if err != nil {
		t.Fatal(err)
	}
	if tokenValue(response.Usage.InputUncached) != 10 || tokenValue(response.Usage.InputCacheWrite) != 3 ||
		tokenValue(response.Usage.InputCacheRead) != 4 || tokenValue(response.Usage.InputTotal) != 17 ||
		tokenValue(response.Usage.OutputReasoning) != 2 || tokenValue(response.Usage.OutputOther) != 4 ||
		tokenValue(response.Usage.Total) != 23 || len(diagnostics) == 0 {
		t.Fatalf("Anthropic accounting or explicit omission diagnostics missing: usage=%+v diagnostics=%+v", response.Usage, diagnostics)
	}
}

func TestOfficialResponseMetadataIsExplicitlyDiagnosed(t *testing.T) {
	engine := NewBuiltinEngine()
	chatBody := []byte(`{
		"id":"chatcmpl_1","object":"chat.completion","created":1,"model":"m",
		"choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],
		"metadata":{"trace":"1"},"moderation":{"status":"passed"},
		"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3,"compute_units":4}
	}`)
	_, _, diagnostics, err := engine.DecodeResponse(llmprotocol.OpenAIChatV1, chatBody)
	if err != nil {
		t.Fatal(err)
	}
	assertDiagnosticFields(t, diagnostics, "metadata", "moderation", "usage.compute_units")

	responsesBody := []byte(`{
		"id":"resp_1","object":"response","created_at":1,"status":"completed","model":"m",
		"output":[{"type":"message","id":"msg_1","role":"assistant","status":"completed","phase":"final_answer","content":[{"type":"output_text","text":"done","annotations":[],"logprobs":[]}]}],
		"background":false,"conversation":{"id":"conv_1"},"max_tool_calls":4,"moderation":{"status":"passed"},
		"output_text":"done","prompt":{"id":"pmpt_1"},"prompt_cache_key":"cache_1",
		"prompt_cache_options":{"retention":"24h"},"prompt_cache_retention":"24h",
		"safety_identifier":"safe_1","service_tier":"default","top_logprobs":0,
		"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3,"compute_units":4}
	}`)
	_, _, diagnostics, err = engine.DecodeResponse(llmprotocol.OpenAIResponsesV1, responsesBody)
	if err != nil {
		t.Fatal(err)
	}
	assertDiagnosticFields(
		t, diagnostics, "background", "conversation", "max_tool_calls", "moderation", "output.content.logprobs", "output.phase", "output_text",
		"output.status", "prompt", "prompt_cache_key", "prompt_cache_options", "prompt_cache_retention",
		"safety_identifier", "service_tier", "top_logprobs", "usage.compute_units",
	)
}

func assertDiagnosticFields(t *testing.T, diagnostics llmprotocol.Diagnostics, expected ...string) {
	t.Helper()
	actual := make([]string, 0, len(diagnostics))
	for _, diagnostic := range diagnostics {
		actual = append(actual, diagnostic.Field)
	}
	sort.Strings(actual)
	sort.Strings(expected)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("diagnostic fields = %v, want %v", actual, expected)
	}
}

func assertProtocolError(t *testing.T, err error, category llmprotocol.ErrorCategory, code string) {
	t.Helper()
	var protocolError *llmprotocol.ProtocolError
	if !errors.As(err, &protocolError) || protocolError.Category != category || code != "" && protocolError.Code != code {
		t.Fatalf("returned %T %v, want %s/%s", err, err, category, code)
	}
}

func fields(values ...string) []string {
	result := append([]string(nil), values...)
	sort.Strings(result)
	return result
}

func assertClosedDiscriminatorInventory(
	t *testing.T,
	name string,
	officialCount int,
	supported []string,
	unsupported []string,
) {
	t.Helper()
	seen := make(map[string]string, officialCount)
	for _, group := range []struct {
		name   string
		values []string
	}{
		{name: "supported", values: supported},
		{name: "unsupported", values: unsupported},
	} {
		for _, value := range group.values {
			if previous, ok := seen[value]; ok {
				t.Fatalf("%s discriminator %q appears in both %s and %s inventories", name, value, previous, group.name)
			}
			seen[value] = group.name
		}
	}
	if len(seen) != officialCount {
		t.Fatalf("%s discriminator inventory has %d unique variants, want %d", name, len(seen), officialCount)
	}
}

func assertClosedFieldDisposition(
	t *testing.T,
	name string,
	official []string,
	dispositions map[string][]string,
) {
	t.Helper()
	seen := make(map[string]string, len(official))
	for disposition, values := range dispositions {
		for _, value := range values {
			if previous, ok := seen[value]; ok {
				t.Fatalf("%s field %q appears in both %s and %s dispositions", name, value, previous, disposition)
			}
			seen[value] = disposition
		}
	}
	actual := make([]string, 0, len(seen))
	for field := range seen {
		actual = append(actual, field)
	}
	sort.Strings(actual)
	if !reflect.DeepEqual(actual, official) {
		t.Fatalf("%s field dispositions are incomplete\n got: %v\nwant: %v", name, actual, official)
	}
}

func jsonFieldNames(value reflect.Type) []string {
	result := make([]string, 0, value.NumField())
	for index := 0; index < value.NumField(); index++ {
		name := value.Field(index).Tag.Get("json")
		if comma := len(name); comma > 0 {
			for offset, char := range name {
				if char == ',' {
					name = name[:offset]
					break
				}
			}
		}
		if name != "" && name != "-" {
			result = append(result, name)
		}
	}
	sort.Strings(result)
	return result
}

func cloneJSONMap(source map[string]any) map[string]any {
	result := make(map[string]any, len(source)+1)
	for key, value := range source {
		result[key] = value
	}
	return result
}
