package classification

import (
	"fmt"
	"strings"
	"testing"
)

func TestEstimateOpenAIRequestContextPlainTextPreservesColdStartHeuristic(t *testing.T) {
	estimate := EstimateOpenAIRequestContext([]byte(`{
		"model":"auto",
		"messages":[{"role":"user","content":"hello"}]
	}`))

	wantTextBytes := len("user") + len("hello")
	if estimate.TextBytes != wantTextBytes {
		t.Fatalf("text bytes = %d, want %d", estimate.TextBytes, wantTextBytes)
	}
	if estimate.StructuredBytes != 0 {
		t.Fatalf("structured bytes = %d, want 0", estimate.StructuredBytes)
	}
	wantFloor := 3 + RequestContextMessageFramingTokens
	if estimate.TokenFloor != wantFloor {
		t.Fatalf("token floor = %d, want %d", estimate.TokenFloor, wantFloor)
	}
	if estimate.EquivalentBytes != wantFloor*RequestContextBytesPerToken {
		t.Fatalf(
			"equivalent bytes = %d, want %d",
			estimate.EquivalentBytes,
			wantFloor*RequestContextBytesPerToken,
		)
	}
}

func TestEstimateOpenAIRequestContextCountsHistoryToolsAndExactNumbers(t *testing.T) {
	priorUser := strings.Repeat("p", 8_000)
	toolResult := strings.Repeat("r", 12_000)
	schemaDescription := strings.Repeat("s", 16_000)
	largeInteger := "90071992547409931234567890123456789"
	body := []byte(fmt.Sprintf(`{
		"model":"auto",
		"messages":[
			{"role":"user","content":%q},
			{"role":"assistant","content":null,"tool_calls":[{
				"type":"function",
				"function":{"name":"lookup","arguments":{"ticket":%s}}
			}]},
			{"role":"tool","content":%q},
			{"role":"user","content":"ok"}
		],
		"tools":[{"type":"function","function":{
			"name":"lookup",
			"description":%q,
			"parameters":{"type":"object","properties":{"ticket":{"type":"integer"}}}
		}}]
	}`, priorUser, largeInteger, toolResult, schemaDescription))

	estimate := EstimateOpenAIRequestContext(body)
	minimumTextBytes := len(priorUser) + len("lookup") + len("ok")
	if estimate.TextBytes < minimumTextBytes {
		t.Fatalf(
			"text bytes = %d, want at least %d",
			estimate.TextBytes,
			minimumTextBytes,
		)
	}
	minimumStructuredBytes := len(schemaDescription) + len(largeInteger) + len(toolResult)
	if estimate.StructuredBytes < minimumStructuredBytes {
		t.Fatalf(
			"structured bytes = %d, want at least %d",
			estimate.StructuredBytes,
			minimumStructuredBytes,
		)
	}
	if estimate.TokenFloor <= len("ok")/RequestContextBytesPerToken {
		t.Fatalf("token floor %d ignored prior/tool/schema context", estimate.TokenFloor)
	}
	if !strings.Contains(string(body), largeInteger) {
		t.Fatal("test fixture lost the exact large integer")
	}
}

func TestEstimateOpenAIRequestContextToolArgumentsKeepExactRawJSONString(t *testing.T) {
	const arguments = `"{\"ticket\":90071992547409931234567890123456789}"`
	body := []byte(`{
		"messages":[{
			"role":"assistant",
			"content":null,
			"tool_calls":[{"function":{"name":"lookup","arguments":` + arguments + `}}]
		}]
	}`)

	estimate := EstimateOpenAIRequestContext(body)
	if estimate.StructuredBytes != len(arguments) {
		t.Fatalf(
			"structured bytes = %d, want exact raw JSON-string size %d",
			estimate.StructuredBytes,
			len(arguments),
		)
	}
	wantTextBytes := len("assistant") + len("lookup")
	if estimate.TextBytes != wantTextBytes {
		t.Fatalf("text bytes = %d, want %d", estimate.TextBytes, wantTextBytes)
	}
}

func TestEstimateOpenAIRequestContextImageUsesFixedPrivateBudget(t *testing.T) {
	makeBody := func(payload string) []byte {
		return []byte(fmt.Sprintf(`{
			"messages":[{"role":"user","content":[
				{"type":"text","text":"hi"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}
			]}]
		}`, payload))
	}

	small := EstimateOpenAIRequestContext(makeBody("AAA"))
	large := EstimateOpenAIRequestContext(makeBody(strings.Repeat("PRIVATE", 20_000)))
	if small != large {
		t.Fatalf("image payload changed content-free estimate: small=%+v large=%+v", small, large)
	}
	if small.ImageCount != 1 {
		t.Fatalf("image count = %d, want 1", small.ImageCount)
	}
	wantFloor := RequestContextImageTokenBudget +
		2 + RequestContextMessageFramingTokens
	if small.TokenFloor != wantFloor {
		t.Fatalf("token floor = %d, want %d", small.TokenFloor, wantFloor)
	}
}

func TestEstimateOpenAIRequestContextUsesStructuredAndEffectiveOutputReservesOnce(t *testing.T) {
	body := []byte(`{
		"messages":[{"role":"user","content":"ok"}],
		"tools":[{"type":"function","function":{
			"name":"emit",
			"parameters":{"type":"object","properties":{"n":{
				"type":"integer","const":9007199254740993123456789
			}}}
		}}],
		"max_tokens":4096,
		"max_completion_tokens":2048
	}`)

	estimate := EstimateOpenAIRequestContext(body)
	if estimate.OutputTokenReserve != 2048 {
		t.Fatalf("output reserve = %d, want preferred value 2048", estimate.OutputTokenReserve)
	}
	if estimate.ToolDefinitionCount != 1 {
		t.Fatalf("tool definition count = %d, want 1", estimate.ToolDefinitionCount)
	}
	if estimate.StructuredBytes == 0 {
		t.Fatal("structured tool schema was not counted")
	}
	minimum := estimate.OutputTokenReserve + estimate.StructuredBytes + estimate.FramingTokens
	if estimate.TokenFloor < minimum {
		t.Fatalf("token floor = %d, want at least %d", estimate.TokenFloor, minimum)
	}
}

func TestEstimateOpenAIRequestContextCountsMalformedMessagesAndLegacyFunctionResults(t *testing.T) {
	const functionResult = `"{\"rows\":[1,2,3]}"`
	body := []byte(`{
		"messages":[
			9007199254740993123456789,
			{"role":"function","name":"lookup","content":` + functionResult + `}
		]
	}`)

	estimate := EstimateOpenAIRequestContext(body)
	minimumStructuredBytes := len("9007199254740993123456789") + len(functionResult)
	if estimate.StructuredBytes < minimumStructuredBytes {
		t.Fatalf(
			"structured bytes = %d, want at least exact malformed/function bytes %d",
			estimate.StructuredBytes,
			minimumStructuredBytes,
		)
	}
	if estimate.MessageCount != 2 {
		t.Fatalf("message count = %d, want 2", estimate.MessageCount)
	}
}

func TestEstimateAnthropicRequestContextCountsToolLoopAndUsesSameImageBudget(t *testing.T) {
	makeBody := func(payload string) []byte {
		return []byte(fmt.Sprintf(`{
			"system":"system context",
			"messages":[
				{"role":"user","content":"prior request"},
				{"role":"assistant","content":[{
					"type":"tool_use","name":"lookup","input":{"id":9007199254740993123456789}
				}]},
				{"role":"user","content":[{
					"type":"tool_result","content":{"rows":[1,2,3],"summary":"result"}
				}]},
				{"role":"user","content":[
					{"type":"text","text":"finish"},
					{"type":"image","source":{"type":"base64","data":"%s"}}
				]}
			],
			"tools":[{"name":"lookup","input_schema":{"type":"object"}}]
		}`, payload))
	}

	small := EstimateAnthropicRequestContext(makeBody("AAA"))
	large := EstimateAnthropicRequestContext(makeBody(strings.Repeat("PRIVATE", 20_000)))
	if small != large {
		t.Fatalf("Anthropic image payload changed estimate: small=%+v large=%+v", small, large)
	}
	if small.ImageCount != 1 || small.TokenFloor <= RequestContextImageTokenBudget {
		t.Fatalf("Anthropic context estimate omitted prompt components: %+v", small)
	}
}

// Reserves follow llmprotocol.InputEstimate: structured bytes beyond the text
// rate, image reserves and media payloads; prose has none.
func TestRequestContextReservesFollowTheInputEstimate(t *testing.T) {
	if estimate := EstimateAnthropicRequestContext([]byte(`{"max_tokens":16,"messages":[{"role":"user","content":"hello"}]}`)); estimate.ReserveTokens != 0 {
		t.Fatalf("prose reserve = %d, want 0", estimate.ReserveTokens)
	}
	input := `{"command":"` + strings.Repeat("x", 395) + `"}`
	document := `{"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"` + strings.Repeat("A", 1000) + `"}}`
	estimate := EstimateAnthropicRequestContext([]byte(`{"max_tokens":16,"messages":[
		{"role":"user","content":"hello"},
		{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"bash","input":` + input + `}]},
		{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AAAA"}},` + document + `]}]}`))
	// The tool input keeps a quarter at the text rate; the image and the
	// document are wholly reserve.
	want := len(input) - (len(input)+3)/4 + RequestContextImageTokenBudget + len(document)
	if estimate.ReserveTokens != want {
		t.Fatalf("reserve = %d, want %d (estimate %+v)", estimate.ReserveTokens, want, estimate)
	}
}
