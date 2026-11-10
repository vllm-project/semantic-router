package extproc

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestToolResultPayloadReachesPIIChannelAcrossWireFormats(t *testing.T) {
	const wantPayload = "customer@example.com"
	tests := []struct {
		name           string
		format         llmprotocol.WireFormat
		body           string
		wantIncomplete bool
	}{
		{
			name:   "openai chat",
			format: llmprotocol.OpenAIChatV1,
			body:   `{"model":"client-model","messages":[{"role":"user","content":"lookup customer"},{"role":"assistant","tool_calls":[{"id":"call-1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call-1","content":"customer@example.com"}]}`,
		},
		{
			name:           "openai responses multimodal",
			format:         llmprotocol.OpenAIResponsesV1,
			wantIncomplete: true,
			body:           `{"model":"client-model","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"lookup customer"}]},{"type":"function_call","id":"item-1","call_id":"call-1","name":"lookup","arguments":"{}"},{"type":"function_call_output","call_id":"call-1","output":[{"type":"input_text","text":"customer@example.com"},{"type":"input_image","image_url":"https://example.invalid/customer.png"}]}]}`,
		},
		{
			name:           "anthropic messages multimodal",
			format:         llmprotocol.AnthropicMessagesV1,
			wantIncomplete: true,
			body:           `{"model":"client-model","max_tokens":32,"messages":[{"role":"user","content":[{"type":"text","text":"lookup customer"}]},{"role":"assistant","content":[{"type":"tool_use","id":"call-1","name":"lookup","input":{}}]},{"role":"user","content":[{"type":"tool_result","tool_use_id":"call-1","content":[{"type":"text","text":"customer@example.com"},{"type":"image","source":{"type":"url","url":"https://example.invalid/customer.png"}}]}]}]}`,
		},
	}

	router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(NewMockResponseStore())}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := &RequestContext{
				RequestID:    "tool-result-wire-format-test",
				SourceFormat: test.format,
				TraceContext: context.Background(),
			}
			request, immediate := router.prepareProtocolRequest([]byte(test.body), ctx)
			if immediate != nil {
				t.Fatalf("prepareProtocolRequest() returned immediate response")
			}
			if request == nil {
				t.Fatal("prepareProtocolRequest() returned nil request")
			}

			snapshot := extractSemanticRequestSignals(request)
			history := signalConversationHistoryFromRequest(request, snapshot)
			if !reflect.DeepEqual(history.toolResultTexts, []string{wantPayload}) {
				t.Fatalf("toolResultTexts = %#v, want [%q]", history.toolResultTexts, wantPayload)
			}
			if history.toolResultScanIncomplete != test.wantIncomplete {
				t.Fatalf("toolResultScanIncomplete = %t, want %t", history.toolResultScanIncomplete, test.wantIncomplete)
			}
			for _, message := range history.nonUserMessages {
				if strings.Contains(message, wantPayload) {
					t.Fatalf("tool result payload leaked into nonUserMessages: %q", message)
				}
			}
		})
	}
}

func TestExtractSignalConversationHistoryKeepsToolResultPayloadInPIIChannel(t *testing.T) {
	toolPayload := "customer@example.com"
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue"}},
			},
			{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{{
					Kind:     llmprotocol.ContentToolCall,
					ToolCall: &llmprotocol.ToolCall{ID: "call-1", Name: "lookup"},
				}},
			},
			{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolResult,
					ToolResult: &llmprotocol.ToolResult{
						CallID:  "call-1",
						Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: toolPayload}},
					},
				}},
			},
		},
	}

	history := extractSignalConversationHistory(req)
	if !reflect.DeepEqual(history.toolResultTexts, []string{toolPayload}) {
		t.Fatalf("toolResultTexts = %#v, want [%q]", history.toolResultTexts, toolPayload)
	}
	for _, message := range history.nonUserMessages {
		if strings.Contains(message, toolPayload) {
			t.Fatalf("tool result payload leaked into nonUserMessages: %q", message)
		}
	}
}

func TestSignalConversationHistoryFromRequestKeepsSnapshotPayloadFree(t *testing.T) {
	toolPayload := "customer@example.com"
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{
			{
				Role:    llmprotocol.RoleUser,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "continue"}},
			},
			{
				Role: llmprotocol.RoleTool,
				Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentToolResult,
					ToolResult: &llmprotocol.ToolResult{
						Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: toolPayload}},
					},
				}},
			},
		},
	}

	snapshot := extractSemanticRequestSignals(req)
	history := signalConversationHistoryFromRequest(req, snapshot)
	if !reflect.DeepEqual(history.toolResultTexts, []string{toolPayload}) {
		t.Fatalf("toolResultTexts = %#v, want [%q]", history.toolResultTexts, toolPayload)
	}
	for _, message := range history.nonUserMessages {
		if strings.Contains(message, toolPayload) {
			t.Fatalf("tool result payload leaked into nonUserMessages: %q", message)
		}
	}
}

func TestSignalConversationHistoryReportsIncompleteToolResultExtraction(t *testing.T) {
	req := &llmprotocol.Request{
		Messages: []llmprotocol.Message{{
			Role: llmprotocol.RoleTool,
			Content: []llmprotocol.Content{{
				Kind: llmprotocol.ContentToolResult,
				ToolResult: &llmprotocol.ToolResult{Content: []llmprotocol.Content{{
					Kind: llmprotocol.ContentImage,
					URL:  "https://example.invalid/customer.png",
				}}},
			}},
		}},
	}

	history := extractSignalConversationHistory(req)
	if len(history.toolResultTexts) != 0 {
		t.Fatalf("toolResultTexts = %#v, want no text blocks", history.toolResultTexts)
	}
	if !history.toolResultScanIncomplete {
		t.Fatal("unsupported tool-result content must be reported as incomplete")
	}
}
