package protocolcodec

import (
	"bytes"
	"encoding/json"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func anthropicResponseWithContent(content string) []byte {
	return []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[` + content +
		`],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
}

func anthropicRequestWithAssistantContent(content string) []byte {
	return []byte(`{"model":"m","max_tokens":16,"messages":[{"role":"user","content":"hi"},` +
		`{"role":"assistant","content":[` + content + `]}]}`)
}

func TestAnthropicResponseToolUseAcceptsToolsetName(t *testing.T) {
	engine := NewBuiltinEngine()
	body := anthropicResponseWithContent(
		`{"type":"tool_use","id":"call_1","name":"lookup","input":{"city":"Paris"},"toolset_name":"web"}`,
	)
	response, _, _, err := engine.DecodeResponse(llmprotocol.AnthropicMessagesV1, body)
	if err != nil {
		t.Fatalf("tool_use toolset_name rejected on provider response: %v", err)
	}
	if len(response.Output) != 1 || len(response.Output[0].Content) != 1 ||
		response.Output[0].Content[0].ToolCall == nil ||
		response.Output[0].Content[0].ToolCall.ID != "call_1" {
		t.Fatalf("decoded tool call changed: %+v", response.Output)
	}
	roundTrip, err := engine.TranslateResponse(llmprotocol.AnthropicMessagesV1, llmprotocol.AnthropicMessagesV1, body, nil)
	if err != nil {
		t.Fatalf("same-format tool response translation failed: %v", err)
	}
	if !bytes.Equal(roundTrip.Body, body) {
		t.Fatalf("same-format response did not preserve toolset_name: %s", roundTrip.Body)
	}
}

func TestAnthropicResponseTextCitations(t *testing.T) {
	engine := NewBuiltinEngine()
	if _, _, _, err := engine.DecodeResponse(
		llmprotocol.AnthropicMessagesV1,
		anthropicResponseWithContent(`{"type":"text","text":"hello","citations":null}`),
	); err != nil {
		t.Fatalf("null citations rejected on provider response: %v", err)
	}
	_, _, _, err := engine.DecodeResponse(
		llmprotocol.AnthropicMessagesV1,
		anthropicResponseWithContent(`{"type":"text","text":"hello","citations":[{"type":"char_location","cited_text":"h"}]}`),
	)
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
}

func TestAnthropicRequestStillRejectsResponseOnlyExtensions(t *testing.T) {
	engine := NewBuiltinEngine()
	_, _, _, err := engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, anthropicRequestWithAssistantContent(
		`{"type":"tool_use","id":"call_1","name":"lookup","input":{},"toolset_name":"web"}`,
	))
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_content_toolset_name")

	_, _, _, err = engine.DecodeRequest(llmprotocol.AnthropicMessagesV1, anthropicRequestWithAssistantContent(
		`{"type":"text","text":"hello","citations":[{"type":"char_location","cited_text":"h"}]}`,
	))
	assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "unsupported_citations")
}

// Sample values that are valid for every variant carrying the field.
var anthropicContentFieldSamples = map[string]string{
	"cache_control":   `{"type":"ephemeral"}`,
	"caller":          `{"type":"direct"}`,
	"citations":       `[{"type":"char_location","cited_text":"h"}]`,
	"content":         `"sunny"`,
	"context":         `"background"`,
	"id":              `"call_1"`,
	"input":           `{}`,
	"is_error":        `false`,
	"name":            `"lookup"`,
	"signature":       `"sig"`,
	"source":          `{"type":"url","url":"https://example.com/a.png"}`,
	"text":            `"hello"`,
	"thinking":        `"work"`,
	"title":           `"doc"`,
	"tool_use_id":     `"call_1"`,
	"toolset_name":    `"web"`,
	"transformations": `[{"type":"crop"}]`,
}

func anthropicContentBlockWith(typeName, extra string) json.RawMessage {
	parts := []string{`"type":"` + typeName + `"`}
	for _, name := range anthropicRequiredContentFields[typeName] {
		parts = append(parts, `"`+name+`":`+anthropicContentFieldSamples[name])
	}
	if extra != "" && extra != "type" && !slices.Contains(anthropicRequiredContentFields[typeName], extra) {
		parts = append(parts, `"`+extra+`":`+anthropicContentFieldSamples[extra])
	}
	return json.RawMessage("{" + strings.Join(parts, ",") + "}")
}

// Every field a row admits must survive both validation steps, so the steps cannot drift apart again.
func TestAnthropicContentFieldRulesAgreeAcrossValidationSteps(t *testing.T) {
	directions := []struct {
		name           string
		providerOutput bool
		rows           map[string]map[string]anthropicFieldRule
	}{
		{"request", false, anthropicRequestContentFields},
		{"response", true, anthropicResponseContentFields},
	}
	extensionFields := make(map[string]bool, len(anthropicExtensionFields))
	for _, field := range anthropicExtensionFields {
		extensionFields[field.name] = true
	}
	for _, direction := range directions {
		for typeName, rules := range direction.rows {
			for name, rule := range rules {
				t.Run(direction.name+"/"+typeName+"/"+name, func(t *testing.T) {
					if !slices.Contains(anthropicKnownContentFields, name) {
						t.Fatalf("%s is not a known content field, so other variants would not reject it", name)
					}
					if _, ok := anthropicContentFieldSamples[name]; !ok && name != "type" {
						t.Fatalf("%s has no sample value", name)
					}
					content, err := decodeAnthropicContentBlock(
						anthropicContentBlockWith(typeName, name), typeName, llmprotocol.DefaultPolicy(), direction.providerOutput,
					)
					switch rule {
					case anthropicFieldAccepted:
						if err != nil {
							t.Fatalf("accepted field was rejected: %v", err)
						}
						if content.Kind == "" {
							t.Fatalf("accepted field decoded to empty content")
						}
					case anthropicFieldUnsupported:
						if !extensionFields[name] {
							t.Fatalf("unsupported field has no extension extractor, so it would pass silently")
						}
						assertProtocolError(t, err, llmprotocol.ErrorUnsupportedFeature, "")
						if direction.providerOutput && strings.Contains(err.Error(), "request contract") {
							t.Fatalf("provider response error names the request contract: %v", err)
						}
					default:
						t.Fatalf("unknown rule %d", rule)
					}
				})
			}
		}
	}
}
