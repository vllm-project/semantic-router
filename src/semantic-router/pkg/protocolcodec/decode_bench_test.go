package protocolcodec

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// agentChatBody builds a coding-agent style chat request: a long system
// prompt, a set of tools with JSON schemas, and `turns` rounds of user text,
// an assistant tool call and the tool's result.
func agentChatBody(turns int) []byte {
	schema := `{"type":"object","properties":{"path":{"type":"string","description":"File path relative to the repository root"},"start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1},"pattern":{"type":"string"},"recursive":{"type":"boolean"}},"required":["path"],"additionalProperties":false}`
	tools := make([]string, 0, 12)
	for i := 0; i < 12; i++ {
		tools = append(tools, fmt.Sprintf(`{"type":"function","function":{"name":"tool_%d","description":"%s","parameters":%s}}`,
			i, strings.Repeat("Reads or searches files in the working tree. ", 4), schema))
	}
	messages := []string{
		fmt.Sprintf(`{"role":"system","content":%q}`, strings.Repeat("You are a careful coding agent. Follow the repository conventions and explain changes briefly. ", 24)),
	}
	result := strings.Repeat("func handler(w http.ResponseWriter, r *http.Request) {\n\tif err := decode(r.Body); err != nil {\n\t\treturn\n\t}\n}\n", 16)
	for i := 0; i < turns; i++ {
		messages = append(messages,
			fmt.Sprintf(`{"role":"user","content":[{"type":"text","text":%q}]}`, "Look at the request handler and find where the body is decoded."),
			fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":"call_%d","type":"function","function":{"name":"tool_%d","arguments":%q}}]}`,
				i, i%12, `{"path":"src/server/handler.go","start_line":1,"end_line":200}`),
			fmt.Sprintf(`{"role":"tool","tool_call_id":"call_%d","content":%q}`, i, result),
		)
	}
	return []byte(fmt.Sprintf(`{"model":"agent-model","stream":true,"max_tokens":4096,"tools":[%s],"messages":[%s]}`,
		strings.Join(tools, ","), strings.Join(messages, ",")))
}

// About 9 KB, 52 KB and 178 KB.
var agentTurns = []int{0, 19, 75}

func BenchmarkDecodeAgentChatRequest(b *testing.B) {
	engine := NewBuiltinEngine()
	policy := llmprotocol.DefaultPolicy()
	for _, turns := range agentTurns {
		body := agentChatBody(turns)
		name := fmt.Sprintf("%dKB", len(body)/1000)
		b.Run(fmt.Sprintf("%s/DecodeRequestForMutation", name), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				if _, _, _, err := engine.DecodeRequestForMutation(llmprotocol.OpenAIChatV1, body); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("%s/decodeWire", name), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				var wire chatRequestWire
				if err := decodeWire(body, &wire, policy); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("%s/validateNoDuplicateKeys", name), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				if err := validateNoDuplicateKeys(body, policy.Limits.JSONDepth); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("%s/validateExactJSONFieldNames", name), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				if err := validateExactJSONFieldNames(body, reflect.TypeOf(&chatRequestWire{})); err != nil {
					b.Fatal(err)
				}
			}
		})
		b.Run(fmt.Sprintf("%s/jsonUnmarshal", name), func(b *testing.B) {
			b.SetBytes(int64(len(body)))
			b.ReportAllocs()
			for b.Loop() {
				var wire chatRequestWire
				if err := json.Unmarshal(body, &wire); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkDecodeChatStreamChunk(b *testing.B) {
	policy := llmprotocol.DefaultPolicy()
	chunk := []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1760000000,"model":"agent-model","choices":[{"index":0,"delta":{"content":"The body is decoded in handler.go before validation."},"finish_reason":null}]}`)
	b.Run("decodeProviderWireVendorAware", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var wire chatChunkWire
			if _, _, err := decodeProviderWireVendorAware(chunk, &wire, policy); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("jsonUnmarshal", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			var wire chatChunkWire
			if err := json.Unmarshal(chunk, &wire); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// agentChatResponse builds a buffered chat completion with an answer of about
// `paragraphs` paragraphs and one tool call.
func agentChatResponse(paragraphs int) []byte {
	answer := strings.Repeat("The request body is decoded in handler.go and validated before routing. ", 12*paragraphs)
	return []byte(fmt.Sprintf(`{"id":"chatcmpl-1","object":"chat.completion","created":1760000000,"model":"agent-model","choices":[{"index":0,"message":{"role":"assistant","content":%q,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"tool_1","arguments":%q}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":5210,"completion_tokens":%d,"total_tokens":%d}}`,
		answer, `{"path":"src/server/handler.go","start_line":1,"end_line":200}`, 180*paragraphs, 5210+180*paragraphs))
}

func BenchmarkDecodeChatResponse(b *testing.B) {
	azure := llmprotocol.DefaultPolicy()
	azure.ResponseVendor = llmprotocol.ResponseVendorAzure
	policies := []struct {
		name   string
		policy llmprotocol.Policy
	}{{"default", llmprotocol.DefaultPolicy()}, {"azure", azure}}
	for _, paragraphs := range []int{1, 8} {
		body := agentChatResponse(paragraphs)
		for _, policy := range policies {
			b.Run(fmt.Sprintf("%dKB/%s", len(body)/1000, policy.name), func(b *testing.B) {
				b.SetBytes(int64(len(body)))
				b.ReportAllocs()
				for b.Loop() {
					var wire chatResponseWire
					if _, _, err := decodeProviderWireVendorAware(body, &wire, policy.policy); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
