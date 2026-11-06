package extproc

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func hallucinationTextMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{
		Role:    role,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}
}

func hallucinationToolResultMessage(callID, text string) llmprotocol.Message {
	return llmprotocol.Message{
		Role: llmprotocol.RoleTool,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentToolResult,
			ToolResult: &llmprotocol.ToolResult{
				CallID:  callID,
				Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
			},
		}},
	}
}

func hallucinationAssistantResponse(text string) *llmprotocol.Response {
	return &llmprotocol.Response{Output: []llmprotocol.OutputItem{{
		Role:    llmprotocol.RoleAssistant,
		Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}},
	}}}
}

// TestHallucinationExtproc is removed - tests are now part of the main ExtProc Suite in extproc_test.go
// This avoids the "Rerunning Suite" error from Ginkgo when multiple RunSpecs are called

var _ = Describe("extractSemanticToolResults", func() {
	It("should extract ordered text from neutral tool results", func() {
		messages := []llmprotocol.Message{
			hallucinationTextMessage(llmprotocol.RoleUser, "What is the weather?"),
			{
				Role: llmprotocol.RoleAssistant,
				Content: []llmprotocol.Content{{
					Kind:     llmprotocol.ContentToolCall,
					ToolCall: &llmprotocol.ToolCall{ID: "call_1", Name: "weather", Arguments: `{}`},
				}},
			},
			hallucinationToolResultMessage("call_1", "The weather is sunny with 25°C"),
			hallucinationToolResultMessage("call_2", "Humidity is 65%"),
		}

		results := extractSemanticToolResults(messages)

		Expect(results).To(HaveLen(2))
		Expect(results[0]).To(Equal("The weather is sunny with 25°C"))
		Expect(results[1]).To(Equal("Humidity is 65%"))
	})

	It("should return empty slice when no tool results exist", func() {
		messages := []llmprotocol.Message{
			hallucinationTextMessage(llmprotocol.RoleUser, "Hello"),
			hallucinationTextMessage(llmprotocol.RoleAssistant, "Hi there!"),
		}

		results := extractSemanticToolResults(messages)
		Expect(results).To(BeEmpty())
	})

	It("should skip empty tool result content", func() {
		messages := []llmprotocol.Message{
			hallucinationToolResultMessage("call_1", ""),
			hallucinationToolResultMessage("call_2", "Valid content"),
		}

		results := extractSemanticToolResults(messages)
		Expect(results).To(HaveLen(1))
		Expect(results[0]).To(Equal("Valid content"))
	})

	It("should ignore incomplete neutral tool result blocks", func() {
		messages := []llmprotocol.Message{
			{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult}}},
			hallucinationToolResultMessage("call_2", "Valid"),
		}

		results := extractSemanticToolResults(messages)
		Expect(results).To(HaveLen(1))
	})
})

var _ = Describe("semanticAssistantContent", func() {
	It("should extract client-visible content from neutral assistant output", func() {
		response := hallucinationAssistantResponse("The capital of France is Paris.")
		content := semanticAssistantContent(response)

		Expect(content).To(Equal("The capital of France is Paris."))
	})

	It("should keep reasoning separate from delivered content", func() {
		response := hallucinationAssistantResponse("Visible answer")
		response.Output[0].Content = append([]llmprotocol.Content{{
			Kind: llmprotocol.ContentReasoning,
			Text: "Hidden reasoning",
		}}, response.Output[0].Content...)

		Expect(semanticAssistantContent(response)).To(Equal("Visible answer"))
	})

	It("should return empty content for nil or non-assistant output", func() {
		Expect(semanticAssistantContent(nil)).To(BeEmpty())
		Expect(semanticAssistantContent(&llmprotocol.Response{Output: []llmprotocol.OutputItem{{
			Role:    llmprotocol.RoleUser,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "not an answer"}},
		}}})).To(BeEmpty())
	})
})

var _ = Describe("Fact-check request state", func() {
	var router *OpenAIRouter

	BeforeEach(func() {
		router = &OpenAIRouter{Config: &config.RouterConfig{}}
	})

	Describe("checkRequestHasTools", func() {
		It("should detect tools in request", func() {
			ctx := &RequestContext{
				SemanticRequest: &llmprotocol.Request{
					Generation: 1,
					Messages:   []llmprotocol.Message{neutralTextMessage(llmprotocol.RoleUser, "What is the weather?")},
					Tools:      []llmprotocol.Tool{{Name: "get_weather", InputSchema: json.RawMessage(`{"type":"object"}`)}},
				},
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeTrue())
		})

		It("should extract tool results from messages", func() {
			ctx := &RequestContext{
				SemanticRequest: &llmprotocol.Request{Generation: 1, Messages: []llmprotocol.Message{{
					Role: llmprotocol.RoleTool,
					Content: []llmprotocol.Content{{
						Kind: llmprotocol.ContentToolResult,
						ToolResult: &llmprotocol.ToolResult{CallID: "call_weather", Content: []llmprotocol.Content{{
							Kind: llmprotocol.ContentText, Text: "The weather is sunny, 25°C in Paris",
						}}},
					}},
				}}},
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeTrue())
		})

		It("should handle request without tools", func() {
			ctx := &RequestContext{
				SemanticRequest: testNeutralRequest("gpt-4", "Hello"),
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeFalse())
		})

		It("should handle nil/empty body", func() {
			ctx := &RequestContext{}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeFalse())
		})
	})
})
