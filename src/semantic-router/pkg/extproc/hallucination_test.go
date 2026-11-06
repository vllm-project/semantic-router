package extproc

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
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

// findProjectRoot finds the project root by looking for go.mod
func findProjectRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
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

var _ = Describe("FactCheckClassifier Integration", Label("model-artifacts"), func() {
	var (
		classifier *classification.FactCheckClassifier
		cfg        *config.FactCheckModelConfig
	)

	BeforeEach(func() {
		modelPath := resolveExtprocTestPath("../../../../models/mom-halugate-sentinel")
		skipExtprocSpecIfModelArtifactsMissing("Fact-check model", modelPath)
		cfg = &config.FactCheckModelConfig{
			ModelID:   modelPath,
			Threshold: 0.7,
		}
		var err error
		classifier, err = classification.NewFactCheckClassifier(cfg)
		Expect(err).NotTo(HaveOccurred())
		err = classifier.Initialize()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should classify factual questions", func() {
		// These questions should trigger fact-check patterns
		factualQuestions := []string{
			"When was the Eiffel Tower built?",
			"Who is the current CEO of Apple?",
			"What is the population of Tokyo?",
		}

		for _, q := range factualQuestions {
			result, err := classifier.Classify(context.Background(), q)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			// Just verify it returns valid result
			Expect(result.Label).To(BeElementOf(
				classification.FactCheckLabelNeeded,
				classification.FactCheckLabelNotNeeded,
			))
		}
	})

	It("should classify code/creative questions", func() {
		// These questions should NOT trigger fact-check
		codeQuestions := []string{
			"Write a Python function to sort a list",
			"Create a poem about the ocean",
			"Help me debug this JavaScript code",
		}

		for _, q := range codeQuestions {
			result, err := classifier.Classify(context.Background(), q)
			Expect(err).NotTo(HaveOccurred())
			Expect(result).NotTo(BeNil())
			// Just verify it returns valid result
			Expect(result.Label).To(BeElementOf(
				classification.FactCheckLabelNeeded,
				classification.FactCheckLabelNotNeeded,
			))
		}
	})
})

var _ = Describe("HallucinationDetector Integration", Label("model-artifacts"), func() {
	// NOTE: These tests require the hallucination detection model to be available
	// Skip if model is not found at HALLUCINATION_MODEL_PATH env var

	var (
		detector *classification.HallucinationDetector
		cfg      *config.HallucinationModelConfig
	)

	getModelPath := func() string {
		if path := os.Getenv("HALLUCINATION_MODEL_PATH"); path != "" {
			return path
		}
		// Try relative path from test directory (extproc -> models)
		relativePath := resolveExtprocTestPath("../../../../../models/mom-halugate-detector")
		if _, err := os.Stat(relativePath); err == nil {
			return relativePath
		}
		// Try from project root
		if root := findProjectRoot(); root != "" {
			projectPath := filepath.Join(root, "models", "mom-halugate-detector")
			if _, err := os.Stat(projectPath); err == nil {
				return projectPath
			}
		}
		return relativePath
	}

	BeforeEach(func() {
		modelPath := getModelPath()
		skipExtprocSpecIfModelArtifactsMissing("Hallucination model", modelPath)

		cfg = &config.HallucinationModelConfig{
			ModelID:   modelPath,
			Threshold: 0.5,
			UseCPU:    true,
		}
		var err error
		detector, err = classification.NewHallucinationDetector(cfg)
		Expect(err).NotTo(HaveOccurred())
		err = detector.Initialize()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should detect grounded answers", func() {
		contextText := "The Eiffel Tower is located in Paris, France. It was built in 1889."
		question := "Where is the Eiffel Tower?"
		answer := "The Eiffel Tower is located in Paris, France."

		result, err := detector.Detect(context.Background(), contextText, question, answer)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeFalse())
	})

	It("should require context", func() {
		_, err := detector.Detect(context.Background(), "", "question", "some answer")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("context is required"))
	})

	It("should handle empty answer", func() {
		result, err := detector.Detect(context.Background(), "context", "question", "")
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeFalse())
	})

	It("should detect hallucinated answers", func() {
		contextText := "The Eiffel Tower was constructed from 1887 to 1889. It is 330 metres tall."
		question := "When was the Eiffel Tower built?"
		// HALLUCINATED: wrong year and wrong height
		answer := "The Eiffel Tower was built in 1950 and is 500 meters tall."

		result, err := detector.Detect(context.Background(), contextText, question, answer)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeTrue())
	})
})
