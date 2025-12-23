package extproc

import (
	"encoding/json"
	"os"
	"path/filepath"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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

// createMockBodyResponse creates a mock ProcessingResponse for testing header mutations
func createMockBodyResponse() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseBody{
			ResponseBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{},
			},
		},
	}
}

// TestHallucinationExtproc is removed - tests are now part of the main ExtProc Suite in extproc_test.go
// This avoids the "Rerunning Suite" error from Ginkgo when multiple RunSpecs are called

var _ = Describe("RequestContext Hallucination Fields", func() {
	var ctx *RequestContext

	BeforeEach(func() {
		ctx = &RequestContext{
			Headers: make(map[string]string),
		}
	})

	It("should initialize with hallucination fields as zero values", func() {
		Expect(ctx.FactCheckNeeded).To(BeFalse())
		Expect(ctx.FactCheckConfidence).To(Equal(float32(0)))
		Expect(ctx.HasToolsForFactCheck).To(BeFalse())
		Expect(ctx.ToolResultsContext).To(BeEmpty())
		Expect(ctx.UserContent).To(BeEmpty())
		Expect(ctx.HallucinationDetected).To(BeFalse())
		Expect(ctx.HallucinationSpans).To(BeNil())
	})

	It("should store fact-check results", func() {
		ctx.FactCheckNeeded = true
		ctx.FactCheckConfidence = 0.85
		ctx.UserContent = "What is the capital of France?"

		Expect(ctx.FactCheckNeeded).To(BeTrue())
		Expect(ctx.FactCheckConfidence).To(Equal(float32(0.85)))
		Expect(ctx.UserContent).To(Equal("What is the capital of France?"))
	})

	It("should store tool context", func() {
		ctx.HasToolsForFactCheck = true
		ctx.ToolResultsContext = "Paris is the capital of France. It has a population of 2.1 million."

		Expect(ctx.HasToolsForFactCheck).To(BeTrue())
		Expect(ctx.ToolResultsContext).To(ContainSubstring("Paris"))
	})

	It("should store hallucination detection results", func() {
		ctx.HallucinationDetected = true
		ctx.HallucinationSpans = []string{"claim 1", "claim 2"}

		Expect(ctx.HallucinationDetected).To(BeTrue())
		Expect(ctx.HallucinationSpans).To(HaveLen(2))
		Expect(ctx.HallucinationSpans).To(ContainElements("claim 1", "claim 2"))
	})

	It("should store unverified factual response flag", func() {
		ctx.FactCheckNeeded = true
		ctx.HasToolsForFactCheck = false
		ctx.UnverifiedFactualResponse = true

		Expect(ctx.UnverifiedFactualResponse).To(BeTrue())
		Expect(ctx.FactCheckNeeded).To(BeTrue())
		Expect(ctx.HasToolsForFactCheck).To(BeFalse())
	})
})

var _ = Describe("extractToolResultsFromMessages", func() {
	It("should extract content from tool role messages", func() {
		messages := []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "What is the weather?",
			},
			map[string]interface{}{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []interface{}{
					map[string]interface{}{
						"id":   "call_1",
						"type": "function",
					},
				},
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_1",
				"content":      "The weather is sunny with 25°C",
			},
			map[string]interface{}{
				"role":         "tool",
				"tool_call_id": "call_2",
				"content":      "Humidity is 65%",
			},
		}

		results := extractToolResultsFromMessages(messages)

		Expect(results).To(HaveLen(2))
		Expect(results[0]).To(Equal("The weather is sunny with 25°C"))
		Expect(results[1]).To(Equal("Humidity is 65%"))
	})

	It("should return empty slice when no tool messages", func() {
		messages := []interface{}{
			map[string]interface{}{
				"role":    "user",
				"content": "Hello",
			},
			map[string]interface{}{
				"role":    "assistant",
				"content": "Hi there!",
			},
		}

		results := extractToolResultsFromMessages(messages)
		Expect(results).To(BeEmpty())
	})

	It("should skip tool messages with empty content", func() {
		messages := []interface{}{
			map[string]interface{}{
				"role":    "tool",
				"content": "",
			},
			map[string]interface{}{
				"role":    "tool",
				"content": "Valid content",
			},
		}

		results := extractToolResultsFromMessages(messages)
		Expect(results).To(HaveLen(1))
		Expect(results[0]).To(Equal("Valid content"))
	})

	It("should handle malformed messages gracefully", func() {
		messages := []interface{}{
			"not a map",
			nil,
			map[string]interface{}{
				"role": 123, // wrong type
			},
			map[string]interface{}{
				"role":    "tool",
				"content": "Valid",
			},
		}

		results := extractToolResultsFromMessages(messages)
		Expect(results).To(HaveLen(1))
	})
})

var _ = Describe("extractAssistantContentFromResponse", func() {
	It("should extract content from valid response", func() {
		response := map[string]interface{}{
			"id": "chatcmpl-123",
			"choices": []interface{}{
				map[string]interface{}{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "The capital of France is Paris.",
					},
				},
			},
		}

		responseBytes, _ := json.Marshal(response)
		content := extractAssistantContentFromResponse(responseBytes)

		Expect(content).To(Equal("The capital of France is Paris."))
	})

	It("should return empty string for empty choices", func() {
		response := map[string]interface{}{
			"id":      "chatcmpl-123",
			"choices": []interface{}{},
		}

		responseBytes, _ := json.Marshal(response)
		content := extractAssistantContentFromResponse(responseBytes)

		Expect(content).To(BeEmpty())
	})

	It("should return empty string for invalid JSON", func() {
		content := extractAssistantContentFromResponse([]byte("invalid json"))
		Expect(content).To(BeEmpty())
	})

	It("should return empty string for nil response", func() {
		content := extractAssistantContentFromResponse(nil)
		Expect(content).To(BeEmpty())
	})
})

var _ = Describe("OpenAIRouter Hallucination Methods", func() {
	var (
		router *OpenAIRouter
		cfg    *config.RouterConfig
	)

	BeforeEach(func() {
		cfg = &config.RouterConfig{}
		router = &OpenAIRouter{
			Config: cfg,
		}
	})

	Describe("shouldPerformHallucinationDetection", func() {
		// Helper to create a decision with hallucination plugin
		createDecisionWithHallucination := func(enabled bool, action string) *config.Decision {
			return &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled":              enabled,
							"hallucination_action": action,
						},
					},
				},
			}
		}

		It("should return false when classifier is nil", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				HasToolsForFactCheck: true,
				ToolResultsContext:   "some context",
				VSRSelectedDecision:  createDecisionWithHallucination(true, "warn"),
			}

			Expect(router.shouldPerformHallucinationDetection(ctx)).To(BeFalse())
		})

		It("should return false when hallucination plugin not enabled for decision", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				HasToolsForFactCheck: true,
				ToolResultsContext:   "some context",
				VSRSelectedDecision:  createDecisionWithHallucination(false, "warn"),
			}

			Expect(router.shouldPerformHallucinationDetection(ctx)).To(BeFalse())
		})

		It("should return false when decision is nil", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				HasToolsForFactCheck: true,
				ToolResultsContext:   "some context",
				VSRSelectedDecision:  nil,
			}

			Expect(router.shouldPerformHallucinationDetection(ctx)).To(BeFalse())
		})

		It("should return false when fact-check not needed", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      false,
				HasToolsForFactCheck: true,
				ToolResultsContext:   "some context",
				VSRSelectedDecision:  createDecisionWithHallucination(true, "warn"),
			}

			Expect(router.shouldPerformHallucinationDetection(ctx)).To(BeFalse())
		})

		It("should return false when no tools available", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				HasToolsForFactCheck: false,
				ToolResultsContext:   "",
				VSRSelectedDecision:  createDecisionWithHallucination(true, "warn"),
			}

			Expect(router.shouldPerformHallucinationDetection(ctx)).To(BeFalse())
		})
	})

	Describe("isHallucinationEnabledForDecision", func() {
		It("should return false when decision is nil", func() {
			Expect(router.isHallucinationEnabledForDecision(nil)).To(BeFalse())
		})

		It("should return false when no hallucination plugin configured", func() {
			decision := &config.Decision{
				Name:    "test_decision",
				Plugins: []config.DecisionPlugin{},
			}
			Expect(router.isHallucinationEnabledForDecision(decision)).To(BeFalse())
		})

		It("should return false when hallucination plugin disabled", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled": false,
						},
					},
				},
			}
			Expect(router.isHallucinationEnabledForDecision(decision)).To(BeFalse())
		})

		It("should return true when hallucination plugin enabled", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled": true,
						},
					},
				},
			}
			Expect(router.isHallucinationEnabledForDecision(decision)).To(BeTrue())
		})
	})

	Describe("getHallucinationActionForDecision", func() {
		It("should return 'header' when decision is nil", func() {
			Expect(router.getHallucinationActionForDecision(nil)).To(Equal("header"))
		})

		It("should return 'header' when no hallucination plugin configured", func() {
			decision := &config.Decision{
				Name:    "test_decision",
				Plugins: []config.DecisionPlugin{},
			}
			Expect(router.getHallucinationActionForDecision(decision)).To(Equal("header"))
		})

		It("should return 'header' when action not specified", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled": true,
						},
					},
				},
			}
			Expect(router.getHallucinationActionForDecision(decision)).To(Equal("header"))
		})

		It("should return 'header' when action is header", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled":              true,
							"hallucination_action": "header",
						},
					},
				},
			}
			Expect(router.getHallucinationActionForDecision(decision)).To(Equal("header"))
		})

		It("should return 'body' when action is body", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled":              true,
							"hallucination_action": "body",
						},
					},
				},
			}
			Expect(router.getHallucinationActionForDecision(decision)).To(Equal("body"))
		})

		It("should return 'none' when action is none", func() {
			decision := &config.Decision{
				Name: "test_decision",
				Plugins: []config.DecisionPlugin{
					{
						Type: "hallucination",
						Configuration: map[string]interface{}{
							"enabled":              true,
							"hallucination_action": "none",
						},
					},
				},
			}
			Expect(router.getHallucinationActionForDecision(decision)).To(Equal("none"))
		})
	})

	Describe("checkRequestHasTools", func() {
		It("should detect tools in request", func() {
			requestWithTools := map[string]interface{}{
				"model": "gpt-4",
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "What is the weather?",
					},
				},
				"tools": []interface{}{
					map[string]interface{}{
						"type": "function",
						"function": map[string]interface{}{
							"name": "get_weather",
						},
					},
				},
			}

			body, _ := json.Marshal(requestWithTools)
			ctx := &RequestContext{
				OriginalRequestBody: body,
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeTrue())
		})

		It("should extract tool results from messages", func() {
			requestWithToolResults := map[string]interface{}{
				"model": "gpt-4",
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "What is the weather?",
					},
					map[string]interface{}{
						"role":    "tool",
						"content": "The weather is sunny, 25°C in Paris",
					},
				},
			}

			body, _ := json.Marshal(requestWithToolResults)
			ctx := &RequestContext{
				OriginalRequestBody: body,
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeTrue())
			Expect(ctx.ToolResultsContext).To(ContainSubstring("sunny"))
			Expect(ctx.ToolResultsContext).To(ContainSubstring("Paris"))
		})

		It("should handle request without tools", func() {
			requestWithoutTools := map[string]interface{}{
				"model": "gpt-4",
				"messages": []interface{}{
					map[string]interface{}{
						"role":    "user",
						"content": "Hello",
					},
				},
			}

			body, _ := json.Marshal(requestWithoutTools)
			ctx := &RequestContext{
				OriginalRequestBody: body,
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeFalse())
			Expect(ctx.ToolResultsContext).To(BeEmpty())
		})

		It("should handle nil/empty body", func() {
			ctx := &RequestContext{
				OriginalRequestBody: nil,
			}

			router.checkRequestHasTools(ctx)

			Expect(ctx.HasToolsForFactCheck).To(BeFalse())
		})
	})

	Describe("checkUnverifiedFactualResponse", func() {
		It("should flag unverified when fact-check needed but no tools", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				FactCheckConfidence:  0.85,
				HasToolsForFactCheck: false,
			}

			router.checkUnverifiedFactualResponse(ctx)

			Expect(ctx.UnverifiedFactualResponse).To(BeTrue())
		})

		It("should not flag when fact-check not needed", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      false,
				HasToolsForFactCheck: false,
			}

			router.checkUnverifiedFactualResponse(ctx)

			Expect(ctx.UnverifiedFactualResponse).To(BeFalse())
		})

		It("should not flag when tools are available", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      true,
				HasToolsForFactCheck: true,
				ToolResultsContext:   "some context",
			}

			router.checkUnverifiedFactualResponse(ctx)

			Expect(ctx.UnverifiedFactualResponse).To(BeFalse())
		})

		It("should not flag when both conditions are false", func() {
			ctx := &RequestContext{
				FactCheckNeeded:      false,
				HasToolsForFactCheck: true,
			}

			router.checkUnverifiedFactualResponse(ctx)

			Expect(ctx.UnverifiedFactualResponse).To(BeFalse())
		})
	})

	Describe("addUnverifiedFactualWarningHeaders", func() {
		It("should not modify response when not unverified", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: false,
			}

			// Create a simple response
			response := createMockBodyResponse()
			result := router.addUnverifiedFactualWarningHeaders(response, ctx)

			// Should return same response unchanged
			Expect(result).To(Equal(response))
		})

		It("should add headers when unverified factual response", func() {
			ctx := &RequestContext{
				UnverifiedFactualResponse: true,
				FactCheckNeeded:           true,
			}

			response := createMockBodyResponse()
			result := router.addUnverifiedFactualWarningHeaders(response, ctx)

			// Should have header mutations
			bodyResp, ok := result.Response.(*ext_proc.ProcessingResponse_ResponseBody)
			Expect(ok).To(BeTrue())
			Expect(bodyResp.ResponseBody.Response).NotTo(BeNil())
			Expect(bodyResp.ResponseBody.Response.HeaderMutation).NotTo(BeNil())
			Expect(bodyResp.ResponseBody.Response.HeaderMutation.SetHeaders).To(HaveLen(3))

			// Check header values
			headerMap := make(map[string]string)
			for _, h := range bodyResp.ResponseBody.Response.HeaderMutation.SetHeaders {
				headerMap[h.Header.Key] = string(h.Header.RawValue)
			}

			Expect(headerMap).To(HaveKeyWithValue("x-vsr-unverified-factual-response", "true"))
			Expect(headerMap).To(HaveKeyWithValue("x-vsr-fact-check-needed", "true"))
			Expect(headerMap).To(HaveKeyWithValue("x-vsr-verification-context-missing", "true"))
		})
	})
})

var _ = Describe("FactCheckClassifier Integration", func() {
	var (
		classifier *classification.FactCheckClassifier
		cfg        *config.FactCheckModelConfig
	)

	BeforeEach(func() {
		cfg = &config.FactCheckModelConfig{
			ModelID:   "../../../../models/mom-halugate-sentinel",
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
			result, err := classifier.Classify(q)
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
			result, err := classifier.Classify(q)
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

var _ = Describe("HallucinationDetector Integration", func() {
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
		relativePath := "../../../../../models/mom-halugate-detector"
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
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			Skip("Skipping: Hallucination model not found at " + modelPath)
		}

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
		context := "The Eiffel Tower is located in Paris, France. It was built in 1889."
		question := "Where is the Eiffel Tower?"
		answer := "The Eiffel Tower is located in Paris, France."

		result, err := detector.Detect(context, question, answer)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeFalse())
	})

	It("should require context", func() {
		_, err := detector.Detect("", "question", "some answer")
		Expect(err).To(HaveOccurred())
		Expect(err.Error()).To(ContainSubstring("context is required"))
	})

	It("should handle empty answer", func() {
		result, err := detector.Detect("context", "question", "")
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeFalse())
	})

	It("should detect hallucinated answers", func() {
		context := "The Eiffel Tower was constructed from 1887 to 1889. It is 330 metres tall."
		question := "When was the Eiffel Tower built?"
		// HALLUCINATED: wrong year and wrong height
		answer := "The Eiffel Tower was built in 1950 and is 500 meters tall."

		result, err := detector.Detect(context, question, answer)
		Expect(err).NotTo(HaveOccurred())
		Expect(result).NotTo(BeNil())
		Expect(result.HallucinationDetected).To(BeTrue())
	})
})
