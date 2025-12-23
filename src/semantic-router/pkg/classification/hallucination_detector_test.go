package classification

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// OpenAI API types for test simulation
type OpenAIChatCompletionRequest struct {
	Model    string              `json:"model"`
	Messages []OpenAIChatMessage `json:"messages"`
	Tools    []OpenAITool        `json:"tools,omitempty"`
}

type OpenAIChatMessage struct {
	Role       string `json:"role"`
	Content    string `json:"content,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`
}

type OpenAITool struct {
	Type     string         `json:"type"`
	Function OpenAIFunction `json:"function"`
}

type OpenAIFunction struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}

type OpenAIChatCompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []OpenAIChatChoice `json:"choices"`
}

type OpenAIChatChoice struct {
	Index        int               `json:"index"`
	Message      OpenAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

// findProjectRootFromTest finds the project root by looking for go.mod
func findProjectRootFromTest() string {
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

// getHallucinationModelPath returns the model path from env or a default
func getHallucinationModelPath() string {
	if path := os.Getenv("HALLUCINATION_MODEL_PATH"); path != "" {
		return path
	}
	// Try relative path from test directory
	relativePath := "../../../../models/mom-halugate-detector"
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	// Try from project root
	if root := findProjectRootFromTest(); root != "" {
		projectPath := filepath.Join(root, "models", "mom-halugate-detector")
		if _, err := os.Stat(projectPath); err == nil {
			return projectPath
		}
	}
	return relativePath
}

// skipIfNoModel skips the test if the hallucination detection model is not available
func skipIfNoModel(t *testing.T) {
	modelPath := getHallucinationModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Skipping test: Hallucination model not found at %s. Set HALLUCINATION_MODEL_PATH env var.", modelPath)
	}
}

// TestHallucinationDetector_RequiresConfig tests that config is required
func TestHallucinationDetector_RequiresConfig(t *testing.T) {
	// Test nil config
	detector, err := NewHallucinationDetector(nil)
	if err == nil {
		t.Error("Expected error for nil config")
	}
	if detector != nil {
		t.Error("Expected nil detector for nil config")
	}

	// Test empty model_id
	cfg := &config.HallucinationModelConfig{
		ModelID:   "",
		Threshold: 0.5,
	}
	detector, err = NewHallucinationDetector(cfg)
	if err == nil {
		t.Error("Expected error for empty model_id")
	}
	if detector != nil {
		t.Error("Expected nil detector for empty model_id")
	}
}

// TestHallucinationDetector_InitializationRequired tests that Initialize must be called
func TestHallucinationDetector_InitializationRequired(t *testing.T) {
	skipIfNoModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	// Should fail if not initialized
	_, err = detector.Detect("context", "question", "answer")
	if err == nil {
		t.Error("Expected error when detecting without initialization")
	}
}

// TestHallucinationDetector_ContextRequired tests that context is required
func TestHallucinationDetector_ContextRequired(t *testing.T) {
	skipIfNoModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	// Should fail with empty context
	_, err = detector.Detect("", "What is X?", "X is Y")
	if err == nil {
		t.Error("Expected error for empty context")
	}
}

// TestHallucinationDetector_EmptyAnswerOK tests that empty answer returns no hallucination
func TestHallucinationDetector_EmptyAnswerOK(t *testing.T) {
	skipIfNoModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}

	result, err := detector.Detect("Some context", "Question?", "")
	if err != nil {
		t.Errorf("Unexpected error for empty answer: %v", err)
	}
	if result.HallucinationDetected {
		t.Error("Empty answer should not be detected as hallucination")
	}
	if result.Confidence != 1.0 {
		t.Errorf("Expected confidence 1.0 for empty answer, got %f", result.Confidence)
	}
}

// TestHallucinationDetector_OpenAIPipeline_GroundedResponse tests the full pipeline
// with a grounded (non-hallucinated) response
func TestHallucinationDetector_OpenAIPipeline_GroundedResponse(t *testing.T) {
	skipIfNoModel(t)

	// 1. Simulate OpenAI request with tool results
	request := OpenAIChatCompletionRequest{
		Model: "gpt-4",
		Messages: []OpenAIChatMessage{
			{Role: "user", Content: "What is the Eiffel Tower's height?"},
			{Role: "assistant", Content: ""},
			{Role: "tool", ToolCallID: "call_1", Content: "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It is 330 metres (1,083 ft) tall and was constructed from 1887 to 1889."},
		},
	}

	// 2. Simulate OpenAI response
	response := OpenAIChatCompletionResponse{
		ID:      "chatcmpl-test-1",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []OpenAIChatChoice{
			{
				Index: 0,
				Message: OpenAIChatMessage{
					Role:    "assistant",
					Content: "The Eiffel Tower is 330 metres tall. It was built between 1887 and 1889.",
				},
				FinishReason: "stop",
			},
		},
	}

	// 3. Extract context from tool messages
	var toolContext string
	for _, msg := range request.Messages {
		if msg.Role == "tool" {
			toolContext += msg.Content + "\n"
		}
	}

	// 4. Extract user question
	var userQuestion string
	for _, msg := range request.Messages {
		if msg.Role == "user" {
			userQuestion = msg.Content
			break
		}
	}

	// 5. Extract assistant answer
	assistantAnswer := response.Choices[0].Message.Content

	// 6. Initialize detector
	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	// 7. Detect hallucination
	result, err := detector.Detect(toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}

	// 8. Verify result - should NOT be hallucination (grounded in context)
	t.Logf("Grounded response test:")
	t.Logf("  Context: %s", toolContext[:min(100, len(toolContext))]+"...")
	t.Logf("  Question: %s", userQuestion)
	t.Logf("  Answer: %s", assistantAnswer)
	t.Logf("  Hallucination detected: %v", result.HallucinationDetected)
	t.Logf("  Confidence: %.3f", result.Confidence)
	t.Logf("  Unsupported spans: %v", result.UnsupportedSpans)

	if result.HallucinationDetected {
		t.Errorf("Expected no hallucination for grounded response, got hallucination with spans: %v", result.UnsupportedSpans)
	}
}

// TestHallucinationDetector_OpenAIPipeline_HallucinatedResponse tests the full pipeline
// with a hallucinated response
func TestHallucinationDetector_OpenAIPipeline_HallucinatedResponse(t *testing.T) {
	skipIfNoModel(t)

	// 1. Simulate OpenAI request with tool results
	request := OpenAIChatCompletionRequest{
		Model: "gpt-4",
		Messages: []OpenAIChatMessage{
			{Role: "user", Content: "When was the Eiffel Tower built?"},
			{Role: "assistant", Content: ""},
			{Role: "tool", ToolCallID: "call_1", Content: "The Eiffel Tower was constructed from 1887 to 1889. It is located in Paris, France and is 330 metres tall."},
		},
	}

	// 2. Simulate OpenAI response with HALLUCINATED content
	response := OpenAIChatCompletionResponse{
		ID:      "chatcmpl-test-2",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []OpenAIChatChoice{
			{
				Index: 0,
				Message: OpenAIChatMessage{
					Role: "assistant",
					// HALLUCINATED: Wrong year (1950 instead of 1887-1889) and wrong height (500m instead of 330m)
					Content: "The Eiffel Tower was built in 1950 and stands at 500 meters tall.",
				},
				FinishReason: "stop",
			},
		},
	}

	// 3. Extract context from tool messages
	var toolContext string
	for _, msg := range request.Messages {
		if msg.Role == "tool" {
			toolContext += msg.Content + "\n"
		}
	}

	// 4. Extract user question
	var userQuestion string
	for _, msg := range request.Messages {
		if msg.Role == "user" {
			userQuestion = msg.Content
			break
		}
	}

	// 5. Extract assistant answer
	assistantAnswer := response.Choices[0].Message.Content

	// 6. Initialize detector
	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	// 7. Detect hallucination
	result, err := detector.Detect(toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}

	// 8. Verify result - SHOULD be hallucination
	t.Logf("Hallucinated response test:")
	t.Logf("  Context: %s", toolContext[:min(100, len(toolContext))]+"...")
	t.Logf("  Question: %s", userQuestion)
	t.Logf("  Answer: %s", assistantAnswer)
	t.Logf("  Hallucination detected: %v", result.HallucinationDetected)
	t.Logf("  Confidence: %.3f", result.Confidence)
	t.Logf("  Unsupported spans: %v", result.UnsupportedSpans)

	if !result.HallucinationDetected {
		t.Errorf("Expected hallucination for fabricated response, but none detected")
	}
}

// TestHallucinationDetector_OpenAIPipeline_MultipleToolResults tests with multiple tool results
func TestHallucinationDetector_OpenAIPipeline_MultipleToolResults(t *testing.T) {
	skipIfNoModel(t)

	// 1. Simulate OpenAI request with multiple tool results
	request := OpenAIChatCompletionRequest{
		Model: "gpt-4",
		Messages: []OpenAIChatMessage{
			{Role: "user", Content: "Compare the heights of the Eiffel Tower and Statue of Liberty"},
			{Role: "assistant", Content: ""},
			{Role: "tool", ToolCallID: "call_1", Content: "The Eiffel Tower is 330 metres (1,083 ft) tall."},
			{Role: "tool", ToolCallID: "call_2", Content: "The Statue of Liberty is 93 metres (305 ft) tall from ground to torch."},
		},
	}

	// 2. Simulate grounded response
	response := OpenAIChatCompletionResponse{
		ID:      "chatcmpl-test-3",
		Object:  "chat.completion",
		Created: 1234567890,
		Model:   "gpt-4",
		Choices: []OpenAIChatChoice{
			{
				Index: 0,
				Message: OpenAIChatMessage{
					Role:    "assistant",
					Content: "The Eiffel Tower at 330 metres is significantly taller than the Statue of Liberty which stands at 93 metres.",
				},
				FinishReason: "stop",
			},
		},
	}

	// 3. Extract combined context from tool messages
	var toolContext string
	for _, msg := range request.Messages {
		if msg.Role == "tool" {
			toolContext += msg.Content + "\n"
		}
	}

	// 4. Extract user question
	var userQuestion string
	for _, msg := range request.Messages {
		if msg.Role == "user" {
			userQuestion = msg.Content
			break
		}
	}

	// 5. Extract assistant answer
	assistantAnswer := response.Choices[0].Message.Content

	// 6. Initialize detector
	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize detector: %v", err)
	}

	// 7. Detect hallucination
	result, err := detector.Detect(toolContext, userQuestion, assistantAnswer)
	if err != nil {
		t.Fatalf("Detection failed: %v", err)
	}

	// 8. Verify result
	t.Logf("Multiple tool results test:")
	t.Logf("  Context: %s", toolContext[:min(100, len(toolContext))]+"...")
	t.Logf("  Question: %s", userQuestion)
	t.Logf("  Answer: %s", assistantAnswer)
	t.Logf("  Hallucination detected: %v", result.HallucinationDetected)
	t.Logf("  Confidence: %.3f", result.Confidence)

	if result.HallucinationDetected {
		t.Errorf("Expected no hallucination for grounded comparison, got: %v", result.UnsupportedSpans)
	}
}

// TestHallucinationDetector_JSONSerialization tests that results can be serialized
func TestHallucinationDetector_JSONSerialization(t *testing.T) {
	result := &HallucinationResult{
		HallucinationDetected: true,
		Confidence:            0.85,
		UnsupportedSpans:      []string{"claim 1", "claim 2"},
		SupportedSpans:        []string{"verified claim"},
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal result: %v", err)
	}

	var decoded HallucinationResult
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal result: %v", err)
	}

	if decoded.HallucinationDetected != result.HallucinationDetected {
		t.Error("HallucinationDetected mismatch after serialization")
	}
	if decoded.Confidence != result.Confidence {
		t.Error("Confidence mismatch after serialization")
	}
	if len(decoded.UnsupportedSpans) != len(result.UnsupportedSpans) {
		t.Error("UnsupportedSpans length mismatch after serialization")
	}
}

// helper function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// getNLIModelPath returns the NLI model path from env or a default
func getNLIModelPath() string {
	if path := os.Getenv("NLI_MODEL_PATH"); path != "" {
		return path
	}
	// Try relative path from test directory
	relativePath := "../../../../models/mom-halugate-explainer"
	if _, err := os.Stat(relativePath); err == nil {
		return relativePath
	}
	// Try from project root
	if root := findProjectRootFromTest(); root != "" {
		projectPath := filepath.Join(root, "models", "mom-halugate-explainer")
		if _, err := os.Stat(projectPath); err == nil {
			return projectPath
		}
	}
	return relativePath
}

// skipIfNoNLIModel skips the test if the NLI model is not available
func skipIfNoNLIModel(t *testing.T) {
	modelPath := getNLIModelPath()
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		t.Skipf("Skipping NLI test: Model not found at %s. Set NLI_MODEL_PATH env var.", modelPath)
	}
}

// TestNLILabel_String tests NLI label string representation
func TestNLILabel_String(t *testing.T) {
	tests := []struct {
		label    NLILabel
		expected string
	}{
		{NLIEntailment, "ENTAILMENT"},
		{NLINeutral, "NEUTRAL"},
		{NLIContradiction, "CONTRADICTION"},
		{NLIError, "ERROR"},
	}

	for _, tc := range tests {
		if tc.label.String() != tc.expected {
			t.Errorf("NLILabel(%d).String() = %s, want %s", tc.label, tc.label.String(), tc.expected)
		}
	}
}

// TestEnhancedHallucinationResult_JSONSerialization tests enhanced result serialization
func TestEnhancedHallucinationResult_JSONSerialization(t *testing.T) {
	result := &EnhancedHallucinationResult{
		HallucinationDetected: true,
		Confidence:            0.92,
		Spans: []EnhancedHallucinationSpan{
			{
				Text:                    "built in 1950",
				Start:                   10,
				End:                     23,
				HallucinationConfidence: 0.89,
				NLILabel:                NLIContradiction,
				NLILabelStr:             "CONTRADICTION",
				NLIConfidence:           0.94,
				Severity:                4,
				Explanation:             "CONTRADICTION: This claim directly conflicts with the provided context",
			},
		},
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("Failed to marshal enhanced result: %v", err)
	}

	var decoded EnhancedHallucinationResult
	err = json.Unmarshal(data, &decoded)
	if err != nil {
		t.Fatalf("Failed to unmarshal enhanced result: %v", err)
	}

	if decoded.HallucinationDetected != result.HallucinationDetected {
		t.Error("HallucinationDetected mismatch after serialization")
	}
	if len(decoded.Spans) != len(result.Spans) {
		t.Errorf("Spans length mismatch: got %d, want %d", len(decoded.Spans), len(result.Spans))
	}
	if len(decoded.Spans) > 0 {
		if decoded.Spans[0].NLILabelStr != "CONTRADICTION" {
			t.Errorf("NLILabelStr mismatch: got %s, want CONTRADICTION", decoded.Spans[0].NLILabelStr)
		}
		if decoded.Spans[0].Severity != 4 {
			t.Errorf("Severity mismatch: got %d, want 4", decoded.Spans[0].Severity)
		}
	}
}

// TestHallucinationDetector_SetNLIConfig tests setting NLI model config
func TestHallucinationDetector_SetNLIConfig(t *testing.T) {
	skipIfNoModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	// Initially NLI should not be initialized
	if detector.IsNLIInitialized() {
		t.Error("NLI should not be initialized before setting config")
	}

	// Set NLI model config
	nliCfg := &config.NLIModelConfig{
		ModelID:   "/path/to/nli/model",
		Threshold: 0.7,
		UseCPU:    true,
	}
	detector.SetNLIConfig(nliCfg)

	// NLI should still not be initialized (just config set)
	if detector.IsNLIInitialized() {
		t.Error("NLI should not be initialized just from setting config")
	}
}

// TestHallucinationDetector_NLIClassification tests NLI classification
// Requires both hallucination detection and NLI models to be available
func TestHallucinationDetector_NLIClassification(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	// Initialize hallucination detector
	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize hallucination detector: %v", err)
	}

	// Set and initialize NLI
	nliCfg := &config.NLIModelConfig{
		ModelID:   getNLIModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	}
	detector.SetNLIConfig(nliCfg)
	err = detector.InitializeNLI()
	if err != nil {
		t.Fatalf("Failed to initialize NLI: %v", err)
	}

	// Test entailment: premise supports hypothesis
	result, err := detector.ClassifyNLI(
		"The Eiffel Tower is located in Paris, France.",
		"The Eiffel Tower is in France.",
	)
	if err != nil {
		t.Fatalf("NLI classification failed: %v", err)
	}
	t.Logf("Entailment test: label=%s, confidence=%.3f", result.LabelStr, result.Confidence)
	if result.Label != NLIEntailment {
		t.Errorf("Expected ENTAILMENT, got %s", result.LabelStr)
	}

	// Test contradiction: premise contradicts hypothesis
	result, err = detector.ClassifyNLI(
		"The Eiffel Tower was built between 1887 and 1889.",
		"The Eiffel Tower was built in 1950.",
	)
	if err != nil {
		t.Fatalf("NLI classification failed: %v", err)
	}
	t.Logf("Contradiction test: label=%s, confidence=%.3f", result.LabelStr, result.Confidence)
	if result.Label != NLIContradiction {
		t.Errorf("Expected CONTRADICTION, got %s", result.LabelStr)
	}

	// Test neutral: premise doesn't address hypothesis
	result, err = detector.ClassifyNLI(
		"The Eiffel Tower is 330 meters tall.",
		"The Eiffel Tower is very popular with tourists.",
	)
	if err != nil {
		t.Fatalf("NLI classification failed: %v", err)
	}
	t.Logf("Neutral test: label=%s, confidence=%.3f", result.LabelStr, result.Confidence)
	// Neutral cases can be tricky, so we just log the result
}

// TestHallucinationDetector_EnhancedDetection tests enhanced detection with NLI
func TestHallucinationDetector_EnhancedDetection(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	// Initialize both models
	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize hallucination detector: %v", err)
	}

	nliCfg := &config.NLIModelConfig{
		ModelID:   getNLIModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	}
	detector.SetNLIConfig(nliCfg)
	err = detector.InitializeNLI()
	if err != nil {
		t.Fatalf("Failed to initialize NLI: %v", err)
	}

	// Test with hallucinated response
	context := "The Eiffel Tower was constructed from 1887 to 1889. It is located in Paris, France and is 330 metres tall."
	question := "When was the Eiffel Tower built?"
	answer := "The Eiffel Tower was built in 1950 and stands at 500 meters tall."

	result, err := detector.DetectWithNLI(context, question, answer)
	if err != nil {
		t.Fatalf("Enhanced detection failed: %v", err)
	}

	t.Logf("Enhanced detection result:")
	t.Logf("  Hallucination detected: %v", result.HallucinationDetected)
	t.Logf("  Confidence: %.3f", result.Confidence)
	t.Logf("  Number of spans: %d", len(result.Spans))

	for i, span := range result.Spans {
		t.Logf("  Span %d:", i)
		t.Logf("    Text: %s", span.Text)
		t.Logf("    Hallucination confidence: %.3f", span.HallucinationConfidence)
		t.Logf("    NLI label: %s (confidence: %.3f)", span.NLILabelStr, span.NLIConfidence)
		t.Logf("    Severity: %d", span.Severity)
		t.Logf("    Explanation: %s", span.Explanation)
	}

	if !result.HallucinationDetected {
		t.Error("Expected hallucination to be detected for fabricated response")
	}
}

// ================================================================================================
// DEMO: NLI FILTERING FALSE POSITIVES
// ================================================================================================

// TestHallucinationDetector_NLI_FiltersFalsePositives demonstrates how NLI can identify
// potential false positives from the hallucination detector by showing ENTAILMENT for flagged spans.
//
// Scenario: The hallucination detector might flag paraphrased content as "hallucination" because the
// exact words don't appear in context, but NLI can recognize semantic equivalence.
func TestHallucinationDetector_NLI_FiltersFalsePositives(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	detector := setupDetectorWithNLI(t)

	t.Log("=== DEMO: NLI Filtering False Positives ===")
	t.Log("")
	t.Log("Token-level hallucination detection can flag semantically")
	t.Log("correct paraphrases as 'hallucinations'. NLI provides a semantic check.")
	t.Log("")

	// Case 1: Paraphrased but semantically correct answer
	// The answer uses different words but means the same thing
	context := "The Great Wall of China is approximately 21,196 kilometers long. It was built over many centuries, with construction beginning in the 7th century BC."
	question := "How long is the Great Wall of China?"
	answer := "The Great Wall stretches for about 21,000 km." // Paraphrased, rounded number

	t.Log("Test Case 1: Paraphrased answer (semantically correct)")
	t.Logf("  Context: %s", context)
	t.Logf("  Question: %s", question)
	t.Logf("  Answer: %s", answer)
	t.Log("")

	// First, check what hallucination detector alone says
	ldResult, err := detector.Detect(context, question, answer)
	if err != nil {
		t.Fatalf("Hallucination detection failed: %v", err)
	}
	t.Logf("  Hallucination detector result:")
	t.Logf("    Hallucination detected: %v", ldResult.HallucinationDetected)
	t.Logf("    Confidence: %.3f", ldResult.Confidence)
	t.Logf("    Flagged spans: %v", ldResult.UnsupportedSpans)

	// Now check with NLI enhancement
	enhancedResult, err := detector.DetectWithNLI(context, question, answer)
	if err != nil {
		t.Fatalf("Enhanced detection failed: %v", err)
	}

	t.Logf("  Enhanced (Hallucination Detector + NLI) result:")
	t.Logf("    Hallucination detected: %v", enhancedResult.HallucinationDetected)
	t.Logf("    Confidence: %.3f", enhancedResult.Confidence)

	// Analyze each span - NLI ENTAILMENT suggests false positive
	for i, span := range enhancedResult.Spans {
		t.Logf("    Span %d: '%s'", i, span.Text)
		t.Logf("      Hallucination detector: %.1f%% confident it's hallucinated", span.HallucinationConfidence*100)
		t.Logf("      NLI verdict: %s (%.1f%% confident)", span.NLILabelStr, span.NLIConfidence*100)

		if span.NLILabel == NLIEntailment {
			t.Logf("      >>> NLI says ENTAILMENT - likely FALSE POSITIVE from hallucination detector!")
			t.Logf("      >>> The content is semantically supported by context despite word differences")
		}
	}
	t.Log("")

	// Case 2: Synonym usage
	context2 := "Mount Everest is the tallest mountain on Earth, standing at 8,849 meters above sea level."
	question2 := "What is the height of Mount Everest?"
	answer2 := "Mount Everest reaches an elevation of 8,849 meters." // "elevation" vs "standing at"

	t.Log("Test Case 2: Synonym usage (elevation vs standing at)")
	t.Logf("  Context: %s", context2)
	t.Logf("  Answer: %s", answer2)

	enhancedResult2, err := detector.DetectWithNLI(context2, question2, answer2)
	if err != nil {
		t.Fatalf("Enhanced detection failed: %v", err)
	}

	t.Logf("  Result: hallucination=%v, confidence=%.3f",
		enhancedResult2.HallucinationDetected, enhancedResult2.Confidence)

	for _, span := range enhancedResult2.Spans {
		if span.NLILabel == NLIEntailment {
			t.Logf("  NLI ENTAILMENT on '%s' - semantic match despite different wording", span.Text)
		}
	}
	t.Log("")
	t.Log("=== KEY INSIGHT ===")
	t.Log("When hallucination detector flags something but NLI says ENTAILMENT,")
	t.Log("consider it a potential false positive. The combined signal is more reliable.")
}

// ================================================================================================
// DEMO: NLI PROVIDES EXPLAINABILITY
// ================================================================================================

// TestHallucinationDetector_NLI_ProvidesExplainability demonstrates how NLI adds
// explainability to hallucination detection by categorizing the type of error.
func TestHallucinationDetector_NLI_ProvidesExplainability(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	detector := setupDetectorWithNLI(t)

	t.Log("=== DEMO: NLI Provides Explainability ===")
	t.Log("")
	t.Log("The hallucination detector tells you THAT something is wrong.")
	t.Log("NLI tells you WHY it's wrong (contradiction vs. unverifiable).")
	t.Log("")

	context := "Albert Einstein was born on March 14, 1879 in Ulm, Germany. He developed the theory of relativity and won the Nobel Prize in Physics in 1921."

	// Test different types of hallucinations
	testCases := []struct {
		name        string
		question    string
		answer      string
		expectedNLI string
		explanation string
	}{
		{
			name:        "Direct Contradiction",
			question:    "When was Einstein born?",
			answer:      "Einstein was born on July 4, 1900.",
			expectedNLI: "CONTRADICTION",
			explanation: "The answer directly contradicts the context (wrong date)",
		},
		{
			name:        "Fabricated Information",
			question:    "Tell me about Einstein's achievements",
			answer:      "Einstein invented the telephone and discovered America.",
			expectedNLI: "NEUTRAL/CONTRADICTION",
			explanation: "The answer contains fabricated claims not in context",
		},
		{
			name:        "Unverifiable Claim",
			question:    "What do you know about Einstein?",
			answer:      "Einstein was born in Germany and his favorite color was blue.",
			expectedNLI: "NEUTRAL",
			explanation: "The favorite color claim cannot be verified from context",
		},
	}

	for _, tc := range testCases {
		t.Logf("--- %s ---", tc.name)
		t.Logf("  Question: %s", tc.question)
		t.Logf("  Answer: %s", tc.answer)
		t.Logf("  Expected: %s (%s)", tc.expectedNLI, tc.explanation)
		t.Log("")

		result, err := detector.DetectWithNLI(context, tc.question, tc.answer)
		if err != nil {
			t.Logf("  Error: %v", err)
			continue
		}

		t.Logf("  Detection result: hallucination=%v", result.HallucinationDetected)

		for i, span := range result.Spans {
			t.Logf("  Span %d analysis:", i)
			t.Logf("    Text: '%s'", span.Text)
			t.Logf("    NLI Classification: %s", span.NLILabelStr)
			t.Logf("    Severity: %d/4", span.Severity)
			t.Logf("    Explanation: %s", span.Explanation)

			// Explain what each NLI label means for the user
			switch span.NLILabel {
			case NLIContradiction:
				t.Log("    >>> CONTRADICTION: This directly conflicts with known facts!")
				t.Log("    >>> Action: HIGH priority fix needed")
			case NLINeutral:
				t.Log("    >>> NEUTRAL: This cannot be verified from the context")
				t.Log("    >>> Action: May need additional sources to verify")
			case NLIEntailment:
				t.Log("    >>> ENTAILMENT: This is actually supported (possible false alarm)")
				t.Log("    >>> Action: Review if this is truly problematic")
			}
		}
		t.Log("")
	}

	t.Log("=== KEY INSIGHT ===")
	t.Log("NLI severity levels help prioritize which hallucinations to address:")
	t.Log("  - CONTRADICTION (severity 4): Factually wrong - must fix")
	t.Log("  - NEUTRAL (severity 2): Unverifiable - may need review")
	t.Log("  - ENTAILMENT (severity 1): Likely OK - low priority")
}

// ================================================================================================
// DEMO: COMBINED SIGNALS ARE MORE RELIABLE
// ================================================================================================

// TestHallucinationDetector_CombinedSignalsMoreReliable demonstrates that using both
// Hallucination detection and NLI together produces more reliable results than either alone.
func TestHallucinationDetector_CombinedSignalsMoreReliable(t *testing.T) {
	skipIfNoModel(t)
	skipIfNoNLIModel(t)

	detector := setupDetectorWithNLI(t)

	t.Log("=== DEMO: Combined Signals Are More Reliable ===")
	t.Log("")
	t.Log("Neither token-level hallucination detection nor NLI is perfect alone:")
	t.Log("  - Token-level detection: Can miss semantic equivalence (false positives)")
	t.Log("  - NLI: Sentence-level, may miss subtle token-level errors")
	t.Log("")
	t.Log("Together, they complement each other:")
	t.Log("  - Token-level detection catches: exact mismatches, fabricated numbers, wrong entities")
	t.Log("  - NLI catches: semantic contradictions, verifies paraphrases")
	t.Log("")

	context := "The Amazon River is approximately 6,400 kilometers long, making it the second longest river in the world after the Nile. It flows through South America."

	// Test cases showing how combined approach works better
	testCases := []struct {
		name                   string
		answer                 string
		hallucinationDetectExp string
		nliExp                 string
		combinedVerdict        string
	}{
		{
			name:                   "Clear Hallucination (both agree)",
			answer:                 "The Amazon River is 10,000 kilometers long and is the longest river in the world.",
			hallucinationDetectExp: "Should flag wrong numbers",
			nliExp:                 "Should say CONTRADICTION",
			combinedVerdict:        "HIGH CONFIDENCE hallucination - both systems agree",
		},
		{
			name:                   "Paraphrase (detector may flag, NLI should support)",
			answer:                 "The Amazon stretches about 6,400 km through South America.",
			hallucinationDetectExp: "May flag due to word differences",
			nliExp:                 "Should say ENTAILMENT",
			combinedVerdict:        "Likely FALSE POSITIVE - NLI confirms semantic match",
		},
		{
			name:                   "Subtle Error (NLI helps confirm)",
			answer:                 "The Amazon River is 6,400 kilometers long, making it the longest river.",
			hallucinationDetectExp: "Should flag 'longest' (context says 'second longest')",
			nliExp:                 "Should say CONTRADICTION",
			combinedVerdict:        "CONFIRMED hallucination - subtle but NLI catches the contradiction",
		},
		{
			name:                   "Added Information (unverifiable)",
			answer:                 "The Amazon River is 6,400 km long and home to pink dolphins.",
			hallucinationDetectExp: "May flag 'pink dolphins'",
			nliExp:                 "Should say NEUTRAL",
			combinedVerdict:        "UNVERIFIABLE - true but not in context, may be OK depending on use case",
		},
	}

	for _, tc := range testCases {
		t.Logf("--- %s ---", tc.name)
		t.Logf("  Answer: %s", tc.answer)
		t.Logf("  Expected hallucination detection: %s", tc.hallucinationDetectExp)
		t.Logf("  Expected NLI: %s", tc.nliExp)
		t.Logf("  Combined verdict: %s", tc.combinedVerdict)
		t.Log("")

		// Run hallucination detection alone
		ldResult, _ := detector.Detect(context, "Tell me about the Amazon River", tc.answer)
		t.Logf("  Hallucination detector alone: hallucination=%v, confidence=%.2f",
			ldResult.HallucinationDetected, ldResult.Confidence)

		// Run combined approach
		combined, _ := detector.DetectWithNLI(context, "Tell me about the Amazon River", tc.answer)
		t.Logf("  Combined result: hallucination=%v, confidence=%.2f",
			combined.HallucinationDetected, combined.Confidence)

		// Analyze the verdict
		if len(combined.Spans) > 0 {
			// Count NLI verdicts
			contradictions := 0
			entailments := 0
			neutrals := 0
			for _, span := range combined.Spans {
				switch span.NLILabel {
				case NLIContradiction:
					contradictions++
				case NLIEntailment:
					entailments++
				case NLINeutral:
					neutrals++
				}
			}

			t.Logf("  NLI breakdown: %d contradictions, %d neutrals, %d entailments",
				contradictions, neutrals, entailments)

			// Decision logic
			if contradictions > 0 {
				t.Log("  >>> DECISION: HIGH CONFIDENCE hallucination (NLI confirms contradiction)")
			} else if entailments > 0 && contradictions == 0 {
				t.Log("  >>> DECISION: Likely FALSE POSITIVE (NLI says content is supported)")
			} else if neutrals > 0 && contradictions == 0 {
				t.Log("  >>> DECISION: UNVERIFIABLE (content not in context, may be OK)")
			}
		} else if ldResult.HallucinationDetected {
			t.Log("  >>> NOTE: Hallucination detector flagged but no spans for NLI analysis")
		} else {
			t.Log("  >>> DECISION: CLEAN - no hallucination detected")
		}
		t.Log("")
	}

	t.Log("=== DECISION MATRIX ===")
	t.Log("")
	t.Log("| Hallucination | NLI           | Verdict                    |")
	t.Log("|---------------|---------------|----------------------------|")
	t.Log("| Hallucination | CONTRADICTION | HIGH CONFIDENCE - fix it   |")
	t.Log("| Hallucination | NEUTRAL       | UNVERIFIABLE - review      |")
	t.Log("| Hallucination | ENTAILMENT    | FALSE POSITIVE - likely OK |")
	t.Log("| Clean         | -             | CLEAN - no issue           |")
	t.Log("")
	t.Log("=== KEY INSIGHT ===")
	t.Log("Use both signals together:")
	t.Log("1. Hallucination detector as first-pass detector (high recall)")
	t.Log("2. NLI to filter false positives and explain errors (high precision)")
	t.Log("3. Combined = better precision AND explainability")
}

// setupDetectorWithNLI is a helper to create and initialize detector with NLI
func setupDetectorWithNLI(t *testing.T) *HallucinationDetector {
	cfg := &config.HallucinationModelConfig{
		ModelID:   getHallucinationModelPath(),
		Threshold: 0.5,
		UseCPU:    true,
	}

	detector, err := NewHallucinationDetector(cfg)
	if err != nil {
		t.Fatalf("Failed to create detector: %v", err)
	}

	err = detector.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize hallucination detector: %v", err)
	}

	nliCfg := &config.NLIModelConfig{
		ModelID:   getNLIModelPath(),
		Threshold: 0.7,
		UseCPU:    true,
	}
	detector.SetNLIConfig(nliCfg)
	err = detector.InitializeNLI()
	if err != nil {
		t.Fatalf("Failed to initialize NLI: %v", err)
	}

	return detector
}
