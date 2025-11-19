package testcases

import (
	"encoding/json"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// KeywordTestCase represents a test case for keyword routing
type KeywordTestCase struct {
	Name               string   `json:"name"`
	Description        string   `json:"description"`
	Query              string   `json:"query"`
	ExpectedCategory   string   `json:"expected_category"`
	ExpectedConfidence float64  `json:"expected_confidence"`
	MatchedKeywords    []string `json:"matched_keywords"`
}

// EmbeddingTestCase represents a test case for embedding-based routing
type EmbeddingTestCase struct {
	Name              string  `json:"name"`
	Description       string  `json:"description"`
	Query             string  `json:"query"`
	ExpectedCategory  string  `json:"expected_category"`
	MinSimilarity     float64 `json:"min_similarity"`
	AggregationMethod string  `json:"aggregation_method"`
	ModelType         string  `json:"model_type"`
}

// HybridTestCase represents a test case for hybrid routing (priority testing)
type HybridTestCase struct {
	Name                  string  `json:"name"`
	Description           string  `json:"description"`
	Query                 string  `json:"query"`
	ExpectedCategory      string  `json:"expected_category"`
	ExpectedRoutingMethod string  `json:"expected_routing_method"` // "keyword", "embedding", "mcp"
	ExpectedConfidence    float64 `json:"expected_confidence"`
}

// EntropyTestCase represents a test case for entropy-based routing
type EntropyTestCase struct {
	Name              string  `json:"name"`
	Description       string  `json:"description"`
	Query             string  `json:"query"`
	ExpectedEntropy   float64 `json:"expected_entropy"`
	ExpectedReasoning bool    `json:"expected_reasoning"`
	EntropyThreshold  float64 `json:"entropy_threshold"`
}

// ReasoningControlTestCase represents a test case for reasoning control
type ReasoningControlTestCase struct {
	Name              string `json:"name"`
	Description       string `json:"description"`
	Query             string `json:"query"`
	Category          string `json:"category"`
	ExpectedReasoning bool   `json:"expected_reasoning"`
	EffortLevel       string `json:"effort_level"`
	ModelFamily       string `json:"model_family"`
}

// ToolSelectionTestCase represents a test case for tool selection
type ToolSelectionTestCase struct {
	Name                string   `json:"name"`
	Description         string   `json:"description"`
	Query               string   `json:"query"`
	ExpectedTools       []string `json:"expected_tools"`
	TopK                int      `json:"top_k"`
	SimilarityThreshold float64  `json:"similarity_threshold"`
}

// LoadKeywordTestCases loads keyword test cases from a JSON file
func LoadKeywordTestCases(path string) ([]KeywordTestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cases []KeywordTestCase
	err = json.Unmarshal(data, &cases)
	return cases, err
}

// LoadEmbeddingTestCases loads embedding test cases from a JSON file
func LoadEmbeddingTestCases(path string) ([]EmbeddingTestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cases []EmbeddingTestCase
	err = json.Unmarshal(data, &cases)
	return cases, err
}

// LoadHybridTestCases loads hybrid routing test cases from a JSON file
func LoadHybridTestCases(path string) ([]HybridTestCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cases []HybridTestCase
	err = json.Unmarshal(data, &cases)
	return cases, err
}

// CreateKeywordTestRules creates standard keyword rules for testing
// Note: Rules are evaluated in order. NOR rule is last to avoid matching everything.
func CreateKeywordTestRules() []config.KeywordRule {
	return []config.KeywordRule{
		{
			Name:          "urgent_request",
			Operator:      "OR",
			Keywords:      []string{"urgent", "immediate", "asap", "emergency"},
			CaseSensitive: false,
		},
		{
			Name:          "sensitive_data",
			Operator:      "AND",
			Keywords:      []string{"SSN", "credit card"},
			CaseSensitive: false,
		},
		{
			Name:          "case_sensitive_test",
			Operator:      "OR",
			Keywords:      []string{"SECRET"},
			CaseSensitive: true,
		},
		{
			Name:          "secret_detection",
			Operator:      "OR",
			Keywords:      []string{"secret"},
			CaseSensitive: false,
		},
		{
			Name:          "version_check",
			Operator:      "OR",
			Keywords:      []string{"1.0", "2.0", "3.0"},
			CaseSensitive: false,
		},
		{
			Name:          "wildcard_test",
			Operator:      "OR",
			Keywords:      []string{"*"},
			CaseSensitive: false,
		},
		// NOR rule at end - matches when NO spam keywords present
		// This will match most text, so it's placed last
		{
			Name:          "spam",
			Operator:      "NOR",
			Keywords:      []string{"buy now", "free money", "click here"},
			CaseSensitive: false,
		},
	}
}

// CreateTestKeywordClassifier creates a keyword classifier instance for testing
func CreateTestKeywordClassifier(rules []config.KeywordRule) (*classification.KeywordClassifier, error) {
	return classification.NewKeywordClassifier(rules)
}
