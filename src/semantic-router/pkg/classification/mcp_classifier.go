package classification

import (
	"context"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	mcpclient "github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp"
)

const (
	// DefaultMCPThreshold is the default confidence threshold for MCP classification.
	// For multi-class classification, a value of 0.5 means that a predicted class must have at least 50% confidence
	// to be selected. Adjust this threshold as needed for your use case.
	DefaultMCPThreshold = 0.5
)

// MCPClassificationResult holds the classification result with routing information from MCP server
type MCPClassificationResult struct {
	Class        int
	Confidence   float32
	CategoryName string
	Model        string // Model recommended by MCP server
	UseReasoning *bool  // Whether to use reasoning (nil means use default)
}

// MCPCategoryInitializer initializes MCP connection for category classification
type MCPCategoryInitializer interface {
	Init(cfg *config.RouterConfig) error
	Close() error
}

// MCPCategoryInference performs classification via MCP
type MCPCategoryInference interface {
	Classify(ctx context.Context, text string) (candle_binding.ClassResult, error)
	ClassifyWithProbabilities(ctx context.Context, text string) (candle_binding.ClassResultWithProbs, error)
	ListCategories(ctx context.Context) (*CategoryMapping, error)
}

// MCPCategoryClassifier implements both MCPCategoryInitializer and MCPCategoryInference.
//
// Protocol Contract:
// This client relies on the MCP server to respect the protocol defined in the
// github.com/vllm-project/semantic-router/src/semantic-router/pkg/mcp/api package.
//
// The MCP server must implement these tools:
//  1. list_categories - Returns api.ListCategoriesResponse
//  2. classify_text - Returns api.ClassifyResponse or api.ClassifyWithProbabilitiesResponse
//
// The MCP server controls both classification AND routing decisions. When the server returns
// "model" and "use_reasoning" in the classification response, the router will use those values.
// If not provided, the router falls back to the default_model configuration.
//
// For detailed type definitions and examples, see the api package documentation.
type MCPCategoryClassifier struct {
	client   mcpclient.MCPClient
	toolName string
	config   *config.RouterConfig
}

// createMCPCategoryInitializer creates an MCP category initializer
func createMCPCategoryInitializer() MCPCategoryInitializer {
	return &MCPCategoryClassifier{}
}

// createMCPCategoryInference creates an MCP category inference from the initializer
func createMCPCategoryInference(initializer MCPCategoryInitializer) MCPCategoryInference {
	if classifier, ok := initializer.(*MCPCategoryClassifier); ok {
		return classifier
	}
	return nil
}

// withMCPCategory creates an option function for MCP category classifier
func withMCPCategory(mcpInitializer MCPCategoryInitializer, mcpInference MCPCategoryInference) option {
	return func(c *Classifier) {
		c.mcpCategoryInitializer = mcpInitializer
		c.mcpCategoryInference = mcpInference
	}
}
