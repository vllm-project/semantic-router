// Package api defines the MCP protocol contract for semantic router classification.
//
// This package provides strongly-typed definitions for the JSON messages exchanged between
// the Go client and MCP classification servers. These types ensure consistency and can be
// used by both client and server implementations.
//
// Protocol Version: 1.0
//
// For Python MCP servers, use these JSON formats when implementing classification tools.
// For other language implementations, these types can serve as reference for the expected format.
package api

// ClassifyRequest represents the arguments for the classify_text tool call
type ClassifyRequest struct {
	Text              string `json:"text"`                         // Text to classify
	WithProbabilities bool   `json:"with_probabilities,omitempty"` // Request full probability distribution
}

// ClassifyResponse represents the response from the classify_text MCP tool.
//
// This is the core classification response format. The MCP server returns both classification
// results (class index and confidence) and optional routing information (model and use_reasoning).
//
// Example JSON:
//
//	{
//	  "class": 3,
//	  "confidence": 0.85,
//	  "model": "openai/gpt-oss-20b",
//	  "use_reasoning": true
//	}
type ClassifyResponse struct {
	// Class is the 0-based index of the predicted category
	Class int `json:"class"`

	// Confidence is the prediction confidence, ranging from 0.0 to 1.0
	Confidence float32 `json:"confidence"`

	// Model is the recommended model for routing this request (optional).
	// If provided, the router will use this model instead of the default_model.
	// Example: "openai/gpt-oss-20b", "anthropic/claude-3-opus"
	Model string `json:"model,omitempty"`

	// UseReasoning indicates whether to enable reasoning mode for this request (optional).
	// If nil/omitted, the router uses its default reasoning configuration.
	// If true, enables reasoning mode. If false, disables reasoning mode.
	UseReasoning *bool `json:"use_reasoning,omitempty"`
}

// ClassifyWithProbabilitiesResponse extends ClassifyResponse with full probability distribution.
//
// This format is used when the client requests detailed classification probabilities for all categories.
// The MCP server should return this format when the classify_text tool is called with
// "with_probabilities": true.
//
// Example JSON:
//
//	{
//	  "class": 3,
//	  "confidence": 0.85,
//	  "probabilities": [0.05, 0.03, 0.07, 0.85, ...],
//	  "model": "openai/gpt-oss-20b",
//	  "use_reasoning": true
//	}
type ClassifyWithProbabilitiesResponse struct {
	// Class is the 0-based index of the predicted category
	Class int `json:"class"`

	// Confidence is the prediction confidence, ranging from 0.0 to 1.0
	Confidence float32 `json:"confidence"`

	// Probabilities is the full probability distribution across all categories.
	// The array length must match the number of categories, and values should sum to ~1.0.
	Probabilities []float32 `json:"probabilities"`

	// Model is the recommended model for routing this request (optional)
	Model string `json:"model,omitempty"`

	// UseReasoning indicates whether to enable reasoning mode for this request (optional)
	UseReasoning *bool `json:"use_reasoning,omitempty"`
}

// ListCategoriesResponse represents the response from the list_categories MCP tool.
//
// This format is used to retrieve the taxonomy of available categories from the MCP server.
// The client calls this tool during initialization to discover which categories the server supports.
//
// Example JSON:
//
//	{
//	  "categories": ["business", "law", "medical", "technical", "general"],
//	  "category_system_prompts": {
//	    "business": "You are a business and finance expert. Provide detailed financial analysis...",
//	    "law": "You are a legal expert. Provide accurate legal information and cite relevant laws...",
//	    "medical": "You are a medical professional. Provide evidence-based health information..."
//	  },
//	  "category_descriptions": {
//	    "business": "Business and finance related queries",
//	    "law": "Legal questions and regulations",
//	    "medical": "Healthcare and medical information"
//	  }
//	}
type ListCategoriesResponse struct {
	// Categories is the ordered list of category names.
	// The index position in this array corresponds to the "class" index in ClassifyResponse.
	// For example, if Categories = ["business", "law", "medical"], then:
	//   - class 0 = "business"
	//   - class 1 = "law"
	//   - class 2 = "medical"
	Categories []string `json:"categories"`

	// CategorySystemPrompts provides optional per-category system prompts that the router
	// can inject when processing queries in specific categories. This allows the MCP server
	// to provide category-specific instructions that guide the LLM's behavior.
	// The map key is the category name, and the value is the system prompt for that category.
	CategorySystemPrompts map[string]string `json:"category_system_prompts,omitempty"`

	// CategoryDescriptions provides optional human-readable descriptions for each category.
	// This can be used for logging, debugging, or providing context to downstream systems.
	CategoryDescriptions map[string]string `json:"category_descriptions,omitempty"`
}
