// Package responseapi provides OpenAI Response API types and handlers.
// The Response API is a stateful API that supports conversation chaining
// via previous_response_id and translates to Chat Completions for backend LLMs.
package responseapi

import (
	"encoding/json"
)

// ResponseAPIRequest represents a request to create a response.
// It follows the OpenAI Response API specification.
type ResponseAPIRequest struct {
	// Model is the model deployment name to use
	Model string `json:"model"`

	// Input can be a string or an array of input items
	// When string, it's treated as a user message
	Input json.RawMessage `json:"input"`

	// PreviousResponseID links this response to a previous one for conversation context
	PreviousResponseID string `json:"previous_response_id,omitempty"`

	// Instructions are system-level instructions for the model
	Instructions string `json:"instructions,omitempty"`

	// Store determines if the response should be stored (default: true)
	Store *bool `json:"store,omitempty"`

	// MaxOutputTokens limits the response length
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	// Temperature controls randomness (0.0-2.0)
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP controls nucleus sampling
	TopP *float64 `json:"top_p,omitempty"`

	// Tools are function definitions available to the model
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice controls how tools are selected
	ToolChoice interface{} `json:"tool_choice,omitempty"`

	// Metadata is user-defined key-value pairs
	Metadata map[string]string `json:"metadata,omitempty"`

	// Stream enables streaming responses
	Stream bool `json:"stream,omitempty"`

	// ConversationID links to a conversation object (optional)
	ConversationID string `json:"conversation_id,omitempty"`
}

// ResponseAPIResponse represents the response from the Response API.
type ResponseAPIResponse struct {
	// ID is the unique identifier for this response (format: resp_xxxx)
	ID string `json:"id"`

	// Object is always "response"
	Object string `json:"object"`

	// CreatedAt is the Unix timestamp of creation
	CreatedAt int64 `json:"created_at"`

	// Model is the model that generated the response
	Model string `json:"model"`

	// Status is the response status: "completed", "failed", "in_progress", "cancelled"
	Status string `json:"status"`

	// Output contains the response output items
	Output []OutputItem `json:"output"`

	// OutputText is a convenience field with concatenated text output
	OutputText string `json:"output_text,omitempty"`

	// PreviousResponseID links to the previous response in the chain
	PreviousResponseID string `json:"previous_response_id,omitempty"`

	// ConversationID links to the conversation object
	ConversationID string `json:"conversation_id,omitempty"`

	// Usage contains token usage statistics
	Usage *Usage `json:"usage,omitempty"`

	// Error contains error details if status is "failed"
	Error *ResponseError `json:"error,omitempty"`

	// IncompleteDetails explains why a response is incomplete
	IncompleteDetails *IncompleteDetails `json:"incomplete_details,omitempty"`

	// Instructions used for this response
	Instructions string `json:"instructions,omitempty"`

	// Metadata from the request
	Metadata map[string]string `json:"metadata,omitempty"`

	// Temperature used
	Temperature *float64 `json:"temperature,omitempty"`

	// TopP used
	TopP *float64 `json:"top_p,omitempty"`

	// MaxOutputTokens used
	MaxOutputTokens *int `json:"max_output_tokens,omitempty"`

	// Tools available
	Tools []Tool `json:"tools,omitempty"`

	// ToolChoice setting
	ToolChoice interface{} `json:"tool_choice,omitempty"`

	// ParallelToolCalls setting
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	// Reasoning information for reasoning models
	Reasoning *Reasoning `json:"reasoning,omitempty"`

	// Text configuration
	Text *TextConfig `json:"text,omitempty"`

	// Truncation strategy
	Truncation *Truncation `json:"truncation,omitempty"`

	// User identifier
	User string `json:"user,omitempty"`

	// ReasoningEffort for reasoning models
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
}

// InputItem represents an input item in a Response API request.
type InputItem struct {
	// Type is the item type: "message", "item_reference"
	Type string `json:"type"`

	// ID is the item identifier (for item_reference)
	ID string `json:"id,omitempty"`

	// Role is the message role: "user", "assistant", "system"
	Role string `json:"role,omitempty"`

	// Content can be string or array of content parts
	Content json.RawMessage `json:"content,omitempty"`

	// Status of the item
	Status string `json:"status,omitempty"`
}

// OutputItem represents an output item in a Response API response.
type OutputItem struct {
	// Type is the item type: "message", "function_call", "function_call_output"
	Type string `json:"type"`

	// ID is the item identifier
	ID string `json:"id,omitempty"`

	// Role is the message role (for message type)
	Role string `json:"role,omitempty"`

	// Content is the message content (for message type)
	Content []ContentPart `json:"content,omitempty"`

	// Status of the output item
	Status string `json:"status,omitempty"`

	// Function call fields
	Name      string `json:"name,omitempty"`
	CallID    string `json:"call_id,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
}

// DeleteResponseResult represents the result of deleting a response.
type DeleteResponseResult struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Deleted bool   `json:"deleted"`
}

// InputItemsListResponse represents the response for GET /v1/responses/{id}/input_items.
type InputItemsListResponse struct {
	Object  string      `json:"object"`
	Data    []InputItem `json:"data"`
	FirstID string      `json:"first_id"`
	LastID  string      `json:"last_id"`
	HasMore bool        `json:"has_more"`
}
