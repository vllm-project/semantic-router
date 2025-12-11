package responseapi

import "time"

// StoredResponse represents a response stored in the backend.
// This is the internal representation used by ResponseStore implementations.
type StoredResponse struct {
	// ID is the response ID (format: resp_xxxx)
	ID string `json:"id"`

	// Object is always "response"
	Object string `json:"object"`

	// CreatedAt is the Unix timestamp of creation
	CreatedAt int64 `json:"created_at"`

	// Model used to generate the response
	Model string `json:"model"`

	// Status of the response
	Status string `json:"status"`

	// Input items for this response
	Input []InputItem `json:"input"`

	// Output items from the model
	Output []OutputItem `json:"output"`

	// OutputText is the concatenated text output
	OutputText string `json:"output_text"`

	// PreviousResponseID links to the previous response
	PreviousResponseID string `json:"previous_response_id,omitempty"`

	// ConversationID links to the conversation
	ConversationID string `json:"conversation_id,omitempty"`

	// Usage statistics
	Usage *Usage `json:"usage,omitempty"`

	// Instructions used
	Instructions string `json:"instructions,omitempty"`

	// Metadata from the request
	Metadata map[string]string `json:"metadata,omitempty"`

	// TTL is the expiration time
	TTL time.Time `json:"ttl,omitempty"`

	// Error details if status is failed
	Error *ResponseError `json:"error,omitempty"`
}

// StoredConversation represents a conversation stored in the backend.
type StoredConversation struct {
	// ID is the conversation ID (format: conv_xxxx)
	ID string `json:"id"`

	// Object is always "conversation"
	Object string `json:"object"`

	// CreatedAt is the Unix timestamp of creation
	CreatedAt int64 `json:"created_at"`

	// UpdatedAt is the Unix timestamp of last update
	UpdatedAt int64 `json:"updated_at"`

	// Metadata from the request
	Metadata map[string]string `json:"metadata,omitempty"`

	// ResponseIDs are the IDs of responses in this conversation (ordered)
	ResponseIDs []string `json:"response_ids,omitempty"`

	// TTL is the expiration time
	TTL time.Time `json:"ttl,omitempty"`
}
