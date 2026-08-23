package responseapi

import (
	"strings"
	"time"
)

// ResponseOwnerMode identifies how the Router established ownership of a
// retained Responses object. It is an internal persistence boundary and is
// never accepted from a client request.
type ResponseOwnerMode string

const (
	// ResponseOwnerAuthenticated identifies an object owned by one authenticated
	// namespace, logical API key, and user.
	ResponseOwnerAuthenticated ResponseOwnerMode = "authenticated"

	// ResponseOwnerAnonymousPublicNamespace identifies an object created on an
	// access-disabled public Router generation. It is isolated by that immutable
	// generation's namespace and has no caller-supplied identity.
	ResponseOwnerAnonymousPublicNamespace ResponseOwnerMode = "anonymous_public_namespace"
)

// ResponseOwner is the exact internal owner of a retained Responses object.
// APIKeyID is the logical key identifier, never credential material.
type ResponseOwner struct {
	Mode        ResponseOwnerMode `json:"mode"`
	NamespaceID string            `json:"namespace_id"`
	APIKeyID    string            `json:"api_key_id,omitempty"`
	UserID      string            `json:"user_id,omitempty"`
}

// Valid reports whether all fields required by the selected ownership mode are
// present and already canonical. Storage never canonicalizes caller input.
func (owner ResponseOwner) Valid() bool {
	if owner.NamespaceID == "" || strings.TrimSpace(owner.NamespaceID) != owner.NamespaceID {
		return false
	}
	switch owner.Mode {
	case ResponseOwnerAuthenticated:
		return owner.APIKeyID != "" && strings.TrimSpace(owner.APIKeyID) == owner.APIKeyID &&
			owner.UserID != "" && strings.TrimSpace(owner.UserID) == owner.UserID
	case ResponseOwnerAnonymousPublicNamespace:
		return owner.APIKeyID == "" && owner.UserID == ""
	default:
		return false
	}
}

// StoredResponse represents a response stored in the backend.
// This is the internal representation used by ResponseStore implementations.
type StoredResponse struct {
	// Owner is the exact process-derived owner of this retained object.
	Owner ResponseOwner `json:"owner"`

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
	// Owner is the exact process-derived owner of this retained object.
	Owner ResponseOwner `json:"owner"`

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
