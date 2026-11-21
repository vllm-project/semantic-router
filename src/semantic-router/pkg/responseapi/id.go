package responseapi

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
)

// ID prefixes following OpenAI conventions
const (
	ResponseIDPrefix            = "resp_"
	ItemIDPrefix                = "item_"
	MessageIDPrefix             = "msg_"
	ReasoningIDPrefix           = "rs_"
	FunctionCallIDPrefix        = "fc_"
	CustomToolCallIDPrefix      = "ctc_"
	ImageGenerationCallIDPrefix = "ig_"
	ConversationIDPrefix        = "conv_"
)

// OutputItemIDPrefix is the prefix the Responses API gives the id of an
// output item of itemType, and item_ for a type it is not listed for.
func OutputItemIDPrefix(itemType string) string {
	switch itemType {
	case ItemTypeMessage:
		return MessageIDPrefix
	case ItemTypeReasoning:
		return ReasoningIDPrefix
	case ItemTypeFunctionCall:
		return FunctionCallIDPrefix
	case ItemTypeCustomToolCall:
		return CustomToolCallIDPrefix
	case ItemTypeImageGenerationCall:
		return ImageGenerationCallIDPrefix
	}
	return ItemIDPrefix
}

// GenerateResponseID generates a new response ID with the resp_ prefix.
func GenerateResponseID() string {
	return ResponseIDPrefix + generateRandomID(24)
}

// GenerateItemID generates a new item ID with the item_ prefix.
func GenerateItemID() string {
	return ItemIDPrefix + generateRandomID(24)
}

// GenerateConversationID generates a new conversation ID with the conv_ prefix.
func GenerateConversationID() string {
	return ConversationIDPrefix + generateRandomID(24)
}

// generateRandomID generates a random hex string of the specified length.
func generateRandomID(length int) string {
	bytes := make([]byte, length/2)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to a simple counter if crypto/rand fails
		return fmt.Sprintf("%024d", 0)
	}
	return hex.EncodeToString(bytes)
}

// IsValidResponseID checks if an ID has the correct response prefix.
func IsValidResponseID(id string) bool {
	return strings.HasPrefix(id, ResponseIDPrefix) && len(id) > len(ResponseIDPrefix)
}
