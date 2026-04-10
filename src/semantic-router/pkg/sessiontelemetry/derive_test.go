package sessiontelemetry

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestDeriveChatCompletionsSessionID_Stability(t *testing.T) {
	msgs := []ChatMessage{
		{Role: "user", Content: "Hello"},
	}
	a := DeriveChatCompletionsSessionID(msgs, "u1")
	b := DeriveChatCompletionsSessionID(msgs, "u1")
	assert.Equal(t, a, b)

	c := DeriveChatCompletionsSessionID(msgs, "u2")
	assert.NotEqual(t, a, c)
}

func TestChatTurnNumber(t *testing.T) {
	assert.Equal(t, 1, ChatTurnNumber([]ChatMessage{{Role: "user", Content: "hi"}}))
	assert.Equal(t, 2, ChatTurnNumber([]ChatMessage{
		{Role: "user", Content: "hi"},
		{Role: "assistant", Content: "x"},
		{Role: "user", Content: "y"},
	}))
}
