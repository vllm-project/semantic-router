package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestFeedbackApplicabilityFromNeutralConversation(t *testing.T) {
	user := func(text string) llmprotocol.Message {
		return llmprotocol.Message{Role: llmprotocol.RoleUser, Content: textBlocks(text)}
	}
	answer := llmprotocol.Message{Role: llmprotocol.RoleAssistant, Content: textBlocks("Here is the answer.")}
	for _, tc := range []struct {
		name     string
		messages []llmprotocol.Message
		want     bool
	}{
		{"first request", []llmprotocol.Message{user("Explain this subject.")}, false},
		{"user follow-up", []llmprotocol.Message{user("First question"), answer, user("That result is wrong.")}, true},
		{"assistant prefill", []llmprotocol.Message{user("Explain this subject."), answer}, false},
		{"tool continuation", []llmprotocol.Message{user("First question"), answer, user("Find an example"), neutralAssistantToolCalls("lookup"), neutralToolResult("call-lookup", "retrieved document")}, false},
		{"tool calls without answer", []llmprotocol.Message{user("Find an example"), neutralAssistantToolCalls("lookup"), user("Use this source")}, false},
		{"empty user turn", []llmprotocol.Message{user("First question"), answer, user("")}, false},
		{"whitespace user turn", []llmprotocol.Message{user("First question"), answer, user(" \n\t")}, false},
		{"image-only user turn", []llmprotocol.Message{user("First question"), answer, {Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentImage}}}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{}}
			history := extractSignalConversationHistory(&llmprotocol.Request{Messages: tc.messages})
			input := router.prepareSignalEvaluationInput(history)
			assert.Equal(t, tc.want, input.hasAssistantReply)
		})
	}
}
