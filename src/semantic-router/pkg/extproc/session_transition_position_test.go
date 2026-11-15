package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func textMessage(role llmprotocol.Role, text string) llmprotocol.Message {
	return llmprotocol.Message{Role: role, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: text}}}
}

func TestSessionPositionFactsGroupToolContinuations(t *testing.T) {
	ctx := &RequestContext{SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "first question"),
		textMessage(llmprotocol.RoleAssistant, "calling tool"),
		// Tool output text the client sent as user content continues the
		// tool loop; it must not advance the turn index.
		{
			Role: llmprotocol.RoleUser, WithToolResults: true,
			Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "tool output"}},
		},
		textMessage(llmprotocol.RoleUser, "second question"),
	}}}

	turnIndex, historyTokens := sessionPositionFacts(ctx)
	require.Equal(t, 1, turnIndex, "two distinct user turns -> zero-based index 1")
	// History excludes the trailing user message; the continuation text stays
	// part of the history estimate.
	require.Equal(t, len("first question")/4+len("calling tool")/4+len("tool output")/4, historyTokens)
}

func TestSessionPositionFactsPreferOriginalSnapshot(t *testing.T) {
	original := contextcompression.CaptureHistory(&llmprotocol.Request{Messages: []llmprotocol.Message{
		textMessage(llmprotocol.RoleUser, "original turn one"),
		textMessage(llmprotocol.RoleAssistant, "original reply"),
		textMessage(llmprotocol.RoleUser, "original turn two"),
	}})
	ctx := &RequestContext{
		OriginalContextHistory: original,
		SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{
			textMessage(llmprotocol.RoleUser, "edited"),
		}},
	}

	turnIndex, historyTokens := sessionPositionFacts(ctx)
	require.Equal(t, 1, turnIndex, "renumbered requests must not move the turn index")
	require.Equal(t, len("original turn one")/4+len("original reply")/4, historyTokens,
		"history excludes only the trailing user message")
}

func TestSessionPositionFactsDegradesWithoutUserHistory(t *testing.T) {
	require.Equal(t, 0, func() int { tIndex, _ := sessionPositionFacts(nil); return tIndex }())

	empty := &RequestContext{}
	turnIndex, historyTokens := sessionPositionFacts(empty)
	require.Equal(t, 0, turnIndex)
	require.Equal(t, 0, historyTokens)

	toolOnly := &RequestContext{SemanticRequest: &llmprotocol.Request{Messages: []llmprotocol.Message{
		{Role: llmprotocol.RoleTool, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "output"}}},
	}}}
	turnIndex, historyTokens = sessionPositionFacts(toolOnly)
	require.Equal(t, 0, turnIndex)
	require.Equal(t, len("output")/4, historyTokens)
}
