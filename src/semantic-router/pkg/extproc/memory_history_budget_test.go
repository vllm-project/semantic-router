package extproc

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestMemorySnapshotBudgetRejectsBeforeSnapshot(t *testing.T) {
	for _, tc := range []struct {
		name   string
		mutate func(*budgetHistory)
	}{
		{"messages", func(ctx *budgetHistory) {
			ctx.messages = make([]llmprotocol.Message, maxMemorySnapshotMessages+1)
		}},
		{"bytes", func(ctx *budgetHistory) {
			ctx.messages[0].Content[0].Text = strings.Repeat("x", maxMemorySnapshotBytes+1)
		}},
		{"blocks", func(ctx *budgetHistory) {
			ctx.messages[0].Content = make([]llmprotocol.Content, maxMemorySnapshotNodes+1)
		}},
		{"nested", func(ctx *budgetHistory) {
			content := []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "tail"}}
			for i := 0; i <= maxMemorySnapshotDepth; i++ {
				content = []llmprotocol.Content{{Kind: llmprotocol.ContentToolResult, ToolResult: &llmprotocol.ToolResult{Content: content}}}
			}
			ctx.messages[0].Content = content
		}},
		{"stored_raw_input", func(ctx *budgetHistory) {
			ctx.retained = []*responseapi.StoredResponse{{
				Input: []responseapi.InputItem{{Type: "message", Content: json.RawMessage(strings.Repeat(" ", maxMemorySnapshotBytes+1))}},
			}}
		}},
		{"combined_history", func(ctx *budgetHistory) {
			ctx.retained = []*responseapi.StoredResponse{{
				Input: make([]responseapi.InputItem, maxMemorySnapshotMessages),
			}}
		}},
		{"response_bytes", func(ctx *budgetHistory) {
			ctx.response.Output[0].Content[0].Text = strings.Repeat("x", maxMemorySnapshotBytes+1)
		}},
		{"response_items", func(ctx *budgetHistory) {
			ctx.response.Output = make([]llmprotocol.OutputItem, maxMemorySnapshotNodes+1)
		}},
		{"response_blocks", func(ctx *budgetHistory) {
			ctx.response.Output[0].Content = make([]llmprotocol.Content, maxMemorySnapshotNodes+1)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := newBudgetHistory()
			tc.mutate(ctx)
			require.ErrorIs(t, validateMemorySnapshotBudget(context.Background(), ctx.messages, ctx.retained, ctx.response), errMemoryHistoryTooLarge)
		})
	}
	ctx := newBudgetHistory()
	require.NoError(t, validateMemorySnapshotBudget(context.Background(), ctx.messages, ctx.retained, ctx.response))
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, validateMemorySnapshotBudget(cancelled, ctx.messages, ctx.retained, ctx.response), context.Canceled)
}

func TestMemorySnapshotBudgetSharesResponseAndHistoryBytes(t *testing.T) {
	ctx := newBudgetHistory()
	ctx.retained = []*responseapi.StoredResponse{{OutputText: "previous answer"}}
	historyBytes := len("user") + len("text") + len(ctx.messages[0].Content[0].Text) + len(ctx.retained[0].OutputText)
	responseBytes := maxMemorySnapshotBytes - historyBytes
	ctx.response.Output[0].Content = []llmprotocol.Content{
		{Kind: llmprotocol.ContentText, Text: strings.Repeat("x", responseBytes-3)},
		{Kind: llmprotocol.ContentRefusal, Text: "界"}, // Three UTF-8 bytes, one rune.
	}
	require.NoError(t, validateMemorySnapshotBudget(context.Background(), ctx.messages, ctx.retained, ctx.response))
	ctx.response.Output[0].Content[1].Text += "x"
	require.ErrorIs(t, validateMemorySnapshotBudget(context.Background(), ctx.messages, ctx.retained, ctx.response), errMemoryHistoryTooLarge)
	require.NoError(t, validateMemorySnapshotBudget(context.Background(), ctx.messages, ctx.retained, nil))
}

type budgetHistory struct {
	messages []llmprotocol.Message
	retained []*responseapi.StoredResponse
	response *llmprotocol.Response
}

func newBudgetHistory() *budgetHistory {
	return &budgetHistory{
		messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Remember my deployment preference."}}}},
		response: &llmprotocol.Response{Output: []llmprotocol.OutputItem{{Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "I will remember."}}}}},
	}
}
