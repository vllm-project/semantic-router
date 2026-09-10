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

func TestMemoryHistoryBudgetRejectsBeforeSnapshot(t *testing.T) {
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
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := newBudgetHistory()
			tc.mutate(ctx)
			require.ErrorIs(t, validateMemoryHistoryBudget(context.Background(), ctx.messages, ctx.retained), errMemoryHistoryTooLarge)
		})
	}
	ctx := newBudgetHistory()
	require.NoError(t, validateMemoryHistoryBudget(context.Background(), ctx.messages, ctx.retained))
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	require.ErrorIs(t, validateMemoryHistoryBudget(cancelled, ctx.messages, ctx.retained), context.Canceled)
}

type budgetHistory struct {
	messages []llmprotocol.Message
	retained []*responseapi.StoredResponse
}

func newBudgetHistory() *budgetHistory {
	return &budgetHistory{messages: []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Remember my deployment preference."}}}}}
}
