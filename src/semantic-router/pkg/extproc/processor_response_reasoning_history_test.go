package extproc

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestStoredReasoningResponseCanResumeWithoutSummary(t *testing.T) {
	for _, scope := range []llmprotocol.ReasoningScope{llmprotocol.ReasoningScopeText, llmprotocol.ReasoningScopeSummary} {
		t.Run(string(scope), func(t *testing.T) {
			store := NewMockResponseStore()
			router := &OpenAIRouter{ResponseAPIFilter: NewResponseAPIFilter(store)}
			ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
			_, immediate := router.prepareProtocolRequest([]byte(`{"model":"synthetic","input":"Retain the symbol.","store":true}`), ctx)
			require.Nil(t, immediate)
			ctx.SemanticResponse = &llmprotocol.Response{
				Generation: 1, ID: ctx.ResponseObjectState.GeneratedResponseID, Model: "synthetic",
				StopReason: llmprotocol.StopEndTurn, Usage: authoritativeZeroUsage(),
				Output: []llmprotocol.OutputItem{
					{ID: "reason", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentReasoning, Reasoning: scope, Text: "The symbol is retained."}}},
					{ID: "answer", Role: llmprotocol.RoleAssistant, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "Noted."}}},
				},
			}
			engine, err := router.protocolEngine()
			require.NoError(t, err)
			_, err = engine.EncodeResponse(llmprotocol.OpenAIResponsesV1, *ctx.SemanticResponse, llmprotocol.Envelope{})
			require.NoError(t, err)
			router.persistResponseObject(ctx)
			stored, err := store.GetResponse(t.Context(), ctx.SemanticResponse.ID)
			require.NoError(t, err)
			before, err := json.Marshal(stored)
			require.NoError(t, err)

			body, err := json.Marshal(map[string]any{"model": "synthetic", "input": "Repeat it.", "previous_response_id": stored.ID, "store": false})
			require.NoError(t, err)
			resume := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, TraceContext: t.Context()}
			_, immediate = router.prepareProtocolRequest(body, resume)
			require.Nil(t, immediate)
			snapshot, err := router.extractRequestSignalSnapshot(resume)
			require.NoError(t, err, "a valid retained reasoning item must survive storage serialization")
			require.Equal(t, 2, snapshot.UserMessageCount)
			require.True(t, snapshot.HasAssistantReply)
			require.True(t, resume.ResponseObjectState.ProviderContextApplied)
			require.Empty(t, resume.SemanticRequest.PreviousResponseID)
			require.Len(t, resume.SemanticRequest.Messages, 4)
			reasoning := resume.SemanticRequest.Messages[1].Content[0]
			require.Equal(t, llmprotocol.ContentReasoning, reasoning.Kind)
			require.Equal(t, scope, reasoning.Reasoning)
			require.Equal(t, "The symbol is retained.", reasoning.Text)
			after, err := json.Marshal(stored)
			require.NoError(t, err)
			require.Equal(t, before, after, "materialization must not mutate the stored object")
		})
	}
}
