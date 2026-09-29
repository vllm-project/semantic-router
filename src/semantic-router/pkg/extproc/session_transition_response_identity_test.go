package extproc

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

func TestResponseSessionHeaderSurvivesHistoryAndReplay(t *testing.T) {
	for _, tc := range []struct {
		name         string
		headerName   string
		headerValue  string
		conversation string
		withHistory  bool
		wantSession  string
	}{
		{name: "standalone", headerName: "x-session-id", headerValue: " client-session ", wantSession: "client-session"},
		{name: "conversation", headerName: "X-Session-ID", headerValue: "client-session", conversation: "conv-test", wantSession: "client-session"},
		{name: "retained_history", headerName: "x-session-id", headerValue: "client-session", withHistory: true, wantSession: "client-session"},
		{name: "absent_header", withHistory: true, wantSession: "respapi:lineage:resp-parent"},
		{name: "blank_header", headerName: "x-session-id", headerValue: "  ", conversation: "conv-test", wantSession: "respapi:conversation:conv-test"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx := &RequestContext{Headers: map[string]string{}, RequestID: "response-session-test"}
			if tc.headerName != "" {
				ctx.Headers[tc.headerName] = tc.headerValue
			}
			// The header phase pins identity before the Responses body is parsed.
			populatePinnedSessionFromHeaders(ctx)
			state := &ResponseObjectState{ConversationID: tc.conversation}
			if tc.withHistory {
				state.PreviousResponseID = "resp-parent"
				state.ConversationHistory = []*responseapi.StoredResponse{{
					ID: "resp-parent", Model: "prior-model",
					Usage: &responseapi.Usage{InputTokens: 12, OutputTokens: 3},
				}}
			}
			state.SessionTrackingID = determineSessionTrackingID(tc.conversation, state.ConversationHistory, "resp-new")
			ctx.ResponseObjectState = state
			populateSessionTransitionFields(ctx)
			require.Equal(t, tc.wantSession, ctx.SessionID)
			record := buildReplayRoutingRecord(ctx, "entrypoint", "selected-model", "route")
			require.Equal(t, tc.wantSession, record.SessionID)
			require.Equal(t, tc.conversation, state.ConversationID, "telemetry identity must not rewrite Responses membership")
			require.Equal(t, state.PreviousResponseID, ctx.PreviousResponseID)
			if tc.withHistory {
				// Header correlation must preserve the actual retained-history facts.
				require.Equal(t, "prior-model", ctx.PreviousModel)
				require.Equal(t, 1, ctx.TurnIndex)
				require.Equal(t, 15, ctx.HistoryTokenCount)
			}
		})
	}
}
