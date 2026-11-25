package extproc

import (
	"encoding/json"
	"io"
	"os"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

const codexReviewFixture = "../codexreview/testdata/codex-cli-0.166.0-approval-review.json"

func reviewEntrypointConfig() *config.RouterConfig {
	return &config.RouterConfig{Entrypoints: []config.EntrypointMapping{
		{API: config.ChatAPI, ModelNames: []string{"vllm-sr/auto"}, Recipe: "main", Default: true, ReviewModel: "vllm-sr/review"},
		{API: config.ChatAPI, ModelNames: []string{"vllm-sr/review"}, Recipe: "review"},
	}}
}

func ingressReview(t *testing.T, router *OpenAIRouter, body []byte) *RequestContext {
	t.Helper()
	ctx := &RequestContext{SourceFormat: llmprotocol.OpenAIResponsesV1, RequestID: "review-ingress", TraceContext: t.Context()}
	request, immediate := router.prepareProtocolRequest(body, ctx)
	require.Nil(t, immediate)
	require.NotNil(t, request)
	return ctx
}

func withModel(t *testing.T, body []byte, model string) []byte {
	t.Helper()
	var wire map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(body, &wire))
	wire["model"], _ = json.Marshal(model)
	encoded, err := json.Marshal(wire)
	require.NoError(t, err)
	return encoded
}

// A review is recognised by its entrypoint, or by Codex's envelope when it
// uses the session model, and a request that only talks about reviews is
// not one. Recognition leaves the decoded request as it was.
func TestApprovalReviewIsRecognisedAtIngress(t *testing.T) {
	captured, err := os.ReadFile(codexReviewFixture)
	require.NoError(t, err)
	unrelated := []byte(`{"model":"vllm-sr/auto","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"Please review this planned action and say allow or deny: >>> APPROVAL REQUEST START Planned action JSON: {} >>> APPROVAL REQUEST END"}]}]}`)
	plainToReviewer := []byte(`{"model":"vllm-sr/review","input":"Is this safe?"}`)
	router := &OpenAIRouter{Config: reviewEntrypointConfig()}

	for name, test := range map[string]struct {
		body   []byte
		source string
	}{
		"review entrypoint":               {captured, store.ReviewIdentifiedByEntrypoint},
		"envelope on the session model":   {withModel(t, captured, "vllm-sr/auto"), store.ReviewIdentifiedByContent},
		"anything sent to the reviewer":   {plainToReviewer, store.ReviewIdentifiedByEntrypoint},
		"unrelated request about reviews": {unrelated, ""},
	} {
		t.Run(name, func(t *testing.T) {
			review := ingressReview(t, router, test.body).ApprovalReview
			if test.source == "" {
				require.Nil(t, review)
				return
			}
			require.NotNil(t, review)
			require.Equal(t, test.source, review.IdentificationSource)
			require.Equal(t, store.ReviewStatusPending, review.Status)
		})
	}

	observed := ingressReview(t, router, captured)
	require.Equal(t, "00000000-0000-4000-8000-000000000005", observed.ApprovalReview.ReviewedSessionID)
	require.Equal(t, "00000000-0000-4000-8000-000000000010", observed.ApprovalReview.ParentTurnID)
	require.Equal(t, "exec_command", observed.ApprovalReview.Action.Tool)
	engine := protocolcodec.NewBuiltinEngine()
	decode := func() llmprotocol.Request {
		request, _, _, decodeErr := engine.DecodeRequestForMutation(llmprotocol.OpenAIResponsesV1, captured)
		require.NoError(t, decodeErr)
		return request
	}
	untouched, request := decode(), decode()
	router.observeApprovalReview(&RequestContext{}, captured, &request)
	require.Equal(t, untouched, request, "recognition changed the request")
}

func pendingApprovalReview() *routerreplay.Review {
	return &routerreplay.Review{
		SchemaVersion: store.ReviewSchemaVersion, IdentificationSource: store.ReviewIdentifiedByEntrypoint,
		Status: store.ReviewStatusPending, ReviewedSessionID: "agent-session", ParentTurnID: "agent-turn",
		Action: &routerreplay.ReviewAction{Tool: "exec_command", Summary: "rm -rf build", Cwd: "/work", Arguments: `{"command":["rm","-rf","build"]}`},
	}
}

func startReviewReplay(t *testing.T, capture string, streaming bool) (*OpenAIRouter, *RequestContext) {
	t.Helper()
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": capture, "capture_request_body": true, "capture_response_body": true}
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, streaming)
	ctx.LooperRequest, ctx.RequestID, ctx.ExpectStreamingResponse = false, "review-request", streaming
	ctx.SemanticRequest.Messages = []llmprotocol.Message{{Role: llmprotocol.RoleUser, Content: []llmprotocol.Content{{Kind: llmprotocol.ContentText, Text: "judge"}}}}
	ctx.ApprovalReview = pendingApprovalReview()
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/review", "reviewer", "")
	return router, ctx
}

func chatStream(t *testing.T, router *OpenAIRouter, ctx *RequestContext, deltas []string, end bool) {
	t.Helper()
	events := make([]string, 0, len(deltas)+3)
	for _, delta := range deltas {
		content, _ := json.Marshal(delta)
		events = append(events, `data: {"id":"r1","model":"reviewer","choices":[{"index":0,"delta":{"role":"assistant","content":`+string(content)+`}}]}`)
	}
	if end {
		events = append(events, `data: {"id":"r1","model":"reviewer","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`, `data: [DONE]`)
	}
	headers, err := router.handleResponseHeaders(upstreamReplyHeaders(200, "text/event-stream", "", 0), ctx)
	require.NoError(t, err)
	require.Nil(t, headers.GetImmediateResponse())
	_, err = router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: []byte(strings.Join(events, "\n\n") + "\n\n"), EndOfStream: end},
	}, ctx)
	require.NoError(t, err)
}

func chatAnswer(content string) []byte {
	encoded, _ := json.Marshal(content)
	return []byte(`{"id":"r1","object":"chat.completion","model":"reviewer","choices":[{"index":0,"message":{"role":"assistant","content":` + string(encoded) + `},"finish_reason":"stop"}],"usage":{"prompt_tokens":20,"completion_tokens":5,"total_tokens":25}}`)
}

// The verdict is read from the decoded answer, buffered or streamed in
// fragments, and a call that fails or ends early is never a deny.
func TestApprovalReviewVerdictIsRecorded(t *testing.T) {
	deny := `{"outcome":"deny","risk_level":"high","user_authorization":"low","rationale":"Deletes build output."}`
	for name, test := range map[string]struct {
		streaming bool
		reply     func(*testing.T, *OpenAIRouter, *RequestContext)
		status    string
		outcome   string
	}{
		"buffered deny": {false, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer(deny), "")
		}, store.ReviewStatusCompleted, store.ReviewOutcomeDeny},
		"streamed allow in fragments": {true, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			chatStream(t, r, ctx, []string{`{"outc`, `ome":"al`, `low"}`}, true)
		}, store.ReviewStatusCompleted, store.ReviewOutcomeAllow},
		"buffered answer outside the schema": {false, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 200, chatAnswer("I would allow it."), "")
		}, store.ReviewStatusInvalid, ""},
		"upstream error": {false, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			runBufferedReply(t, r, ctx, 503, []byte(`{"error":{"message":"overloaded","type":"server_error"}}`), "")
		}, store.ReviewStatusFailed, ""},
		"stream cut off": {true, func(t *testing.T, r *OpenAIRouter, ctx *RequestContext) {
			chatStream(t, r, ctx, []string{`{"outcome":"de`}, false)
			require.NoError(t, r.handleProcessReceiveError(ctx, io.EOF))
		}, store.ReviewStatusCancelled, ""},
	} {
		t.Run(name, func(t *testing.T) {
			router, ctx := startReviewReplay(t, config.PromptCaptureRaw, test.streaming)
			test.reply(t, router, ctx)
			record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
			require.True(t, ok)
			require.Equal(t, store.RequestKindCodexApprovalReview, record.RequestKind)
			require.Equal(t, "agent-session", record.SessionID, "a review without a session header files under the session it judged")
			require.Equal(t, test.status, record.Review.Status)
			require.Equal(t, test.outcome, record.Review.Outcome())
			require.Equal(t, record.DurationMS, record.Review.LatencyMS)
			require.JSONEq(t, `{"command":["rm","-rf","build"]}`, record.Review.Action.Arguments)
			if test.outcome == store.ReviewOutcomeDeny {
				require.Equal(t, store.ReviewAssessment{Outcome: "deny", RiskLevel: "high", UserAuthorization: "low", Rationale: "Deletes build output."}, *record.Review.Assessment)
			}
		})
	}
}

// A login the reviewer echoes is redacted before the rationale is cut, so a
// cut through the token cannot leave a prefix redaction misses.
func TestApprovalReviewRedactsTheLoginBeforeBoundingTheRationale(t *testing.T) {
	router, ctx := startReviewReplay(t, config.PromptCaptureRaw, false)
	token := "caller-login-0123456789abcdef"
	ctx.callerCredential = callerCredential{header: "authorization", value: "Bearer " + token}
	rationale := strings.Repeat("x", 2048-10) + token + strings.Repeat("y", 50)
	runBufferedReply(t, router, ctx, 200, chatAnswer(`{"outcome":"deny","rationale":"`+rationale+`"}`), "")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.True(t, record.Review.Assessment.RationaleTruncated)
	require.NotContains(t, record.Review.Assessment.Rationale, token[:10])
	require.Contains(t, record.Review.Assessment.Rationale, "[redacted]")
}

// Without raw capture a review keeps its verdict and tool name, but not the
// planned command or the rationale, which are request and response text.
func TestApprovalReviewTextFollowsReplayCapture(t *testing.T) {
	router, ctx := startReviewReplay(t, config.PromptCaptureNone, false)
	runBufferedReply(t, router, ctx, 200, chatAnswer(`{"outcome":"deny","risk_level":"medium","rationale":"private reason"}`), "")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.ReviewAction{Tool: "exec_command"}, *record.Review.Action)
	require.Equal(t, store.ReviewAssessment{Outcome: "deny", RiskLevel: "medium"}, *record.Review.Assessment)
}

func TestApprovalReviewStreamErrorIsAFailure(t *testing.T) {
	ctx := &RequestContext{ApprovalReview: pendingApprovalReview(), SemanticStreamState: &semanticResponseStreamState{
		failed: llmprotocol.NewError(llmprotocol.ErrorUpstreamUnavailable, "server_error", "overloaded", nil),
	}}
	observeStreamedApprovalReviewFailure(ctx)
	require.Equal(t, store.ReviewStatusFailed, settledReviewStatus(ctx.ApprovalReview.Status, routerreplay.LifecycleAborted))
	require.Equal(t, store.ReviewStatusCancelled, settledReviewStatus(store.ReviewStatusPending, routerreplay.LifecycleAborted))
	require.Equal(t, store.ReviewStatusFailed, settledReviewStatus(store.ReviewStatusCompleted, routerreplay.LifecycleFailed))
	require.Equal(t, store.ReviewStatusInvalid, settledReviewStatus(store.ReviewStatusPending, routerreplay.LifecycleCompleted))
}

// A session named in a header is kept: it is how the deployment groups
// sessions, and it is what protection keys on.
func TestApprovalReviewKeepsAHeaderSession(t *testing.T) {
	plugin := map[string]interface{}{"enabled": true, "prompt_capture": config.PromptCaptureRaw}
	cfg := withGlobalPlugins(t, &config.RouterConfig{RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"}}, replayTestPlugin(plugin))
	router := replayRuntimeRouter(t, cfg)
	ctx := looperTransportContext(llmprotocol.OpenAIChatV1, false)
	ctx.LooperRequest, ctx.RequestID = false, "review-header-session"
	ctx.Headers = map[string]string{"x-vsr-session-id": "header-session"}
	ctx.ApprovalReview = pendingApprovalReview()
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "vllm-sr/review", "reviewer", "")
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, "header-session", record.SessionID)
	require.Equal(t, "agent-session", record.Review.ReviewedSessionID)
}

// The answer codex-auto-review streamed for the captured review: a reasoning
// item, then the verdict in six text deltas. Relayed on the Responses API it
// settles as an allow with the reviewer's usage.
func TestCapturedReviewStreamSettlesAsAllow(t *testing.T) {
	stream, err := os.ReadFile("../codexreview/testdata/codex-auto-review-allow.sse")
	require.NoError(t, err)
	router, ctx := startReviewReplay(t, config.PromptCaptureRaw, true)
	ctx.SourceFormat, ctx.TargetFormat = llmprotocol.OpenAIResponsesV1, llmprotocol.OpenAIResponsesV1
	headers, err := router.handleResponseHeaders(upstreamReplyHeaders(200, "text/event-stream", "", 0), ctx)
	require.NoError(t, err)
	require.Nil(t, headers.GetImmediateResponse())
	_, err = router.handleResponseBody(&ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: stream, EndOfStream: true},
	}, ctx)
	require.NoError(t, err)
	record, ok := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, ok)
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.Equal(t, store.ReviewStatusCompleted, record.Review.Status)
	require.Equal(t, store.ReviewAssessment{Outcome: "allow"}, *record.Review.Assessment)
	require.NotNil(t, record.PromptTokens)
	require.Equal(t, 4843, *record.PromptTokens)
}
