package extproc

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// maxAgentKindLength bounds the subagent kind a harness names.
const maxAgentKindLength = 64

// replayRunParent is the client request a Looper hop is made for: its run
// and its request id. A hop carries none of the client's headers, so its
// record takes them from the request instead of reading a session of its own.
type replayRunParent struct {
	sessionID string
	agent     *store.AgentIdentity
	requestID string
}

type replayRunParentKey struct{}

// withReplayRunParent marks the context Looper hops run under with the run
// of the client request reqCtx.
func (r *OpenAIRouter) withReplayRunParent(ctx context.Context, reqCtx *RequestContext) context.Context {
	// The hop's record is filed where the request's own record is.
	record := store.Record{SessionID: reqCtx.SessionID}
	r.populateReplayIdentity(&record, reqCtx)
	return context.WithValue(ctx, replayRunParentKey{}, replayRunParent{
		sessionID: record.SessionID, agent: record.Agent, requestID: reqCtx.RequestID,
	})
}

func replayRunParentFrom(ctx context.Context) *replayRunParent {
	if ctx == nil {
		return nil
	}
	if parent, ok := ctx.Value(replayRunParentKey{}).(replayRunParent); ok {
		return &parent
	}
	return nil
}

// applyReplayRunParent files a Looper hop's record in its client request's
// run, under the agent that made the request.
func applyReplayRunParent(record *store.Record, parent *replayRunParent) {
	if parent == nil {
		return
	}
	record.SessionID, record.Agent, record.ParentRequestID = parent.sessionID, parent.agent, parent.requestID
}

// harnessRunSession is the session the harness names for a request's agent
// run: the one protection reads, else the one a supported harness sends,
// pinned or Codex's session-id. A run keeps it whichever identity headers
// protection is configured with and whether protection can key the call.
func harnessRunSession(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) string {
	if sessionID, _ := protectionSessionID(ctx, cfg); sessionID != "" {
		return sessionID
	}
	sessionID, _ := protectionSessionID(ctx, config.RouterLearningProtectionConfig{})
	return sessionID
}

// harnessRunAgent is the agent of its run a request comes from, as the
// harness names it: protection's agent, else the one a supported harness
// sends. Codex names the thread that spawned a thread; a thread its root
// spawned, and every Claude Code subagent, hangs off the main agent.
func harnessRunAgent(ctx *RequestContext, cfg config.RouterLearningProtectionConfig) *store.AgentIdentity {
	agentID, _ := protectionAgentID(ctx, cfg)
	if agentID == "" {
		agentID, _ = protectionAgentID(ctx, config.RouterLearningProtectionConfig{})
	}
	identity := store.AgentIdentity{
		Harness:  requestHarness(ctx),
		ID:       shortLearningIdentityHash(agentID),
		ParentID: shortLearningIdentityHash(codexParentThread(ctx)),
	}
	if identity.ID != "" {
		identity.Kind = boundedIdentityValue(headerValueCI(ctx, headers.XOpenAISubagent), maxAgentKindLength)
	}
	if identity == (store.AgentIdentity{}) {
		return nil
	}
	return &identity
}

// requestTraceRun is the run a request's spans join when its harness sent
// no trace context: the session and agent its record is filed under, read
// from the headers alone.
func (r *OpenAIRouter) requestTraceRun(ctx *RequestContext) tracing.Run {
	cfg := config.RouterLearningProtectionConfig{}
	if r.Config != nil {
		cfg = r.Config.RouterLearning.Protection
	}
	run := tracing.Run{SessionID: harnessRunSession(ctx, cfg)}
	if agent := harnessRunAgent(ctx, cfg); agent != nil {
		run.Harness, run.AgentID, run.ParentAgentID, run.AgentKind = agent.Harness, agent.ID, agent.ParentID, agent.Kind
	}
	return run
}

// codexParentThread is the thread that spawned a Codex thread, empty when
// the root thread, whose id is the session's, spawned it.
func codexParentThread(ctx *RequestContext) string {
	parent := strings.TrimSpace(headerValueCI(ctx, headers.XCodexParentThreadID))
	session := strings.TrimSpace(headerValueCI(ctx, headers.CodexSessionID))
	if parent == "" || parent == session || parent == ctx.AgentTurn.SessionID {
		return ""
	}
	return parent
}

// requestHarness names the supported harness whose identity headers a
// request carries.
func requestHarness(ctx *RequestContext) string {
	switch {
	case headerValueCI(ctx, headers.XClaudeCodeSessionID) != "" || headerValueCI(ctx, headers.XClaudeCodeAgentID) != "":
		return store.HarnessClaudeCode
	case headerValueCI(ctx, headers.CodexSessionID) != "" || ctx.AgentTurn.SessionID != "" || ctx.AgentTurn.ThreadID != "":
		return store.HarnessCodex
	default:
		return ""
	}
}

func boundedIdentityValue(value string, limit int) string {
	value = strings.TrimSpace(value)
	if len(value) > limit {
		return ""
	}
	return value
}
