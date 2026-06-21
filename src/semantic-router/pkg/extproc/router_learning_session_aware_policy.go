package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	sessionAwareIdentityStatusMissing     sessionAwareIdentityStatus = "missing"
	sessionAwareIdentityStatusNotRequired sessionAwareIdentityStatus = "not_required"
	sessionAwareIdentityStatusPresent     sessionAwareIdentityStatus = "present"
)

type sessionAwareIdentityStatus string

type sessionAwareLearningDetail struct {
	trace    *selection.SessionPolicyTrace
	identity sessionAwareIdentityDiagnostics
}

type sessionAwareIdentityDiagnostics struct {
	scope         string
	sessionHeader string
	convoHeader   string
	session       sessionAwareIdentityPart
	conversation  sessionAwareIdentityPart
	memoryKeyHash string
}

type sessionAwareIdentityPart struct {
	source   string
	required bool
	status   sessionAwareIdentityStatus
	hash     string
}

func buildSessionAwareLearningPolicy(
	ctx *RequestContext,
	cfg config.SessionAwareLearningConfig,
	mode string,
	action routerLearningAction,
	reason string,
	scope string,
) routerLearningPolicy {
	policy := newRouterLearningPolicy(routerLearningMethodSessionAware)
	policy.Mode = mode
	policy.Action = action
	policy.Reason = reason
	policy.Scope = scope
	policy.Detail = &sessionAwareLearningDetail{
		identity: newSessionAwareIdentityDiagnostics(
			scope,
			cfg.HeaderName("session"),
			cfg.HeaderName("conversation"),
			strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("session"))),
			strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("conversation"))),
			"",
		),
	}
	return policy
}

func learningPolicyFromSessionAwareResult(
	result *selection.SelectionResult,
	identity sessionAwareLearningIdentity,
	mode string,
) routerLearningPolicy {
	trace := sessionAwareTraceFromResult(result)
	policy := newRouterLearningPolicy(routerLearningMethodSessionAware)
	policy.Mode = mode
	policy.Scope = identity.scope
	policy.Action = sessionAwareLearningAction(trace)
	policy.Reason = sessionAwareLearningReason(trace)
	policy.Detail = &sessionAwareLearningDetail{
		trace: trace,
		identity: newSessionAwareIdentityDiagnostics(
			identity.scope,
			identity.sessionHeader,
			identity.conversationHeader,
			identity.sessionID,
			identity.conversationID,
			identity.memoryKey,
		),
	}
	return policy
}

func sessionAwareTraceFromResult(result *selection.SelectionResult) *selection.SessionPolicyTrace {
	if result == nil {
		return nil
	}
	return result.SessionPolicy
}

func sessionAwareLearningAction(trace *selection.SessionPolicyTrace) routerLearningAction {
	if trace == nil {
		return routerLearningActionNoop
	}
	switch {
	case trace.HardLocked:
		return routerLearningActionHardLock
	case strings.TrimSpace(trace.CurrentModel) == "":
		return routerLearningActionSelect
	case trace.SelectedModel == trace.CurrentModel:
		return routerLearningActionStay
	default:
		return routerLearningActionSwitch
	}
}

func sessionAwareLearningReason(trace *selection.SessionPolicyTrace) string {
	if trace == nil {
		return ""
	}
	return firstNonEmpty(trace.HardLockReason, trace.DecisionReason)
}

func (d *sessionAwareLearningDetail) appendLearningPolicyFields(out map[string]interface{}) {
	if d == nil {
		return
	}
	if d.trace != nil {
		for key, value := range d.trace.ToMap() {
			switch key {
			case "session_id", "user_id", "action", "reason", "learning", "adaptation", "mode", "scope":
				continue
			default:
				out[key] = value
			}
		}
	}
	d.identity.appendLearningPolicyFields(out)
}

func newSessionAwareIdentityDiagnostics(
	scope string,
	sessionHeader string,
	conversationHeader string,
	sessionID string,
	conversationID string,
	memoryKey string,
) sessionAwareIdentityDiagnostics {
	conversationRequired := scope == config.RouterLearningScopeConversation
	return sessionAwareIdentityDiagnostics{
		scope:         scope,
		sessionHeader: sessionHeader,
		convoHeader:   conversationHeader,
		session:       newSessionAwareIdentityPart(sessionHeader, sessionID, true),
		conversation:  newSessionAwareIdentityPart(conversationHeader, conversationID, conversationRequired),
		memoryKeyHash: shortLearningIdentityHash(memoryKey),
	}
}

func newSessionAwareIdentityPart(
	headerName string,
	value string,
	required bool,
) sessionAwareIdentityPart {
	part := sessionAwareIdentityPart{
		source:   "header:" + headerName,
		required: required,
		status:   sessionAwareIdentityStatusNotRequired,
	}
	if required {
		part.status = sessionAwareIdentityStatusMissing
	}
	if strings.TrimSpace(value) != "" {
		part.status = sessionAwareIdentityStatusPresent
		part.hash = shortLearningIdentityHash(value)
	}
	return part
}

func (d sessionAwareIdentityDiagnostics) appendLearningPolicyFields(out map[string]interface{}) {
	identity := map[string]interface{}{
		"scope": d.scope,
		"headers": map[string]interface{}{
			"session":      d.sessionHeader,
			"conversation": d.convoHeader,
		},
		"session":      d.session.toMap(),
		"conversation": d.conversation.toMap(),
	}
	if d.memoryKeyHash != "" {
		identity["memory_key_hash"] = d.memoryKeyHash
	}
	out["identity"] = identity
}

func (p sessionAwareIdentityPart) toMap() map[string]interface{} {
	out := map[string]interface{}{
		"source":   p.source,
		"required": p.required,
		"status":   string(p.status),
	}
	if p.hash != "" {
		out["hash"] = p.hash
	}
	return out
}

func shortLearningIdentityHash(value string) string {
	if strings.TrimSpace(value) == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:16]
}
