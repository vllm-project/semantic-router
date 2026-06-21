package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

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
	policy.Set("identity", sessionAwareIdentityDiagnostics(
		scope,
		cfg.HeaderName("session"),
		cfg.HeaderName("conversation"),
		strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("session"))),
		strings.TrimSpace(headerValueCI(ctx, cfg.HeaderName("conversation"))),
		"",
	))
	return policy
}

func learningPolicyFromSessionAwareResult(
	result *selection.SelectionResult,
	identity sessionAwareLearningIdentity,
	mode string,
) routerLearningPolicy {
	fields := map[string]interface{}{}
	if result != nil && result.SessionPolicy != nil {
		fields = result.SessionPolicy.ToMap()
	}
	delete(fields, "session_id")
	delete(fields, "user_id")
	annotateSessionAwareLearningAction(fields)
	policy := routerLearningPolicyFromMap(routerLearningMethodSessionAware, mode, identity.scope, fields)
	policy.Set("identity", sessionAwareIdentityDiagnostics(
		identity.scope,
		identity.sessionHeader,
		identity.conversationHeader,
		identity.sessionID,
		identity.conversationID,
		identity.memoryKey,
	))
	return policy
}

func sessionAwareIdentityDiagnostics(
	scope string,
	sessionHeader string,
	conversationHeader string,
	sessionID string,
	conversationID string,
	memoryKey string,
) map[string]interface{} {
	conversationRequired := scope == config.RouterLearningScopeConversation
	identity := map[string]interface{}{
		"scope": scope,
		"headers": map[string]interface{}{
			"session":      sessionHeader,
			"conversation": conversationHeader,
		},
		"session":      sessionAwareIdentityPart(sessionHeader, sessionID, true),
		"conversation": sessionAwareIdentityPart(conversationHeader, conversationID, conversationRequired),
	}
	if memoryKey != "" {
		identity["memory_key_hash"] = shortLearningIdentityHash(memoryKey)
	}
	return identity
}

func sessionAwareIdentityPart(headerName string, value string, required bool) map[string]interface{} {
	status := "not_required"
	if required {
		status = "missing"
	}
	part := map[string]interface{}{
		"source":   "header:" + headerName,
		"required": required,
		"status":   status,
	}
	if strings.TrimSpace(value) != "" {
		part["status"] = "present"
		part["hash"] = shortLearningIdentityHash(value)
	}
	return part
}

func shortLearningIdentityHash(value string) string {
	sum := sha256.Sum256([]byte(value))
	return hex.EncodeToString(sum[:])[:16]
}

func annotateSessionAwareLearningAction(policy map[string]interface{}) {
	if policy == nil {
		return
	}
	if policy["action"] == nil {
		current := replayPolicyString(policy, "current_model")
		selected := replayPolicyString(policy, "selected_model")
		switch {
		case replayPolicyBool(policy, "hard_locked"):
			policy["action"] = string(routerLearningActionHardLock)
		case current == "":
			policy["action"] = string(routerLearningActionSelect)
		case selected == current:
			policy["action"] = string(routerLearningActionStay)
		default:
			policy["action"] = string(routerLearningActionSwitch)
		}
	}
	if policy["reason"] == nil {
		reason := firstNonEmpty(
			replayPolicyString(policy, "hard_lock_reason"),
			replayPolicyString(policy, "decision_reason"),
		)
		if reason != "" {
			policy["reason"] = reason
		}
	}
}
