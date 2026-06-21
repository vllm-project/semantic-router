package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	routerLearningDefaultSessionHeader      = "x-session-id"
	routerLearningDefaultConversationHeader = "x-conversation-id"
)

func learningStateKeyFromRequest(scope string, input routerLearningInput) (string, bool) {
	decisionName, sessionID, conversationID := learningIdentityPartsFromRequest(input)
	return learningStateKeyFromParts(scope, decisionName, sessionID, conversationID)
}

func learningStateKeyFromParts(scope string, decisionName string, sessionID string, conversationID string) (string, bool) {
	switch strings.TrimSpace(scope) {
	case "", config.RouterLearningScopeDecision:
		decisionName = strings.TrimSpace(decisionName)
		if decisionName == "" {
			decisionName = "_global"
		}
		return "decision:" + decisionName, true
	case config.RouterLearningScopeConversation:
		sessionID = strings.TrimSpace(sessionID)
		conversationID = strings.TrimSpace(conversationID)
		if sessionID == "" || conversationID == "" {
			return "", false
		}
		return "conversation:" + sessionID + "/" + conversationID, true
	case config.RouterLearningScopeSession:
		sessionID = strings.TrimSpace(sessionID)
		if sessionID == "" {
			return "", false
		}
		return "session:" + sessionID, true
	default:
		return "", false
	}
}

func learningIdentityPartsFromRequest(input routerLearningInput) (decisionName string, sessionID string, conversationID string) {
	if input.selCtx != nil {
		decisionName = strings.TrimSpace(input.selCtx.DecisionName)
		sessionID = strings.TrimSpace(input.selCtx.SessionID)
	}
	if input.ctx == nil {
		return decisionName, sessionID, conversationID
	}
	if decisionName == "" {
		decisionName = strings.TrimSpace(input.ctx.VSRSelectedDecisionName)
	}
	if sessionID == "" {
		sessionID = strings.TrimSpace(input.ctx.SessionID)
	}
	if sessionID == "" {
		sessionID = strings.TrimSpace(headerValueCI(input.ctx, routerLearningDefaultSessionHeader))
	}
	conversationID = strings.TrimSpace(input.ctx.VSRLearningConversationID)
	if conversationID == "" {
		conversationID = strings.TrimSpace(headerValueCI(input.ctx, routerLearningDefaultConversationHeader))
	}
	return decisionName, sessionID, conversationID
}

func learningUserIDFromRequest(input routerLearningInput) string {
	if input.selCtx != nil && strings.TrimSpace(input.selCtx.UserID) != "" {
		return strings.TrimSpace(input.selCtx.UserID)
	}
	if input.ctx != nil {
		return strings.TrimSpace(extractUserID(input.ctx))
	}
	return ""
}
