package extproc

import (
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const (
	routerLearningMethodAdaptation routerLearningMethod = "adaptation"
	routerLearningMethodProtection routerLearningMethod = "protection"

	routerLearningActionBypass            routerLearningAction = "bypass"
	routerLearningActionEstablishBaseline routerLearningAction = "establish_baseline"
	routerLearningActionAllowSampling     routerLearningAction = "allow_sampling"
	routerLearningActionSuppressSampling  routerLearningAction = "suppress_sampling"
	routerLearningActionKeepBase          routerLearningAction = "keep_base"
	routerLearningActionProposeSwitch     routerLearningAction = "propose_switch"
	routerLearningActionObserve           routerLearningAction = "observe"
	routerLearningActionHoldCurrent       routerLearningAction = "hold_current"
	routerLearningActionAllowSwitch       routerLearningAction = "allow_switch"
	routerLearningActionRescueSwitch      routerLearningAction = "rescue_switch"
)

type routerLearningMethod string

type routerLearningAction string

type routerLearningInput struct {
	selCtx           *selection.SelectionContext
	baseResult       *selection.SelectionResult
	selectedModelRef *config.ModelRef
	ctx              *RequestContext
}

type routerLearningDecision struct {
	selectionContext *selection.SelectionContext
	selectionResult  *selection.SelectionResult
	selectedModelRef *config.ModelRef
	changesModel     bool
	policy           routerLearningPolicy
}

type routerLearningIdentity struct {
	sessionID          string
	conversationID     string
	memoryKey          string
	scope              string
	sessionHeader      string
	conversationHeader string
}

func firstNonNilSelectionContext(values ...*selection.SelectionContext) *selection.SelectionContext {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstNonNilSelectionResult(values ...*selection.SelectionResult) *selection.SelectionResult {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func firstNonNilModelRef(values ...*config.ModelRef) *config.ModelRef {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func learningChangesModel(baseResult *selection.SelectionResult, result *selection.SelectionResult) bool {
	if baseResult == nil || result == nil {
		return false
	}
	return strings.TrimSpace(baseResult.SelectedModel) != "" &&
		strings.TrimSpace(result.SelectedModel) != "" &&
		baseResult.SelectedModel != result.SelectedModel
}

func selectedModelName(result *selection.SelectionResult) string {
	if result == nil {
		return ""
	}
	return strings.TrimSpace(result.SelectedModel)
}
