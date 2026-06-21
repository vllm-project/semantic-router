package extproc

import "strings"

const routerLearningPolicyName = "router_learning"

type routerLearningPolicy struct {
	Adaptation routerLearningMethod
	Mode       string
	Scope      string
	Action     routerLearningAction
	Reason     string
	Details    routerLearningPolicyDetails
	Experience routerLearningExperienceDiagnostics
}

func newRouterLearningPolicy(method routerLearningMethod) routerLearningPolicy {
	return routerLearningPolicy{
		Adaptation: method,
	}
}

func (p routerLearningPolicy) Empty() bool {
	return p.Adaptation == "" &&
		p.Mode == "" &&
		p.Scope == "" &&
		p.Action == "" &&
		p.Reason == "" &&
		p.Details.Empty() &&
		p.Experience.Empty()
}

func (p routerLearningPolicy) ToMap() map[string]interface{} {
	if p.Empty() {
		return nil
	}
	fields := newRouterLearningPolicyFields()
	p.Details.appendTo(fields)
	if !p.Experience.Empty() {
		p.Experience.appendLearningPolicyFields(fields)
	}
	fields.SetString(learningPolicyFieldLearning, routerLearningPolicyName)
	if p.Adaptation != "" {
		fields.SetString(learningPolicyFieldAdaptation, string(p.Adaptation))
	}
	if strings.TrimSpace(p.Mode) != "" {
		fields.SetString(learningPolicyFieldMode, p.Mode)
	}
	if strings.TrimSpace(p.Scope) != "" {
		fields.SetString(learningPolicyFieldScope, p.Scope)
	}
	if p.Action != "" {
		fields.SetString(learningPolicyFieldAction, string(p.Action))
	}
	if strings.TrimSpace(p.Reason) != "" {
		fields.SetString(learningPolicyFieldReason, p.Reason)
	}
	return fields.ToMap()
}

func (p routerLearningPolicy) String(key string) string {
	return p.StringField(routerLearningPolicyField(key))
}

func (p routerLearningPolicy) StringField(field routerLearningPolicyField) string {
	switch field {
	case learningPolicyFieldLearning:
		if !p.Empty() {
			return routerLearningPolicyName
		}
	case learningPolicyFieldAdaptation:
		return string(p.Adaptation)
	case learningPolicyFieldMode:
		return strings.TrimSpace(p.Mode)
	case learningPolicyFieldScope:
		return strings.TrimSpace(p.Scope)
	case learningPolicyFieldAction:
		return string(p.Action)
	case learningPolicyFieldReason:
		return strings.TrimSpace(p.Reason)
	default:
		return newRouterLearningPolicyFieldsFromPolicy(p).String(field)
	}
	return ""
}

func (p routerLearningPolicy) Bool(key string) bool {
	return p.BoolField(routerLearningPolicyField(key))
}

func (p routerLearningPolicy) BoolField(field routerLearningPolicyField) bool {
	return newRouterLearningPolicyFieldsFromPolicy(p).Bool(field)
}

func (p routerLearningPolicy) SessionPhase() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(string(trace.Phase))
	}
	return p.StringField(learningPolicyFieldPhase)
}

func (p routerLearningPolicy) CurrentModel() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(trace.CurrentModel)
	}
	return p.StringField(learningPolicyFieldCurrentModel)
}

func (p routerLearningPolicy) BaseSelectedModel() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(trace.BaseSelectedModel)
	}
	return p.StringField(learningPolicyFieldBaseSelectedModel)
}

func (p routerLearningPolicy) SelectedModel() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(trace.SelectedModel)
	}
	return p.StringField(learningPolicyFieldSelectedModel)
}

func (p routerLearningPolicy) HardLocked() bool {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return trace.HardLocked
	}
	return p.BoolField(learningPolicyFieldHardLocked)
}

func (p routerLearningPolicy) HardLockReason() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(trace.HardLockReason)
	}
	return p.StringField(learningPolicyFieldHardLockReason)
}

func (p routerLearningPolicy) DecisionReason() string {
	if trace := p.Details.SessionAwareTrace(); trace != nil {
		return strings.TrimSpace(trace.DecisionReason)
	}
	return p.StringField(learningPolicyFieldDecisionReason)
}

func sessionAwareLearningPolicyForContext(ctx *RequestContext) (routerLearningPolicy, bool) {
	if ctx == nil {
		return routerLearningPolicy{}, false
	}
	if len(ctx.VSRLearningPolicies) > 0 {
		if policy, ok := ctx.VSRLearningPolicies[routerLearningMethodSessionAware]; ok && !policy.Empty() {
			return policy, true
		}
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return routerLearningPolicy{}, false
	}
	if ctx.VSRLearningPolicy.Adaptation == "" || ctx.VSRLearningPolicy.Adaptation == routerLearningMethodSessionAware {
		return *ctx.VSRLearningPolicy, true
	}
	return routerLearningPolicy{}, false
}
