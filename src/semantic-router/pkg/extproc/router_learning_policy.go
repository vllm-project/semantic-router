package extproc

import "strings"

const routerLearningPolicyName = "router_learning"

type routerLearningPolicy struct {
	Method  routerLearningMethod
	Mode    string
	Scope   string
	Action  routerLearningAction
	Reason  string
	Details routerLearningPolicyDetails
}

type routerLearningPolicies struct {
	Adaptation routerLearningPolicy
	Protection routerLearningPolicy
}

func newRouterLearningPolicy(method routerLearningMethod) routerLearningPolicy {
	return routerLearningPolicy{
		Method: method,
	}
}

func (p routerLearningPolicies) Empty() bool {
	return p.Adaptation.Empty() && p.Protection.Empty()
}

func (p *routerLearningPolicies) Set(policy routerLearningPolicy) {
	if p == nil || policy.Empty() {
		return
	}
	switch policy.Method {
	case routerLearningMethodAdaptation:
		p.Adaptation = policy
	case routerLearningMethodProtection, "":
		p.Protection = policy
	}
}

func (p routerLearningPolicies) Policy(method routerLearningMethod) (routerLearningPolicy, bool) {
	switch method {
	case routerLearningMethodAdaptation:
		if !p.Adaptation.Empty() {
			return p.Adaptation, true
		}
	case routerLearningMethodProtection:
		if !p.Protection.Empty() {
			return p.Protection, true
		}
	}
	return routerLearningPolicy{}, false
}

func (p routerLearningPolicy) Empty() bool {
	return p.Method == "" &&
		p.Mode == "" &&
		p.Scope == "" &&
		p.Action == "" &&
		p.Reason == "" &&
		p.Details.Empty()
}

func (p routerLearningPolicy) ToMap() map[string]interface{} {
	if p.Empty() {
		return nil
	}
	out := p.Details.ToMap()
	if out == nil {
		out = map[string]interface{}{}
	}
	setLearningPolicyString(out, learningPolicyFieldLearning, routerLearningPolicyName)
	if p.Method != "" {
		setLearningPolicyString(out, learningPolicyFieldMethod, string(p.Method))
	}
	if strings.TrimSpace(p.Mode) != "" {
		setLearningPolicyString(out, learningPolicyFieldMode, p.Mode)
	}
	if strings.TrimSpace(p.Scope) != "" {
		setLearningPolicyString(out, learningPolicyFieldScope, p.Scope)
	}
	if p.Action != "" {
		setLearningPolicyString(out, learningPolicyFieldAction, string(p.Action))
	}
	if strings.TrimSpace(p.Reason) != "" {
		setLearningPolicyString(out, learningPolicyFieldReason, p.Reason)
	}
	return out
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
	case learningPolicyFieldMethod:
		return string(p.Method)
	case learningPolicyFieldMode:
		return strings.TrimSpace(p.Mode)
	case learningPolicyFieldScope:
		return strings.TrimSpace(p.Scope)
	case learningPolicyFieldAction:
		return string(p.Action)
	case learningPolicyFieldReason:
		return strings.TrimSpace(p.Reason)
	default:
		return p.Details.StringField(field)
	}
	return ""
}

func (p routerLearningPolicy) Bool(key string) bool {
	return p.BoolField(routerLearningPolicyField(key))
}

func (p routerLearningPolicy) BoolField(field routerLearningPolicyField) bool {
	return p.Details.BoolField(field)
}

func (p routerLearningPolicy) SessionPhase() string {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(string(trace.Phase))
	}
	return p.StringField(learningPolicyFieldPhase)
}

func setLearningPolicyString(out map[string]interface{}, field routerLearningPolicyField, value string) {
	if out == nil || field == "" || strings.TrimSpace(value) == "" {
		return
	}
	out[string(field)] = strings.TrimSpace(value)
}

func setLearningPolicyNumber(out map[string]interface{}, field routerLearningPolicyField, value float64) {
	if out == nil || field == "" || value == 0 {
		return
	}
	out[string(field)] = value
}

func setLearningPolicyInt(out map[string]interface{}, field routerLearningPolicyField, value int) {
	if out == nil || field == "" || value == 0 {
		return
	}
	out[string(field)] = value
}

func setLearningPolicyValue(out map[string]interface{}, field routerLearningPolicyField, value interface{}) {
	if out == nil || field == "" || value == nil {
		return
	}
	out[string(field)] = value
}

func mergeLearningPolicyMap(out map[string]interface{}, values map[string]interface{}) {
	if out == nil {
		return
	}
	for key, value := range values {
		if strings.TrimSpace(key) == "" || value == nil {
			continue
		}
		out[key] = value
	}
}

func (p routerLearningPolicy) CurrentModel() string {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(trace.CurrentModel)
	}
	return p.StringField(learningPolicyFieldCurrentModel)
}

func (p routerLearningPolicy) BaseSelectedModel() string {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(trace.BaseSelectedModel)
	}
	return p.StringField(learningPolicyFieldBaseSelectedModel)
}

func (p routerLearningPolicy) SelectedModel() string {
	if p.Action == routerLearningActionObserve {
		if finalModel := p.StringField(learningPolicyFieldFinalModel); finalModel != "" {
			return finalModel
		}
	}
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(trace.SelectedModel)
	}
	return p.StringField(learningPolicyFieldSelectedModel)
}

func (p routerLearningPolicy) HardLocked() bool {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return trace.HardLocked
	}
	return p.BoolField(learningPolicyFieldHardLocked)
}

func (p routerLearningPolicy) HardLockReason() string {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(trace.HardLockReason)
	}
	return p.StringField(learningPolicyFieldHardLockReason)
}

func (p routerLearningPolicy) DecisionReason() string {
	if trace := p.Details.ProtectionTrace(); trace != nil {
		return strings.TrimSpace(trace.DecisionReason)
	}
	return p.StringField(learningPolicyFieldDecisionReason)
}

func protectionLearningPolicyForContext(ctx *RequestContext) (routerLearningPolicy, bool) {
	if ctx == nil {
		return routerLearningPolicy{}, false
	}
	if policy, ok := ctx.VSRLearningPolicies.Policy(routerLearningMethodProtection); ok {
		return policy, true
	}
	if ctx.VSRLearningPolicy == nil || ctx.VSRLearningPolicy.Empty() {
		return routerLearningPolicy{}, false
	}
	if ctx.VSRLearningPolicy.Method == "" || ctx.VSRLearningPolicy.Method == routerLearningMethodProtection {
		return *ctx.VSRLearningPolicy, true
	}
	return routerLearningPolicy{}, false
}
