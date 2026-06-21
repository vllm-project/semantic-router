package extproc

import "strings"

const routerLearningPolicyName = "router_learning"

type routerLearningPolicy struct {
	Adaptation routerLearningMethod
	Mode       string
	Scope      string
	Action     routerLearningAction
	Reason     string
	Fields     map[string]interface{}
}

func newRouterLearningPolicy(method routerLearningMethod) routerLearningPolicy {
	return routerLearningPolicy{
		Adaptation: method,
		Fields:     map[string]interface{}{},
	}
}

func routerLearningPolicyFromMap(
	method routerLearningMethod,
	mode string,
	scope string,
	fields map[string]interface{},
) routerLearningPolicy {
	policy := newRouterLearningPolicy(method)
	if fields != nil {
		policy.Fields = cloneReplayInterfaceMap(fields)
	}
	policy.Mode = firstNonEmpty(replayPolicyString(policy.Fields, "mode"), mode)
	policy.Scope = firstNonEmpty(replayPolicyString(policy.Fields, "scope"), scope)
	policy.Action = routerLearningAction(replayPolicyString(policy.Fields, "action"))
	policy.Reason = replayPolicyString(policy.Fields, "reason")
	delete(policy.Fields, "learning")
	delete(policy.Fields, "adaptation")
	delete(policy.Fields, "mode")
	delete(policy.Fields, "scope")
	delete(policy.Fields, "action")
	delete(policy.Fields, "reason")
	return policy
}

func (p routerLearningPolicy) Empty() bool {
	return p.Adaptation == "" && p.Mode == "" && p.Scope == "" && p.Action == "" && p.Reason == "" && len(p.Fields) == 0
}

func (p routerLearningPolicy) ToMap() map[string]interface{} {
	if p.Empty() {
		return nil
	}
	result := cloneReplayInterfaceMap(p.Fields)
	if result == nil {
		result = map[string]interface{}{}
	}
	result["learning"] = routerLearningPolicyName
	if p.Adaptation != "" {
		result["adaptation"] = string(p.Adaptation)
	}
	if strings.TrimSpace(p.Mode) != "" {
		result["mode"] = p.Mode
	}
	if strings.TrimSpace(p.Scope) != "" {
		result["scope"] = p.Scope
	}
	if p.Action != "" {
		result["action"] = string(p.Action)
	}
	if strings.TrimSpace(p.Reason) != "" {
		result["reason"] = p.Reason
	}
	return result
}

func (p routerLearningPolicy) String(key string) string {
	switch key {
	case "learning":
		if !p.Empty() {
			return routerLearningPolicyName
		}
	case "adaptation":
		return string(p.Adaptation)
	case "mode":
		return strings.TrimSpace(p.Mode)
	case "scope":
		return strings.TrimSpace(p.Scope)
	case "action":
		return string(p.Action)
	case "reason":
		return strings.TrimSpace(p.Reason)
	default:
		return replayPolicyString(p.Fields, key)
	}
	return ""
}

func (p routerLearningPolicy) Bool(key string) bool {
	return replayPolicyBool(p.Fields, key)
}

func (p *routerLearningPolicy) Set(key string, value interface{}) {
	if p == nil || strings.TrimSpace(key) == "" {
		return
	}
	if p.Fields == nil {
		p.Fields = map[string]interface{}{}
	}
	p.Fields[key] = value
}
