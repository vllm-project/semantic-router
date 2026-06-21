package extproc

import "strings"

const routerLearningPolicyName = "router_learning"

type routerLearningPolicyDetail interface {
	appendLearningPolicyFields(map[string]interface{})
}

type routerLearningPolicy struct {
	Adaptation routerLearningMethod
	Mode       string
	Scope      string
	Action     routerLearningAction
	Reason     string
	Detail     routerLearningPolicyDetail
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
		p.Detail == nil &&
		p.Experience.Empty()
}

func (p routerLearningPolicy) ToMap() map[string]interface{} {
	if p.Empty() {
		return nil
	}
	result := map[string]interface{}{}
	if p.Detail != nil {
		p.Detail.appendLearningPolicyFields(result)
	}
	if !p.Experience.Empty() {
		p.Experience.appendLearningPolicyFields(result)
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
		return replayPolicyString(p.ToMap(), key)
	}
	return ""
}

func (p routerLearningPolicy) Bool(key string) bool {
	return replayPolicyBool(p.ToMap(), key)
}
