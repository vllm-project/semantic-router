package config

import "fmt"

func safetyRuleForBinding(rules []SafetyRule, name string) (*SafetyRule, bool) {
	var found *SafetyRule
	var hazard bool
	for i := range rules {
		if name == "safety."+rules[i].Name {
			if found != nil {
				return nil, false
			}
			found, hazard = &rules[i], false
		}
		if name == "safety."+rules[i].Name+".hazard" && rules[i].Hazard != nil {
			if found != nil {
				return nil, false
			}
			found, hazard = &rules[i], true
		}
	}
	return found, hazard
}

func validateSafetyModelBinding(rules []SafetyRule, name string, decl ModelBinding, deployment ModelDeployment) error {
	rule, hazard := safetyRuleForBinding(rules, name)
	if rule == nil {
		return fmt.Errorf("safety binding must identify exactly one recipe signal head")
	}
	if decl.MappingPath != "" {
		return fmt.Errorf("safety head labels are declared by the signal; mapping_path is not supported")
	}
	want := RemoteClassifierContractLabelDistribution
	if hazard {
		want = RemoteClassifierContractLabelScores
	}
	if decl.Contract != want {
		return fmt.Errorf("safety head requires %s", want)
	}
	if deployment.Provider == "http" && decl.Adapter != RemoteClassifierProtocolHTTPClassify {
		return fmt.Errorf("safety HTTP head requires http_classify")
	}
	return nil
}
