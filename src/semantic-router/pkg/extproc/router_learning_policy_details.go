package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"

type routerLearningPolicyDetails struct {
	Adaptation *routerLearningAdaptationDiagnostics
	Protection *routerLearningProtectionDiagnostics
}

func (d routerLearningPolicyDetails) Empty() bool {
	return d.Adaptation == nil && d.Protection == nil
}

func (d routerLearningPolicyDetails) ProtectionTrace() *selection.SessionPolicyTrace {
	if d.Protection == nil {
		return nil
	}
	return d.Protection.trace
}

func (d routerLearningPolicyDetails) ToMap() map[string]interface{} {
	out := map[string]interface{}{}
	if d.Adaptation != nil {
		mergeLearningPolicyMap(out, d.Adaptation.toPolicyMap())
	}
	if d.Protection != nil {
		mergeLearningPolicyMap(out, d.Protection.toPolicyMap())
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func (d routerLearningPolicyDetails) StringField(field routerLearningPolicyField) string {
	if d.Adaptation != nil {
		if value := d.Adaptation.stringField(field); value != "" {
			return value
		}
	}
	if d.Protection != nil {
		return d.Protection.stringField(field)
	}
	return ""
}

func (d routerLearningPolicyDetails) BoolField(field routerLearningPolicyField) bool {
	if d.Adaptation != nil && d.Adaptation.boolField(field) {
		return true
	}
	if d.Protection != nil {
		return d.Protection.boolField(field)
	}
	return false
}
