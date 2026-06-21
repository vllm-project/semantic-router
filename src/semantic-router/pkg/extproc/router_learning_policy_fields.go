package extproc

import "strings"

const (
	learningPolicyFieldLearning      routerLearningPolicyField = "learning"
	learningPolicyFieldAdaptation    routerLearningPolicyField = "adaptation"
	learningPolicyFieldMode          routerLearningPolicyField = "mode"
	learningPolicyFieldScope         routerLearningPolicyField = "scope"
	learningPolicyFieldAction        routerLearningPolicyField = "action"
	learningPolicyFieldReason        routerLearningPolicyField = "reason"
	learningPolicyFieldExperience    routerLearningPolicyField = "experience"
	learningPolicyFieldIdentity      routerLearningPolicyField = "identity"
	learningPolicyFieldAlgorithm     routerLearningPolicyField = "algorithm"
	learningPolicyFieldStateKey      routerLearningPolicyField = "state_key_hash"
	learningPolicyFieldGoals         routerLearningPolicyField = "goals"
	learningPolicyFieldInitialRating routerLearningPolicyField = "initial_rating"
	learningPolicyFieldKFactor       routerLearningPolicyField = "k_factor"
	learningPolicyFieldScores        routerLearningPolicyField = "scores"
	learningPolicyFieldRatings       routerLearningPolicyField = "ratings"
	learningPolicyFieldPreferences   routerLearningPolicyField = "preferences"

	learningPolicyFieldPhase             routerLearningPolicyField = "phase"
	learningPolicyFieldCurrentModel      routerLearningPolicyField = "current_model"
	learningPolicyFieldBaseSelectedModel routerLearningPolicyField = "base_selected_model"
	learningPolicyFieldSelectedModel     routerLearningPolicyField = "selected_model"
	learningPolicyFieldSelectedScore     routerLearningPolicyField = "selected_score"
	learningPolicyFieldSelectedRating    routerLearningPolicyField = "selected_rating"
	learningPolicyFieldSelectedPref      routerLearningPolicyField = "selected_preference"
	learningPolicyFieldHardLocked        routerLearningPolicyField = "hard_locked"
	learningPolicyFieldHardLockReason    routerLearningPolicyField = "hard_lock_reason"
	learningPolicyFieldDecisionReason    routerLearningPolicyField = "decision_reason"
	learningPolicyFieldUserHash          routerLearningPolicyField = "user_hash"
)

type routerLearningPolicyField string

type routerLearningPolicyFields struct {
	values map[routerLearningPolicyField]interface{}
}

func newRouterLearningPolicyFields() *routerLearningPolicyFields {
	return &routerLearningPolicyFields{values: map[routerLearningPolicyField]interface{}{}}
}

func newRouterLearningPolicyFieldsFromPolicy(policy routerLearningPolicy) *routerLearningPolicyFields {
	fields := newRouterLearningPolicyFields()
	policy.Details.appendTo(fields)
	if !policy.Experience.Empty() {
		policy.Experience.appendLearningPolicyFields(fields)
	}
	return fields
}

func (f *routerLearningPolicyFields) Set(field routerLearningPolicyField, value interface{}) {
	if f == nil || field == "" || value == nil {
		return
	}
	f.values[field] = value
}

func (f *routerLearningPolicyFields) SetString(field routerLearningPolicyField, value string) {
	if strings.TrimSpace(value) == "" {
		return
	}
	f.Set(field, strings.TrimSpace(value))
}

func (f *routerLearningPolicyFields) SetBool(field routerLearningPolicyField, value bool) {
	f.Set(field, value)
}

func (f *routerLearningPolicyFields) SetNumber(field routerLearningPolicyField, value interface{}) {
	f.Set(field, value)
}

func (f *routerLearningPolicyFields) String(field routerLearningPolicyField) string {
	if f == nil {
		return ""
	}
	value, ok := f.values[field]
	if !ok {
		return ""
	}
	typed, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(typed)
}

func (f *routerLearningPolicyFields) Bool(field routerLearningPolicyField) bool {
	if f == nil {
		return false
	}
	value, ok := f.values[field]
	if !ok {
		return false
	}
	typed, ok := value.(bool)
	return ok && typed
}

func (f *routerLearningPolicyFields) ToMap() map[string]interface{} {
	if f == nil || len(f.values) == 0 {
		return nil
	}
	result := make(map[string]interface{}, len(f.values))
	for field, value := range f.values {
		result[string(field)] = value
	}
	return result
}
