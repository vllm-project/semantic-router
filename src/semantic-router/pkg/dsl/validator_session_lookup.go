package dsl

import (
	"fmt"
	"strings"
)

func (v *Validator) checkSessionMetricRuleConstraints(s *SignalDecl, context string) {
	kind := strings.ToLower(strings.TrimSpace(getStringFieldOrEmpty(s.Fields, "kind")))
	table := strings.TrimSpace(getStringFieldOrEmpty(s.Fields, "table"))
	state := strings.TrimSpace(getStringFieldOrEmpty(s.Fields, "state"))
	if kind == "" {
		if table != "" {
			kind = "lookup"
		} else if state != "" {
			kind = "state"
		}
	}
	switch kind {
	case "state":
		v.checkSessionRuleConstraints(s, context)
	case "lookup":
		v.checkLookupRuleConstraints(s, context)
	default:
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: session_metric requires kind state or lookup, or inferable state/table fields", context), nil)
	}
}

func (v *Validator) sessionStatePaths() map[string]string {
	out := make(map[string]string)
	for _, ss := range v.prog.SessionStates {
		for _, f := range ss.Fields {
			out[ss.Name+"."+f.Name] = f.TypeName
		}
	}
	return out
}

func (v *Validator) checkSessionRuleConstraints(s *SignalDecl, context string) {
	paths := v.sessionStatePaths()
	state, ok := getStringField(s.Fields, "state")
	if !ok || strings.TrimSpace(state) == "" {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: 'state' is required (SESSION_STATE dotted path)", context), nil)
		return
	}
	typ, exists := paths[state]
	if !exists {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: state %q is not declared on any SESSION_STATE", context, state), nil)
		return
	}
	if typ != "float" && typ != "int" {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: state %q must reference a numeric SESSION_STATE field (int or float), got %q", context, state, typ), nil)
	}
	norm := strings.ToLower(strings.TrimSpace(getStringFieldOrEmpty(s.Fields, "normalize")))
	switch norm {
	case "", "identity", "minmax":
	default:
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: unsupported normalize %q (supported: identity, minmax)", context, norm), nil)
	}
	if norm == "minmax" {
		_, hasMin := getFloat64Field(s.Fields, "min")
		_, hasMax := getFloat64Field(s.Fields, "max")
		if !hasMin || !hasMax {
			v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: normalize minmax requires numeric min and max", context), nil)
			return
		}
		minV, _ := getFloat64Field(s.Fields, "min")
		maxV, _ := getFloat64Field(s.Fields, "max")
		if maxV <= minV {
			v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: minmax requires max > min", context), nil)
		}
	}
}

func getStringFieldOrEmpty(fields map[string]Value, key string) string {
	s, _ := getStringField(fields, key)
	return s
}

func (v *Validator) checkLookupRuleConstraints(s *SignalDecl, context string) {
	paths := v.sessionStatePaths()
	table, ok := getStringField(s.Fields, "table")
	if !ok || strings.TrimSpace(table) == "" {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: 'table' is required", context), nil)
	}
	raw, ok := s.Fields["key"]
	if !ok {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: 'key' array is required", context), nil)
		return
	}
	av, ok := raw.(ArrayValue)
	if !ok || len(av.Items) == 0 {
		v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: 'key' must be a non-empty string array", context), nil)
		return
	}
	for _, item := range av.Items {
		sv, ok := item.(StringValue)
		if !ok || strings.TrimSpace(sv.V) == "" {
			v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: lookup key entries must be non-empty strings", context), nil)
			continue
		}
		if _, exists := paths[sv.V]; !exists {
			v.addDiag(DiagConstraint, s.Pos, fmt.Sprintf("%s: lookup key %q is not a declared SESSION_STATE field path", context, sv.V), nil)
		}
	}
}
