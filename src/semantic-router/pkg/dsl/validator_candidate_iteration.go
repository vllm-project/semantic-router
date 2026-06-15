package dsl

import "fmt"

func addRouteModelsToSymbolSet(modelSet map[string]bool, route *RouteDecl) {
	for _, m := range route.Models {
		if m.Model != "" {
			modelSet[m.Model] = true
		}
	}
	for _, iter := range route.CandidateIterations {
		for _, m := range iter.Models {
			if m.Model != "" {
				modelSet[m.Model] = true
			}
		}
	}
}

func (v *Validator) checkCandidateIterationReferences(route *RouteDecl, iter *CandidateIterationDecl) {
	if iter == nil {
		return
	}
	for _, mr := range iter.Models {
		if mr.Model == "" || v.modelNames[mr.Model] {
			v.checkRouteLoRAReference(route, mr)
			continue
		}
		v.addDiag(DiagWarning, mr.Pos,
			fmt.Sprintf("Model %q is not declared in the top-level model catalog", mr.Model),
			v.suggestModel(mr.Model),
		)
	}
}

func routeHasModelCandidates(route *RouteDecl) bool {
	if len(route.Models) > 0 {
		return true
	}
	for _, iter := range route.CandidateIterations {
		if iter.Source == "models" && len(iter.Models) > 0 && candidateIterationDeclEmitsVariableModel(iter) {
			return true
		}
	}
	return false
}

func (v *Validator) checkCandidateIterationConstraints(route *RouteDecl, iter *CandidateIterationDecl, routeContext string) {
	if iter == nil {
		return
	}
	context := fmt.Sprintf("%s FOR %s", routeContext, iter.Variable)
	v.checkCandidateIterationVariable(iter, context)
	v.checkCandidateIterationSource(route, iter, context)
	v.checkCandidateIterationOutputs(iter, context)
}

func (v *Validator) checkCandidateIterationVariable(iter *CandidateIterationDecl, context string) {
	if iter.Variable == "" {
		v.addDiag(DiagConstraint, iter.Pos,
			fmt.Sprintf("%s: iterator variable is required", context),
			nil,
		)
	}
}

func (v *Validator) checkCandidateIterationSource(route *RouteDecl, iter *CandidateIterationDecl, context string) {
	switch iter.Source {
	case "decision.candidates":
		if len(route.Models) == 0 {
			v.addDiag(DiagConstraint, iter.Pos,
				fmt.Sprintf("%s: decision.candidates requires MODEL candidates in the same route", context),
				nil,
			)
		}
	case "models":
		if len(iter.Models) == 0 {
			v.addDiag(DiagConstraint, iter.Pos,
				fmt.Sprintf("%s: explicit model source must contain at least one model", context),
				nil,
			)
		}
	default:
		v.addDiag(DiagConstraint, iter.Pos,
			fmt.Sprintf("%s: unsupported candidate iteration source %q. Supported sources: decision.candidates or an explicit model list", context, iter.Source),
			nil,
		)
	}
}

func (v *Validator) checkCandidateIterationOutputs(iter *CandidateIterationDecl, context string) {
	for _, output := range iter.Outputs {
		if output.Type != "model" {
			v.addDiag(DiagConstraint, output.Pos,
				fmt.Sprintf("%s: unsupported candidate iteration output %q. Supported output: MODEL %s", context, output.Type, iter.Variable),
				nil,
			)
			continue
		}
		if output.Value != iter.Variable {
			v.addDiag(DiagConstraint, output.Pos,
				fmt.Sprintf("%s: MODEL output must reference iterator variable %q, got %q", context, iter.Variable, output.Value),
				nil,
			)
		}
	}
}

func candidateIterationDeclEmitsVariableModel(iter *CandidateIterationDecl) bool {
	for _, output := range iter.Outputs {
		if output.Type == "model" && output.Value == iter.Variable {
			return true
		}
	}
	return false
}
