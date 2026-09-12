package dsl

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// iterEmitsVariable reports whether the iteration has at least one MODEL output
// that references the iterator variable, which is the canonical
// bounded-iteration output contract.
func iterEmitsVariable(iter config.CandidateIterationConfig) bool {
	for _, output := range iter.Outputs {
		if output.Type == "model" && output.Value == iter.Variable {
			return true
		}
	}
	return false
}

func rawToCandidateIteration(r *rawCandidateForDecl) (*CandidateIterationDecl, []error) {
	iter := &CandidateIterationDecl{
		Variable: r.Var,
		Source:   "decision.candidates",
		Pos:      posFromLexer(r.Pos),
	}
	var errs []error
	if r.Source != nil {
		switch {
		case r.Source.Ref != nil:
			iter.Source = unquoteIdent(*r.Source.Ref)
		case r.Source.Models != nil:
			iter.Source = "models"
			for _, model := range r.Source.Models.Models {
				ref, modelErrs := rawToModelRef(model)
				iter.Models = append(iter.Models, ref)
				errs = append(errs, modelErrs...)
			}
		}
	}
	for _, item := range r.Body {
		if item.Model == nil {
			continue
		}
		output, outputErrs := rawToCandidateIterationModelOutput(item)
		iter.Outputs = append(iter.Outputs, output)
		errs = append(errs, outputErrs...)
	}
	return iter, errs
}

func rawToCandidateIterationModelOutput(item *rawCandidateIterationItem) (*CandidateIterationOutputDecl, []error) {
	output := &CandidateIterationOutputDecl{
		Type: "model",
		Pos:  posFromLexer(item.Pos),
	}
	var errs []error
	for _, model := range item.Model.Models {
		ref, modelErrs := rawToModelRef(model)
		output.Models = append(output.Models, ref)
		errs = append(errs, modelErrs...)
		if output.Value == "" {
			output.Value = ref.Model
		}
	}
	return output, errs
}
