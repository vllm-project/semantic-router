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

func rawToCandidateIteration(r *rawCandidateForDecl) *CandidateIterationDecl {
	iter := &CandidateIterationDecl{
		Variable: r.Var,
		Source:   "decision.candidates",
		Pos:      posFromLexer(r.Pos),
	}
	if r.Source != nil {
		switch {
		case r.Source.Ref != nil:
			iter.Source = unquoteIdent(*r.Source.Ref)
		case r.Source.Models != nil:
			iter.Source = "models"
			for _, model := range r.Source.Models.Models {
				iter.Models = append(iter.Models, rawToModelRef(model))
			}
		}
	}
	for _, item := range r.Body {
		if item.Model == nil {
			continue
		}
		iter.Outputs = append(iter.Outputs, rawToCandidateIterationModelOutput(item))
	}
	return iter
}

func rawToCandidateIterationModelOutput(item *rawCandidateIterationItem) *CandidateIterationOutputDecl {
	output := &CandidateIterationOutputDecl{
		Type: "model",
		Pos:  posFromLexer(item.Pos),
	}
	for _, model := range item.Model.Models {
		ref := rawToModelRef(model)
		output.Models = append(output.Models, ref)
		if output.Value == "" {
			output.Value = ref.Model
		}
	}
	return output
}
