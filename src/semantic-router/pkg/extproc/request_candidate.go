package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// candidateModelIdentity returns the effective candidate identity for a ModelRef.
// LoRA adapters use LoRAName; bare base models use Model.
func candidateModelIdentity(ref config.ModelRef) string {
	if ref.LoRAName != "" {
		return ref.LoRAName
	}
	return ref.Model
}

// backendModelForCandidate keeps a selected adapter's owner attached to its
// request-local identity. Adapter names are not necessarily unique across
// model cards, so runtime backend metadata must not be recovered from the name.
func (ctx *RequestContext) backendModelForCandidate(model string) string {
	if ctx == nil || ctx.LooperRequest || ctx.VSRSelectedCandidate == nil {
		return model
	}
	ref := ctx.VSRSelectedCandidate
	if ref.LoRAName != "" && ref.LoRAName == model && ref.Model != "" {
		return ref.Model
	}
	return model
}

// decisionForCandidate provides a request-local reasoning view without mutating
// the shared decision or its plugin/candidate configuration. Both standard wire
// codecs and provider extensions must use the same exact post-policy choice.
func (ctx *RequestContext) decisionForCandidate(model string) *config.Decision {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	ref := ctx.VSRSelectedCandidate
	if ref != nil && candidateModelIdentity(*ref) == model {
		decision := *ctx.VSRSelectedDecision
		decision.ModelRefs = []config.ModelRef{*ref}
		return &decision
	}
	for i := range ctx.VSREligibleModelRefs {
		r := &ctx.VSREligibleModelRefs[i]
		if candidateModelIdentity(*r) == model {
			decision := *ctx.VSRSelectedDecision
			decision.ModelRefs = []config.ModelRef{*r}
			return &decision
		}
	}
	for i := range ctx.VSRSelectedDecision.ModelRefs {
		r := &ctx.VSRSelectedDecision.ModelRefs[i]
		if candidateModelIdentity(*r) == model {
			decision := *ctx.VSRSelectedDecision
			decision.ModelRefs = []config.ModelRef{*r}
			return &decision
		}
	}
	return ctx.VSRSelectedDecision
}

// decisionForBackend gives backend reasoning projection the selected base
// model's view of the exact candidate without mutating the shared decision.
func (ctx *RequestContext) decisionForBackend(model string) *config.Decision {
	if ctx != nil && ctx.LooperRequest {
		return ctx.VSRSelectedDecision
	}
	decision := ctx.decisionForCandidate(model)
	if decision == nil || ctx.backendModelForCandidate(model) == model ||
		len(decision.ModelRefs) != 1 || decision.ModelRefs[0].LoRAName == "" {
		return decision
	}
	ref := decision.ModelRefs[0]
	ref.LoRAName = ""
	decision.ModelRefs = []config.ModelRef{ref}
	return decision
}
