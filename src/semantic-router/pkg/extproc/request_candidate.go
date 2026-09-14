package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

// decisionForCandidate provides a request-local reasoning view without mutating
// the shared decision or its plugin/candidate configuration. Both standard wire
// codecs and provider extensions must use the same exact post-policy choice.
func (ctx *RequestContext) decisionForCandidate(model string) *config.Decision {
	if ctx == nil || ctx.VSRSelectedDecision == nil {
		return nil
	}
	ref := ctx.VSRSelectedCandidate
	if ref == nil || (ref.Model != model && (ref.LoRAName == "" || ref.LoRAName != model)) {
		return ctx.VSRSelectedDecision
	}
	decision := *ctx.VSRSelectedDecision
	decision.ModelRefs = []config.ModelRef{*ref}
	return &decision
}
