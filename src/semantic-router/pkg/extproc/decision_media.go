package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

// requestDecisionContext scopes user attachments to request routing and selection.
// Response checks use selectionRequestContext without adding request media.
func requestDecisionContext(ctx *RequestContext) context.Context {
	parent := selectionRequestContext(ctx)
	if ctx == nil {
		return parent
	}
	return modelservice.WithDecisionMedia(parent, ctx.decisionMedia)
}

func (r *OpenAIRouter) prepareRequestDecisionMedia(sources []*decisionmedia.Source) map[string]modelservice.DecisionMedia {
	return modelservice.PrepareDecisionMedia(r.Config.ModelDeployments, sources, func(name string) (modelservice.ModelCard, bool) {
		if r.signals != nil && r.signals.modelLease != nil {
			return r.signals.modelLease.CurrentCard(name)
		}
		card, ok := r.decisionCards[name]
		return card, ok
	})
}
