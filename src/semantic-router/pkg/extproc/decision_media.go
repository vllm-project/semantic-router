package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
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

// prepareRequestDecisionMedia uses this generation's observed cards. It never
// discovers a model on the request path, and owns no backend-request mutation.
func (r *OpenAIRouter) prepareRequestDecisionMedia(sources []*decisionmedia.Source) map[string]modelservice.DecisionMedia {
	if len(sources) == 0 {
		return nil
	}
	result := make(map[string]modelservice.DecisionMedia)
	for name, deployment := range r.Config.ModelDeployments {
		if deployment.Input.Media == nil {
			continue
		}
		card, ok := r.decisionCards[name]
		if r.signals != nil && r.signals.modelLease != nil {
			card, ok = r.signals.modelLease.CurrentCard(name)
		}
		if !ok || modelservice.ValidateDecisionMedia(deployment, card) != nil {
			continue
		}
		policy := decisionmedia.Policy{
			Images: decisionMediaLimit(deployment.Input.Media.Images, 4, card.ImageMaxBytes),
			Videos: decisionMediaLimit(deployment.Input.Media.Videos, 1, card.VideoMaxBytes),
		}
		selected := decisionmedia.Select(sources, policy, modelservice.RecordMediaSkip)
		result[name] = modelservice.DecisionMedia{Images: selected.Images, Videos: selected.Videos}
	}
	return result
}

func decisionMediaLimit(limit *config.DecisionMediaLimit, count, bytes int) *decisionmedia.Limit {
	if limit == nil {
		return nil
	}
	if limit.MaxCount > 0 {
		count = limit.MaxCount
	}
	if limit.MaxBytes > 0 {
		bytes = limit.MaxBytes
	}
	return &decisionmedia.Limit{Count: count, Bytes: bytes}
}
