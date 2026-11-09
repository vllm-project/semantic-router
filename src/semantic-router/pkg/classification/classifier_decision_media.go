package classification

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/decisionmedia"
)

// RequestDecisionContext attaches media for request signals and selection only.
// Use the original context for response-stage or unrelated judgments.
func (c *Classifier) RequestDecisionContext(ctx context.Context, sources []*decisionmedia.Source) context.Context {
	if c == nil || c.Config == nil || c.models == nil || c.models.runtime == nil {
		return ctx
	}
	return modelservice.WithDecisionMedia(ctx, modelservice.PrepareDecisionMedia(c.Config.ModelDeployments, sources, c.models.runtime.CurrentDeploymentCard))
}
