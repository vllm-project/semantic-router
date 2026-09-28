package native

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Factories own preparation only. The instance pool, admission and task registry
// continue to own lifecycle and execution; factories never create global models.
type embeddingFactory func(*Runtime, context.Context, config.ResolvedModelBinding, embedding.Options) (*preparedEmbedding, error)

var embeddingFactories = map[string]embeddingFactory{
	"candle":   (*Runtime).candleEmbedding,
	"ort":      (*Runtime).ortEmbedding,
	"openvino": (*Runtime).openvinoEmbedding,
}

func (r *Runtime) prepareEmbedding(ctx context.Context, spec config.ResolvedModelBinding, view embedding.Options) (*preparedEmbedding, error) {
	factory, ok := embeddingFactories[spec.Deployment.Provider]
	if !ok {
		return nil, fmt.Errorf("%w: native embedding provider %q", binding.ErrCapability, spec.Deployment.Provider)
	}
	return factory(r, ctx, spec, view)
}
