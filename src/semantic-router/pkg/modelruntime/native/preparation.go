package native

import (
	"context"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// finishNativeTask publishes an owned task only after its typed warmup succeeds.
// Failed preparation closes the same resource reference.
func finishNativeTask[I, O any](ctx context.Context, spec config.ResolvedModelBinding, task *binding.Task[I, O], capability binding.Capability, resource *binding.Resource, infer func(context.Context, io.Closer, I) (O, error), warmup I) (*binding.Resolved[I, O], error) {
	bound, err := task.Resolve(taskIdentity(spec), capability, resource, infer)
	if err == nil {
		_, err = bound.Call(ctx, string(spec.Recipe), warmup)
	}
	if err != nil {
		_ = resource.Close()
		return nil, err
	}
	bound.Ready()
	return bound, nil
}
