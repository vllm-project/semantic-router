//go:build !openvino || windows || !cgo

package native

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func (r *Runtime) openvinoEmbedding(context.Context, config.ResolvedModelBinding, embedding.Options) (*preparedEmbedding, error) {
	return nil, fmt.Errorf("%w: OpenVINO requires an openvino cgo build", binding.ErrCapability)
}

func (r *Runtime) openvinoSequence(context.Context, config.ResolvedModelBinding) (*binding.Resolved[string, tasks.LabelDistribution], error) {
	return nil, fmt.Errorf("%w: OpenVINO requires an openvino cgo build", binding.ErrCapability)
}
