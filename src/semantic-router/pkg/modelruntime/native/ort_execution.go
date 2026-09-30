package native

import (
	"fmt"

	ort "github.com/vllm-project/semantic-router/onnx-binding/instance"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Explicit GPU input budgets become concrete session shapes before ORT
// partitions the graph. CPU deployments retain dynamic short inputs. Adapters
// select this policy before the pool hashes the complete execution options.
func prepareFixedGPUExecution(options ort.Options) (ort.Options, error) {
	if options.Provider == "cpu" {
		return options, nil
	}
	if options.MaxInputTokens < 1 {
		return options, fmt.Errorf("%w: GPU deployment requires explicit input.max_tokens for its fixed execution budget", binding.ErrCapability)
	}
	options.ExecutionMaxInputTokens = options.MaxInputTokens
	return options, nil
}
