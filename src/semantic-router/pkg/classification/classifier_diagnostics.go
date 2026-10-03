package classification

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
)

// ModelDiagnosticRuntime exposes already-prepared tasks to the management
// service. The caller must hold the classifier generation's lifetime lease.
func (c *Classifier) ModelDiagnosticRuntime() *serving.Runtime {
	if c == nil || c.models == nil {
		return nil
	}
	return c.models.runtime
}

// EmbeddingDiagnosticRuntime exposes the recipe's prepared embedding and
// relevance bindings, which the native runtime still serves.
func (c *Classifier) EmbeddingDiagnosticRuntime() *native.Runtime {
	if c == nil || c.models == nil {
		return nil
	}
	return c.models.embeddingRuntime
}
