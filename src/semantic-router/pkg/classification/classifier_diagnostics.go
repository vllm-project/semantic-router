package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/native"

// ModelDiagnosticRuntime exposes already-prepared tasks to the management
// service. The caller must hold the classifier generation's lifetime lease.
func (c *Classifier) ModelDiagnosticRuntime() *native.Runtime {
	if c == nil || c.models == nil {
		return nil
	}
	return c.models.runtime
}
