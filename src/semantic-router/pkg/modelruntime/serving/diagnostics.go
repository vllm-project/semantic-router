package serving

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// DiagnosticResult retains the exact prepared identity beside its typed output.
// No endpoint or resource compatibility key is included in this descriptor.
type DiagnosticResult[T any] struct {
	Binding binding.PreparedBinding `json:"binding"`
	Result  T                       `json:"result"`
}
