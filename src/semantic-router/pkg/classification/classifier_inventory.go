package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"

// PreparedBindings reports the generation shared by this classifier and its
// sibling recipes: its model runtime bindings and its embedding bindings. A
// constructed classifier alone is not readiness evidence.
func (c *Classifier) PreparedBindings() ([]binding.PreparedBinding, bool) {
	if c == nil || c.models == nil || c.models.runtime == nil {
		return nil, false
	}
	prepared := c.models.runtime.PreparedBindings()
	if c.models.embeddingRuntime != nil {
		prepared = append(prepared, c.models.embeddingRuntime.PreparedBindings()...)
	}
	return prepared, true
}
