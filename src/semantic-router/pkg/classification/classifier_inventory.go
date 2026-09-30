package classification

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"

// PreparedBindings reports the generation shared by this classifier and its
// sibling recipes. A constructed classifier alone is not readiness evidence.
func (c *Classifier) PreparedBindings() ([]binding.PreparedBinding, bool) {
	if c == nil || c.models == nil || c.models.runtime == nil {
		return nil, false
	}
	return c.models.runtime.PreparedBindings(), true
}
