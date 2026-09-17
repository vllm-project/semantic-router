package looper

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkflowStateNamespaceIsInjective(t *testing.T) {
	names := []config.RecipeName{
		"tenant/a",
		"tenant_a",
		"tenant?a",
		"tenant-a",
		config.DefaultRecipeName,
	}
	seen := map[string]config.RecipeName{}
	for _, name := range names {
		ns := workflowStateNamespace(name)
		if ns == "" {
			t.Fatalf("namespace for %q is empty", name)
		}
		if previous, ok := seen[ns]; ok {
			t.Fatalf("namespace %q collides for %q and %q", ns, previous, name)
		}
		seen[ns] = name
	}
}
