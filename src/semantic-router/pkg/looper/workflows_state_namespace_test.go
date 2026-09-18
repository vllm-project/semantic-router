package looper

import (
	"strings"
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

func TestWorkflowStateStoreFileNameKeepsShortNamespacedNames(t *testing.T) {
	namespaced, err := workflowNamespacedStateID(config.DefaultRecipeName, "short-id")
	if err != nil {
		t.Fatal(err)
	}
	got := workflowStateStoreFileName(namespaced)
	if got != namespaced+".json" {
		t.Fatalf("short name hashed unexpectedly: %q", got)
	}
	if !strings.Contains(got, "__") {
		t.Fatalf("namespaced filename %q missing recipe separator", got)
	}
}

func TestWorkflowStateStoreFileNameHashesOverNAMEMAX(t *testing.T) {
	recipe := config.RecipeName(strings.Repeat("a", 170))
	id := strings.Repeat("b", 24)
	namespaced, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		t.Fatal(err)
	}
	plain := namespaced + ".json"
	if len(plain) <= workflowStateFileNameMaxBytes {
		t.Fatalf("fixture too short: %d", len(plain))
	}
	got := workflowStateStoreFileName(namespaced)
	if got == plain {
		t.Fatal("overlong namespaced filename was not hashed")
	}
	if len(got) > workflowStateFileNameMaxBytes {
		t.Fatalf("hashed name length %d exceeds NAME_MAX", len(got))
	}
	if strings.Contains(got, "__") {
		t.Fatalf("hashed name %q collides with namespaced form", got)
	}
	if !strings.HasPrefix(got, "sha256-") || !strings.HasSuffix(got, ".json") {
		t.Fatalf("hashed name = %q", got)
	}
}

func TestWorkflowStateStoreFileNameHashesBeforeTempBasenameExceedsNAMEMAX(t *testing.T) {
	recipe := config.RecipeName(strings.Repeat("a", 150))
	id := strings.Repeat("b", 24)
	namespaced, err := workflowNamespacedStateID(recipe, id)
	if err != nil {
		t.Fatal(err)
	}
	plain := namespaced + ".json"
	if len(plain) > workflowStateFileNameMaxBytes {
		t.Fatalf("final .json basename should still fit NAME_MAX: %d", len(plain))
	}
	tmpBasename := plain + ".tmp-" + strings.Repeat("c", 24)
	if len(tmpBasename) <= workflowStateFileNameMaxBytes {
		t.Fatalf("fixture temp basename does not exceed NAME_MAX: %d", len(tmpBasename))
	}
	got := workflowStateStoreFileName(namespaced)
	if got == plain {
		t.Fatal("namespaced filename was not hashed for temp write budget")
	}
	if len(got)+workflowStateStoreTempSuffixMaxBytes > workflowStateFileNameMaxBytes {
		t.Fatalf("hashed store name %q still exceeds temp basename budget", got)
	}
}

func TestWorkflowStateStoreFileNameTempBasenameBudgetBoundary(t *testing.T) {
	maxUnhashed := workflowStateFileNameMaxBytes - len(".json") - workflowStateStoreTempSuffixMaxBytes

	atLimit := strings.Repeat("a", maxUnhashed)
	if got := workflowStateStoreFileName(atLimit); got != atLimit+".json" {
		t.Fatalf("at-limit name hashed unexpectedly: %q", got)
	}

	over := strings.Repeat("a", maxUnhashed+1)
	if got := workflowStateStoreFileName(over); got == over+".json" {
		t.Fatal("over-limit namespaced filename was not hashed")
	}
}
