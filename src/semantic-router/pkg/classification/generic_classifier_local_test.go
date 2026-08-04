package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLocalClassifierRejectsHotReloadWithDifferentModel(t *testing.T) {
	localClassifierLifecycle.Lock()
	previous := localClassifierLifecycle.signature
	localClassifierLifecycle.signature = "existing"
	localClassifierLifecycle.Unlock()
	t.Cleanup(func() {
		localClassifierLifecycle.Lock()
		localClassifierLifecycle.signature = previous
		localClassifierLifecycle.Unlock()
	})

	_, err := newLocalLabelClassifier(config.ClassifierSignalRule{
		Name:      "risk",
		Type:      "local",
		ModelPath: "models/new-risk",
		Labels:    []string{"SAFE", "RISKY"},
	})
	if err == nil || !strings.Contains(err.Error(), "restart the router") {
		t.Fatalf("newLocalLabelClassifier() error = %v, want restart requirement", err)
	}
}
