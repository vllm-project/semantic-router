package classification

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLocalClassifierLoadFailureDoesNotReserveGlobalSlot(t *testing.T) {
	for _, name := range []string{"first", "second"} {
		_, err := newLocalLabelClassifier(config.ClassifierSignalRule{Name: name, Type: "local", ModelPath: filepath.Join(t.TempDir(), name), UseCPU: true, Labels: []string{"SAFE", "RISKY"}})
		if err == nil || strings.Contains(err.Error(), "restart the router") {
			t.Fatalf("failed candidate should report its own load failure: %v", err)
		}
	}
}
