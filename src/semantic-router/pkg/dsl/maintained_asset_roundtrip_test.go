package dsl

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestMaintainedAMDConfigHasNoUndefinedComplexitySignals(t *testing.T) {
	assetPath := filepath.Join("..", "..", "..", "..", "deploy", "amd", "config.yaml")
	data, err := os.ReadFile(assetPath)
	if err != nil {
		t.Fatalf("failed to read deploy/amd/config.yaml: %v", err)
	}

	cfg, err := config.ParseYAMLBytes(data)
	if err != nil {
		t.Fatalf("ParseYAMLBytes error: %v", err)
	}

	dslText, err := DecompileRouting(cfg)
	if err != nil {
		t.Fatalf("DecompileRouting error: %v", err)
	}

	diags, errs := Validate(dslText)
	if len(errs) > 0 {
		t.Fatalf("Validate parse errors: %v", errs)
	}

	for _, diag := range diags {
		if diag.Level == DiagWarning && strings.Contains(diag.Message, "Signal 'complexity'(") && strings.Contains(diag.Message, "is not defined") {
			t.Fatalf("unexpected undefined complexity signal warning: %s", diag.Message)
		}
	}
}
