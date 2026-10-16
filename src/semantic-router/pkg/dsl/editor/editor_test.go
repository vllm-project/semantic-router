package editor

import "testing"

func TestEmptyDiagnosticHelpers(t *testing.T) {
	if len(convertDiagnostics(nil)) != 0 {
		t.Fatal("unexpected diagnostics")
	}
	if joinErrors(nil) != "[]" {
		t.Fatal("unexpected errors")
	}
}

func TestDecompilePreservesEnvironmentExpressions(t *testing.T) {
	result := Decompile(`version: v0.3
routing:
  decisions:
    - name: direct
      priority: 1
      modelRefs:
        - model: ${EDITOR_MODEL}
`)
	if result.Error != "" {
		t.Fatal(result.Error)
	}
	// Transport must never substitute the Dashboard's environment into user text.
	t.Setenv("EDITOR_MODEL", "private-runtime-value")
	again := Decompile(`version: v0.3
routing:
  decisions:
    - name: direct
      priority: 1
      modelRefs:
        - model: ${EDITOR_MODEL}
`)
	if again != result {
		t.Fatal("server environment changed editor output")
	}
}
