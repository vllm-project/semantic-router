package evaluationplane

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestCommandProcessSharesTheProtectedServiceDiagnosticSink(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write config: %v", err)
	}
	var diagnostics bytes.Buffer
	service, err := NewService(Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		CodeRevision: testSourceRevision, DiagnosticSink: &diagnostics,
	})
	if err != nil {
		t.Fatalf("NewService: %v", err)
	}
	t.Cleanup(func() { _ = service.Close() })

	process, ok := service.process.(*CommandProcess)
	if !ok {
		t.Fatalf("service process type = %T, want *CommandProcess", service.process)
	}
	if process.diagnosticSink != &diagnostics {
		t.Fatal("worker diagnostics are not routed to the protected service sink")
	}
}
