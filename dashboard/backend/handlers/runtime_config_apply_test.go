package handlers

import "testing"

func TestValidateManagedContainerExecArgsAllowsVirtualenvPython(t *testing.T) {
	err := validateManagedContainerExecArgs([]string{
		"/opt/vllm-sr-dashboard-venv/bin/python3",
		"-c",
		"print('ok')",
	})
	if err != nil {
		t.Fatalf("expected virtualenv python command to be allowed, got %v", err)
	}
}

func TestIsPythonCommandDetectsAbsolutePythonPath(t *testing.T) {
	if !isPythonCommand("/opt/vllm-sr-dashboard-venv/bin/python3") {
		t.Fatal("expected absolute virtualenv python path to be recognized")
	}
	if isPythonCommand("/bin/sh") {
		t.Fatal("did not expect non-python binary to be recognized")
	}
}
