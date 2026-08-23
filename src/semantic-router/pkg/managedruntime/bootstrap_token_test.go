package managedruntime

import (
	"os"
	"path/filepath"
	"testing"
)

func TestReadBootstrapTokenTracksFileFinalization(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router-token")
	if err := os.WriteFile(path, []byte("bootstrap-token-that-is-at-least-thirty-two-bytes\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	token, present, err := readBootstrapToken(path, "")
	if err != nil || string(token) != "bootstrap-token-that-is-at-least-thirty-two-bytes" {
		t.Fatalf("read bootstrap token = %q, %v", token, err)
	}
	if available, probeErr := present(); probeErr != nil || !available {
		t.Fatalf("initial presence = %v, %v", available, probeErr)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	if available, probeErr := present(); probeErr != nil || available {
		t.Fatalf("final presence = %v, %v", available, probeErr)
	}
}

func TestReadBootstrapTokenAllowsFinalizedFileSource(t *testing.T) {
	path := filepath.Join(t.TempDir(), "router-token")
	token, present, err := readBootstrapToken(path, "")
	if err != nil || token != nil {
		t.Fatalf("finalized bootstrap token = %q, %v", token, err)
	}
	if available, probeErr := present(); probeErr != nil || available {
		t.Fatalf("finalized presence = %v, %v", available, probeErr)
	}
}

func TestReadBootstrapTokenEnvironmentRequiresRollout(t *testing.T) {
	const name = "VLLM_SR_TEST_BOOTSTRAP_TOKEN"
	t.Setenv(name, "bootstrap-token-that-is-at-least-thirty-two-bytes")
	token, present, err := readBootstrapToken("", name)
	if err != nil || len(token) == 0 {
		t.Fatalf("environment bootstrap token = %q, %v", token, err)
	}
	if err := os.Unsetenv(name); err != nil {
		t.Fatal(err)
	}
	if available, probeErr := present(); probeErr != nil || !available {
		t.Fatalf("environment source changed without rollout = %v, %v", available, probeErr)
	}
}

func TestReadBootstrapTokenRejectsInsecureOrIndirectFile(t *testing.T) {
	directory := t.TempDir()
	path := filepath.Join(directory, "router-token")
	if err := os.WriteFile(path, []byte("bootstrap-token-that-is-at-least-thirty-two-bytes"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, _, err := readBootstrapToken(path, ""); err == nil {
		t.Fatal("readBootstrapToken() accepted a group/world-readable credential")
	}
	if err := os.Chmod(path, 0o600); err != nil {
		t.Fatal(err)
	}
	link := filepath.Join(directory, "router-token-link")
	if err := os.Symlink(path, link); err != nil {
		t.Fatal(err)
	}
	if _, _, err := readBootstrapToken(link, ""); err == nil {
		t.Fatal("readBootstrapToken() accepted a symbolic link")
	}
}

func TestReadRecoveryTokenRequiresExplicitEnabledAuthority(t *testing.T) {
	const name = "VLLM_SR_TEST_RECOVERY_TOKEN"
	t.Setenv(name, "recovery-token-that-is-at-least-thirty-two-bytes")
	if token, err := readRecoveryToken(false, "", name); err != nil || token != nil {
		t.Fatalf("disabled recovery token = %q, %v", token, err)
	}
	token, err := readRecoveryToken(true, "", name)
	if err != nil || string(token) != "recovery-token-that-is-at-least-thirty-two-bytes" {
		t.Fatalf("enabled recovery token = %q, %v", token, err)
	}
	zero(token)
	t.Setenv(name, "too-short")
	if _, err := readRecoveryToken(true, "", name); err == nil {
		t.Fatal("readRecoveryToken() accepted a short credential")
	}
}
