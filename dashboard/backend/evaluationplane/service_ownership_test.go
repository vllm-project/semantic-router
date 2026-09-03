package evaluationplane

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestEvaluationStoreOwnershipSubprocessHelper(t *testing.T) {
	if os.Getenv("VLLM_SR_EVALUATION_OWNERSHIP_HELPER") != "1" {
		t.Skip("subprocess helper")
	}
	ownership, err := acquireEvaluationStoreOwnership(os.Getenv("VLLM_SR_EVALUATION_OWNERSHIP_ROOT"))
	if err != nil {
		t.Fatalf("acquire subprocess ownership: %v", err)
	}
	defer func() {
		if releaseErr := ownership.release(); releaseErr != nil {
			t.Fatalf("release subprocess ownership: %v", releaseErr)
		}
	}()
	if _, err := fmt.Fprintln(os.Stdout, "locked"); err != nil {
		t.Fatalf("signal subprocess ownership: %v", err)
	}
	if _, err := io.Copy(io.Discard, os.Stdin); err != nil {
		t.Fatalf("wait for subprocess ownership release: %v", err)
	}
}

func TestEvaluationStoreOwnershipRejectsSecondProcessRecoversAfterCrashAndIsReentrantInProcess(t *testing.T) {
	root := filepath.Join(t.TempDir(), "evaluation")
	if err := os.Mkdir(root, 0o700); err != nil {
		t.Fatalf("create evaluation root: %v", err)
	}
	configPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(configPath, []byte("version: v0.3\nrouting:\n  modelCards: []\n"), 0o600); err != nil {
		t.Fatalf("write evaluation config: %v", err)
	}
	//nolint:gosec // G204: the command is Go's current test binary with a compile-time test selector.
	command := exec.Command(os.Args[0], "-test.run=^TestEvaluationStoreOwnershipSubprocessHelper$")
	command.Env = append(os.Environ(),
		"VLLM_SR_EVALUATION_OWNERSHIP_HELPER=1",
		"VLLM_SR_EVALUATION_OWNERSHIP_ROOT="+root,
	)
	stdin, stdinErr := command.StdinPipe()
	if stdinErr != nil {
		t.Fatalf("open subprocess stdin: %v", stdinErr)
	}
	stdout, stdoutErr := command.StdoutPipe()
	if stdoutErr != nil {
		t.Fatalf("open subprocess stdout: %v", stdoutErr)
	}
	if err := command.Start(); err != nil {
		t.Fatalf("start ownership subprocess: %v", err)
	}
	if line := bufio.NewScanner(stdout); !line.Scan() || line.Text() != "locked" {
		_ = stdin.Close()
		_ = command.Wait()
		t.Fatalf("subprocess did not acquire evaluation store ownership")
	}
	options := Options{
		DataDir: root, PythonPath: "python3", ConfigPath: configPath,
		CodeRevision: testSourceRevision, MaxConcurrent: 1, Process: &controlledProcess{},
	}
	if _, err := NewService(options); !errors.Is(err, ErrConflict) {
		_ = stdin.Close()
		_ = command.Wait()
		t.Fatalf("second-process ownership error=%v, want ErrConflict", err)
	}
	if err := command.Process.Kill(); err != nil {
		t.Fatalf("crash ownership subprocess: %v", err)
	}
	if err := stdin.Close(); err != nil {
		t.Fatalf("close crashed subprocess stdin: %v", err)
	}
	if err := command.Wait(); err != nil {
		var exitError *exec.ExitError
		if !errors.As(err, &exitError) {
			t.Fatalf("wait crashed ownership subprocess: %v", err)
		}
	}
	first, firstErr := NewService(options)
	if firstErr != nil {
		t.Fatalf("open owner after subprocess release: %v", firstErr)
	}
	t.Cleanup(func() { _ = first.Close() })
	second, secondErr := NewService(options)
	if secondErr != nil {
		t.Fatalf("same-process second service: %v", secondErr)
	}
	if err := second.Close(); err != nil {
		t.Fatalf("close same-process second service: %v", err)
	}
}
