//go:build cgo

package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/compatibility"
)

// Run with CANDLE_MODEL_PATH pointing at the pinned tiny BERT fixture described
// in tools/make/models.mk. Each native qualification gets a fresh process because
// Candle initializes a process-global classifier. No model downloads occur here.
func TestNativeCandleCPUCommandRoundTrip(t *testing.T) {
	modelPath := os.Getenv("CANDLE_MODEL_PATH")
	if modelPath == "" {
		t.Skip("set CANDLE_MODEL_PATH to the pinned tiny BERT CPU fixture")
	}
	digest, err := compatibility.DigestLocalCandleArtifact(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	const artifactDigest = "sha256:a873629a9154f8a5f3a1579074638e27108ba71cc05932c29a3710b0e17feb87"
	if digest != artifactDigest {
		t.Fatalf("model is not the pinned tiny BERT fixture: %s", digest)
	}
	runCommand := func(args ...string) ([]byte, error) {
		command := exec.Command(os.Args[0], append([]string{"-test.run=^TestModelcompatCommandProcess$", "--"}, args...)...)
		command.Env = append(os.Environ(), "MODEL_COMPAT_COMMAND_PROCESS=1")
		return command.CombinedOutput()
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	output, err := runCommand("qualify-candle-cpu",
		"--model-path", modelPath,
		"--artifact-revision", "325bf1727142e5f4216ca8e3eef68752321979ac",
		// This is a command integration fixture, not published release evidence.
		"--router-revision", "synthetic-integration-test-working-tree",
		"--labels", "LABEL_0,LABEL_1",
		"--suite", "../../pkg/modelruntime/compatibility/testdata/tiny-random-bert-cpu-suite-v1.json",
		"--output", path)
	if err != nil {
		t.Fatalf("native qualification failed: %v\n%s", err, output)
	}
	receipt, expected := readPassingCanonicalReceipt(t, path)
	if !strings.Contains(string(output), expected) {
		t.Fatalf("qualification did not report complete receipt digest: %s", output)
	}
	if output, err := runCommand("validate", "--expected-digest", expected, path); err != nil {
		t.Fatalf("round-trip validation failed: %v\n%s", err, output)
	}
	receipt.Checks[0].Passed = false
	tampered, err := json.Marshal(receipt)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, tampered, 0o600); err != nil {
		t.Fatal(err)
	}
	if output, err := runCommand("validate", "--expected-digest", expected, path); err == nil || !strings.Contains(string(output), "does not match expected") {
		t.Fatalf("tampered evidence must fail integrity validation: %v\n%s", err, output)
	}
}

func readPassingCanonicalReceipt(t *testing.T, path string) (compatibility.Receipt, string) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := compatibility.ParseReceipt(data)
	if err != nil {
		t.Fatalf("native logs must not contaminate the receipt: %v\n%s", err, data)
	}
	if failed := compatibility.FailedCheckNames(receipt); len(failed) != 0 {
		t.Fatalf("native conformance checks failed: %v", failed)
	}
	canonical, err := receipt.CanonicalJSON()
	if err != nil || !bytes.Equal(data, canonical) {
		t.Fatalf("command output file is not canonical: %v", err)
	}
	expected, err := receipt.Digest()
	if err != nil {
		t.Fatal(err)
	}
	return receipt, expected
}

func TestModelcompatCommandProcess(t *testing.T) {
	if os.Getenv("MODEL_COMPAT_COMMAND_PROCESS") != "1" {
		return
	}
	separator := slices.Index(os.Args, "--")
	if separator < 0 {
		t.Fatal("missing command separator")
	}
	os.Args = append([]string{os.Args[0]}, os.Args[separator+1:]...)
	main()
	os.Exit(0)
}
