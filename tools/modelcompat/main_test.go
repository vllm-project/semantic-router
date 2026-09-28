package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/compatibility"
)

type fakeRuntime struct {
	initialized bool
	malformed   bool
}

func (f *fakeRuntime) Initialize(_ string, _ int) error {
	f.initialized = true
	return nil
}

func (f *fakeRuntime) Classify(input string) (compatibility.ClassificationResult, error) {
	if !f.initialized {
		return compatibility.ClassificationResult{}, errors.New("unavailable")
	}
	if input == "oversized" {
		return compatibility.ClassificationResult{}, errors.New("too large")
	}
	if f.malformed && input == "hello" {
		return compatibility.ClassificationResult{Class: 1, Confidence: 0.8}, nil
	}
	return compatibility.ClassificationResult{
		Class:         1,
		Confidence:    0.8,
		Probabilities: []float32{0.2, 0.8},
		NumClasses:    2,
	}, nil
}

func TestRunQualifyCandleCPUWritesPassingReceipt(t *testing.T) {
	artifact := writeArtifact(t)
	suite := writeSuite(t)
	output := filepath.Join(t.TempDir(), "receipt.json")
	runtime := &fakeRuntime{}

	err := run([]string{
		"qualify-candle-cpu",
		"--model-path", artifact,
		"--artifact-revision", "model-commit",
		"--router-revision", "router-commit",
		"--labels", "LABEL_0,LABEL_1",
		"--device-profile", "linux/amd64/cpu",
		"--suite", suite,
		"--output", output,
	}, runtime, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("run() error = %v", err)
	}
	data, err := os.ReadFile(output)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	receipt, err := compatibility.ParseReceipt(data)
	if err != nil {
		t.Fatalf("ParseReceipt() error = %v", err)
	}
	if receipt.Subject.ArtifactRevision != "model-commit" || receipt.Subject.RouterRevision != "router-commit" {
		t.Fatalf("receipt subject = %+v", receipt.Subject)
	}
	if failed := compatibility.FailedCheckNames(receipt); len(failed) != 0 {
		t.Fatalf("failed checks = %v, want none", failed)
	}
}

func TestRunQualifyCandleCPUWritesEvidenceBeforeReturningFailure(t *testing.T) {
	artifact := writeArtifact(t)
	suite := writeSuite(t)
	output := filepath.Join(t.TempDir(), "receipt.json")

	err := run([]string{
		"qualify-candle-cpu",
		"--model-path", artifact,
		"--artifact-revision", "model-commit",
		"--router-revision", "router-commit",
		"--labels", "LABEL_0,LABEL_1",
		"--suite", suite,
		"--output", output,
	}, &fakeRuntime{malformed: true}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), compatibility.CheckLabelParity) {
		t.Fatalf("run() error = %v, want failed label parity", err)
	}
	data, readErr := os.ReadFile(output)
	if readErr != nil {
		t.Fatalf("ReadFile() error = %v", readErr)
	}
	receipt, parseErr := compatibility.ParseReceipt(data)
	if parseErr != nil {
		t.Fatalf("ParseReceipt() error = %v", parseErr)
	}
	if failed := compatibility.FailedCheckNames(receipt); len(failed) != 1 || failed[0] != compatibility.CheckLabelParity {
		t.Fatalf("failed checks = %v", failed)
	}
}

func TestRunValidateAcceptsFailedEvidence(t *testing.T) {
	data, err := os.ReadFile("../../src/semantic-router/pkg/modelruntime/compatibility/testdata/local-candle-cpu-v1.json")
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	var receipt compatibility.Receipt
	if err = json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	receipt.Checks[0].Passed = false
	data, err = json.Marshal(receipt)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err = os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}

	var stdout bytes.Buffer
	if err = run([]string{"validate", path}, nil, &stdout); err != nil {
		t.Fatalf("run(validate) error = %v", err)
	}
	if !strings.Contains(stdout.String(), "failed checks: 1") {
		t.Fatalf("validate output = %q", stdout.String())
	}
}

func TestRunValidateExpectedDigest(t *testing.T) {
	path := "../../src/semantic-router/pkg/modelruntime/compatibility/testdata/local-candle-cpu-v1.json"
	const digest = "sha256:d02895afb86cb4ffeb146f0902f14048410cf70909612ab0da5bc097e1870c31"
	var stdout bytes.Buffer
	if err := run([]string{"validate", "--expected-digest", digest, path}, nil, &stdout); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(stdout.String(), digest) {
		t.Fatalf("validator did not report the complete receipt digest: %s", stdout.String())
	}
	for _, wrong := range []string{"", "invalid", "sha256:" + strings.Repeat("0", 64)} {
		if err := run([]string{"validate", "--expected-digest", wrong, path}, nil, &bytes.Buffer{}); err == nil {
			t.Fatalf("wrong expected digest %q was accepted", wrong)
		}
	}
}

func TestRunQualifyRequiresOutputFile(t *testing.T) {
	for _, output := range []string{"", "-"} {
		runtime := &fakeRuntime{}
		err := run([]string{
			"qualify-candle-cpu", "--model-path", "unused", "--artifact-revision", "model-commit",
			"--router-revision", "router-commit", "--labels", "A,B", "--suite", "unused",
			"--output", output,
		}, runtime, &bytes.Buffer{})
		if err == nil || !strings.Contains(err.Error(), "--output") || runtime.initialized {
			t.Fatalf("output %q: error=%v initialized=%v", output, err, runtime.initialized)
		}
	}
}

func TestWriteReceiptDoesNotOverwriteEvidence(t *testing.T) {
	data, err := os.ReadFile("../../src/semantic-router/pkg/modelruntime/compatibility/testdata/local-candle-cpu-v1.json")
	if err != nil {
		t.Fatal(err)
	}
	receipt, err := compatibility.ParseReceipt(data)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err = writeReceipt(path, receipt); err != nil {
		t.Fatal(err)
	}
	want, err := receipt.CanonicalJSON()
	if err != nil {
		t.Fatal(err)
	}
	receipt.Checks[0].Passed = false
	if err = writeReceipt(path, receipt); !errors.Is(err, os.ErrExist) {
		t.Fatalf("overwrite error = %v, want os.ErrExist", err)
	}
	got, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(got, want) {
		t.Fatalf("existing evidence changed: %v", err)
	}
}

func writeArtifact(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	for name, content := range map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "weights",
	} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", name, err)
		}
	}
	return directory
}

func writeSuite(t *testing.T) string {
	t.Helper()
	suite := compatibility.QualificationSuite{
		SchemaVersion: compatibility.QualificationSuiteSchemaVersionV1,
		LabelParity: []compatibility.LabelProbe{
			{Name: "basic", Input: "hello", WantLabel: "LABEL_1"},
		},
		InputBounds: []compatibility.InputBoundaryProbe{
			{Name: "empty", Input: ""},
			{Name: "oversized", Input: "oversized", WantError: true},
		},
		DeadlineBehavior: compatibility.DeadlineProbe{Input: "deadline", MaxDurationMillis: 1000},
	}
	data, err := json.Marshal(suite)
	if err != nil {
		t.Fatalf("Marshal() error = %v", err)
	}
	path := filepath.Join(t.TempDir(), "suite.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}
