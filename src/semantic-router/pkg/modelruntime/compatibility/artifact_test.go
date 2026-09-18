package compatibility

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDigestLocalCandleArtifact(t *testing.T) {
	first := writeTestCandleArtifact(t, map[string]string{
		"config.json":       `{"model_type":"bert"}`,
		"tokenizer.json":    `{"version":"1.0"}`,
		"model.safetensors": "weights",
	})
	second := writeTestCandleArtifact(t, map[string]string{
		"config.json":       `{"model_type":"bert"}`,
		"tokenizer.json":    `{"version":"1.0"}`,
		"model.safetensors": "weights",
	})

	firstDigest, err := DigestLocalCandleArtifact(first)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() error = %v", err)
	}
	secondDigest, err := DigestLocalCandleArtifact(second)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() second error = %v", err)
	}
	if firstDigest != secondDigest {
		t.Fatalf("identical artifact digests = %q and %q", firstDigest, secondDigest)
	}

	if err = os.WriteFile(filepath.Join(second, "tokenizer.json"), []byte(`{"version":"2.0"}`), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	changedDigest, err := DigestLocalCandleArtifact(second)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() changed error = %v", err)
	}
	if changedDigest == firstDigest {
		t.Fatalf("changed artifact digest = %q, want different from %q", changedDigest, firstDigest)
	}
}

func TestDigestLocalCandleArtifactMatchesLoaderWeightPreference(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":       "config",
		"tokenizer.json":    "tokenizer",
		"model.safetensors": "preferred",
		"pytorch_model.bin": "fallback",
	})
	before, err := DigestLocalCandleArtifact(artifact)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() error = %v", err)
	}
	const want = "sha256:3406b092da0c9ac32b232f4366a0b86526604dd29bffbee3bb34dcc4260cded9"
	if before != want {
		t.Fatalf("DigestLocalCandleArtifact() = %q, want golden %q", before, want)
	}
	if err = os.WriteFile(filepath.Join(artifact, "pytorch_model.bin"), []byte("changed fallback"), 0o600); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	after, err := DigestLocalCandleArtifact(artifact)
	if err != nil {
		t.Fatalf("DigestLocalCandleArtifact() repeated error = %v", err)
	}
	if after != before {
		t.Fatalf("fallback-only drift changed selected artifact digest: %q != %q", after, before)
	}
}

func TestDigestLocalCandleArtifactRejectsIncompleteArtifact(t *testing.T) {
	artifact := writeTestCandleArtifact(t, map[string]string{
		"config.json":    "config",
		"tokenizer.json": "tokenizer",
	})
	_, err := DigestLocalCandleArtifact(artifact)
	if err == nil || !strings.Contains(err.Error(), "model.safetensors or pytorch_model.bin") {
		t.Fatalf("DigestLocalCandleArtifact() error = %v, want missing weights", err)
	}
}

func writeTestCandleArtifact(t *testing.T, files map[string]string) string {
	t.Helper()
	directory := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(content), 0o600); err != nil {
			t.Fatalf("WriteFile(%q) error = %v", name, err)
		}
	}
	return directory
}
