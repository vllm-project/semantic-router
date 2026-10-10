package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelArtifactAttestsEverySnapshotFile(t *testing.T) {
	dir := t.TempDir()
	revision := strings.Repeat("a", 40)
	write := func(name, content string) {
		t.Helper()
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	write("config.json", `{"architectures": ["VelaOmni"], "max_text_length": 256}`)
	write("components/text/tokenizer.json", "{}")
	write(".cache/huggingface/download/config.json.metadata", revision)
	got, err := modelArtifact(dir, "fixture/omni", revision)
	if err != nil || got.Repository != "fixture/omni" || got.MaxTextLength != 256 || len(got.Files) != 2 {
		t.Fatalf("invalid evidence: %+v %v", got, err)
	}
	if got.Files["components/text/tokenizer.json"] != "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" {
		t.Fatalf("unexpected digest: %+v", got.Files)
	}
	if _, err := modelArtifact(dir, "fixture/omni", "main"); err == nil {
		t.Fatal("accepted a branch for a revision")
	}
	if err := os.Symlink("/etc/hostname", filepath.Join(dir, "linked")); err != nil {
		t.Fatal(err)
	}
	if _, err := modelArtifact(dir, "fixture/omni", revision); err == nil {
		t.Fatal("accepted a link out of the snapshot")
	}
	write("config.json", `{"architectures": ["Other"], "max_text_length": 256}`)
	if _, err := modelArtifact(dir, "fixture/omni", revision); err == nil {
		t.Fatal("accepted another architecture")
	}
}
