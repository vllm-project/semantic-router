package main

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modelDir writes loader inputs and, when snapshot is set, a download record
// in the real hub format: snapshot on line 1, an etag on line 2 that is the
// sha256 of the bytes for the weights and the git blob sha1 for the rest.
func modelDir(t *testing.T, snapshot string, files ...string) string {
	t.Helper()
	dir := t.TempDir()
	meta := filepath.Join(dir, ".cache", "huggingface", "download")
	if err := os.MkdirAll(meta, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, name := range files {
		data := []byte(name + " bytes")
		if err := os.WriteFile(filepath.Join(dir, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
		if snapshot == "" {
			continue
		}
		etag := gitBlobSHA1(data)
		if name == "model.safetensors" {
			sum := sha256.Sum256(data)
			etag = hex.EncodeToString(sum[:])
		}
		if err := os.WriteFile(filepath.Join(meta, name+".metadata"), []byte(snapshot+"\n"+etag+"\n1.0\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func writeRecord(t *testing.T, dir, name, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ".cache", "huggingface", "download", name+".metadata"), []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// The report binds the loaded model bytes: every loader input is hashed, and
// a download record that resolved to another snapshot fails the run.
func TestModelArtifact_HashesFilesAndChecksSnapshot(t *testing.T) {
	all := append(append([]string{}, requiredModelFiles...), optionalModelFiles...)
	dir := modelDir(t, "abc123", all...)
	hashes, err := modelArtifact(dir, "abc123")
	if err != nil {
		t.Fatalf("pinned model rejected: %v", err)
	}
	for _, name := range all {
		if got := hashes[name]; got != fileSHA(filepath.Join(dir, name)) {
			t.Errorf("%s hash = %q, want the file's sha256", name, got)
		}
	}
	if _, err := modelArtifact(dir, "other"); err == nil {
		t.Fatal("snapshot mismatch accepted")
	}
}

// A file replaced after download no longer matches the etag the record
// certifies, for both the LFS (sha256) and git-blob (sha1) forms.
func TestModelArtifact_RejectsBytesThatDisagreeWithTheRecord(t *testing.T) {
	for _, name := range []string{"model.safetensors", "config.json"} {
		t.Run(name, func(t *testing.T) {
			dir := modelDir(t, "abc123", requiredModelFiles...)
			if err := os.WriteFile(filepath.Join(dir, name), []byte("replaced"), 0o644); err != nil {
				t.Fatal(err)
			}
			_, err := modelArtifact(dir, "abc123")
			if err == nil || !strings.Contains(err.Error(), "not the file the download record certifies") {
				t.Fatalf("replaced %s accepted (err=%v)", name, err)
			}
		})
	}
}

// Without download records the hashes still bind the bytes; a missing
// loader input is an error, a missing optional sidecar is not.
func TestModelArtifact_WithoutRecordsAndMissingFiles(t *testing.T) {
	dir := modelDir(t, "", requiredModelFiles...)
	hashes, err := modelArtifact(dir, "anything")
	if err != nil {
		t.Fatalf("hand-copied model rejected: %v", err)
	}
	if len(hashes) != len(requiredModelFiles) {
		t.Fatalf("hashed %d files, want %d required ones", len(hashes), len(requiredModelFiles))
	}
	if err := os.Remove(filepath.Join(dir, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	if _, err := modelArtifact(dir, "anything"); err == nil {
		t.Fatal("missing model.safetensors accepted")
	}
}

// An incomplete or unrecognized record cannot vouch for the file and must
// not be treated as "no record".
func TestModelArtifact_RejectsIncompleteRecords(t *testing.T) {
	for name, content := range map[string]string{
		"empty":          "\n",
		"snapshot only":  "abc123\n",
		"unknown etag":   "abc123\nnot-a-hash\n",
		"quoted sha1 ok": "", // filled below: quoted etags are accepted
	} {
		t.Run(name, func(t *testing.T) {
			dir := modelDir(t, "abc123", requiredModelFiles...)
			if name == "quoted sha1 ok" {
				data, _ := os.ReadFile(filepath.Join(dir, "config.json"))
				writeRecord(t, dir, "config.json", "abc123\n\""+gitBlobSHA1(data)+"\"\n")
				if _, err := modelArtifact(dir, "abc123"); err != nil {
					t.Fatalf("quoted etag rejected: %v", err)
				}
				return
			}
			writeRecord(t, dir, "config.json", content)
			if _, err := modelArtifact(dir, "abc123"); err == nil {
				t.Fatalf("record %q accepted", content)
			}
		})
	}
}
