package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestModelArtifactBindsDerivedManifest(t *testing.T) {
	dir := t.TempDir()
	revision := strings.Repeat("a", 40)
	weights := []byte("tiny fixture graph")
	sum := sha256.Sum256(weights)
	manifest := map[string]interface{}{"format_version": 1, "adapter": "vela_omni", "source": map[string]string{"repo_id": "fixture/omni", "revision": revision}, "max_text_length": 256, "files": map[string]string{"image.onnx": hex.EncodeToString(sum[:])}}
	write := func() {
		t.Helper()
		data, err := json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
		if err = os.WriteFile(filepath.Join(dir, omniManifestFile), data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(dir, "image.onnx"), weights, 0o600); err != nil {
		t.Fatal(err)
	}
	write()
	got, err := modelArtifact(dir, revision)
	if err != nil || got.Repository != "fixture/omni" || len(got.Files) != 2 || got.MaxTextLength != 256 {
		t.Fatalf("invalid evidence: %+v %v", got, err)
	}
	if _, err := modelArtifact(dir, strings.Repeat("b", 40)); err == nil {
		t.Fatal("accepted different revision")
	}
	if err := os.WriteFile(filepath.Join(dir, "image.onnx"), []byte("changed"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := modelArtifact(dir, revision); err == nil {
		t.Fatal("accepted replaced graph")
	}
	manifest["files"] = map[string]string{"../outside": hex.EncodeToString(sum[:])}
	write()
	if _, err := modelArtifact(dir, revision); err == nil {
		t.Fatal("accepted path escape")
	}
}
