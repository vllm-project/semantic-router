package modelassets

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveFrozenManifest(t *testing.T) {
	root := t.TempDir()
	manifest := Manifest{Provider: "candle", Models: []Artifact{{Env: "VLLM_SR_DOMAIN_MODEL", Path: "models/current", RepoID: "current/domain", Revision: "current-checkpoint"}}}
	data, _ := json.Marshal(manifest)
	path := filepath.Join(root, "manifest.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv(ManifestEnv, path)
	t.Setenv("VLLM_SR_DOMAIN_MODEL", "")
	artifact, err := Resolve("domain", root)
	if err != nil || artifact.Revision != "current-checkpoint" || artifact.Path != filepath.Join(root, "models/current") {
		t.Fatalf("base source did not use frozen current model: %+v %v", artifact, err)
	}
	t.Setenv("VLLM_SR_DOMAIN_MODEL", filepath.Join(root, "snapshot"))
	artifact, err = Resolve("domain", root)
	if err != nil || artifact.Path != filepath.Join(root, "snapshot") {
		t.Fatalf("explicit snapshot override lost: %+v %v", artifact, err)
	}
	if _, err = Resolve("embedding", root); err == nil {
		t.Fatal("missing manifest model fell back to base defaults")
	}
}

func TestContentsIdentityFollowsWeightsNotDownloadCache(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "model.safetensors")
	if err := os.WriteFile(path, []byte("first"), 0o600); err != nil {
		t.Fatal(err)
	}
	first, err := ContentsSHA256(root)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.Mkdir(filepath.Join(root, ".cache"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(filepath.Join(root, ".cache", "download"), []byte("metadata"), 0o600); err != nil {
		t.Fatal(err)
	}
	same, err := ContentsSHA256(root)
	if err != nil || same != first {
		t.Fatal("download metadata changed checkpoint identity")
	}
	if err = os.WriteFile(path, []byte("replaced"), 0o600); err != nil {
		t.Fatal(err)
	}
	changed, err := ContentsSHA256(root)
	if err != nil || changed == first {
		t.Fatal("in-place weights replacement kept checkpoint identity")
	}
}

func TestContentsIdentitySupportsHFSnapshotLinks(t *testing.T) {
	root := t.TempDir()
	snapshot := filepath.Join(root, "snapshots", "revision")
	blobs := filepath.Join(root, "blobs")
	for _, directory := range []string{snapshot, blobs} {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	weight := []byte("shared weights")
	if err := os.WriteFile(filepath.Join(blobs, "weights"), weight, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("../../blobs/weights", filepath.Join(snapshot, "model.safetensors")); err != nil {
		t.Fatal(err)
	}
	plain := t.TempDir()
	if err := os.WriteFile(filepath.Join(plain, "model.safetensors"), weight, 0o600); err != nil {
		t.Fatal(err)
	}
	want, err := ContentsSHA256(plain)
	if err != nil {
		t.Fatal(err)
	}
	alias := filepath.Join(root, "selected")
	if err = os.Symlink(snapshot, alias); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{snapshot, alias} {
		got, hashErr := ContentsSHA256(path)
		if hashErr != nil || got != want {
			t.Fatalf("snapshot links changed content identity: %s %v", got, hashErr)
		}
	}
	if err = os.Symlink(blobs, filepath.Join(snapshot, "not-a-model-file")); err != nil {
		t.Fatal(err)
	}
	if _, err = ContentsSHA256(snapshot); err == nil {
		t.Fatal("accepted a directory as an artifact file")
	}
}
