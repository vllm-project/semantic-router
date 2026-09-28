package modeldownload

import (
	"crypto/sha1" //nolint:gosec // Match HF's Git blob etag in test fixtures.
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeHFRevisionArtifact(t *testing.T, spec ModelSpec, name, content string, lfs bool) {
	t.Helper()
	path := filepath.Join(spec.LocalPath, name)
	metadata := filepath.Join(spec.LocalPath, ".cache", "huggingface", "download", name+".metadata")
	for _, directory := range []string{filepath.Dir(path), filepath.Dir(metadata)} {
		if err := os.MkdirAll(directory, 0o700); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	etag := fmt.Sprintf("%x", sha256.Sum256([]byte(content)))
	if !lfs {
		etag = fmt.Sprintf("%x", sha1.Sum([]byte(fmt.Sprintf("blob %d\x00%s", len(content), content)))) //nolint:gosec // Git fixture identity.
	}
	if err := os.WriteFile(metadata, []byte(fmt.Sprintf("%s\n%s\n%d\n", spec.Revision, etag, time.Now().Unix())), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestPinnedDownloadDoesNotAcceptStalePresentWeights(t *testing.T) {
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	for _, name := range []string{"config.json", "model.safetensors"} {
		if err := os.WriteFile(filepath.Join(spec.LocalPath, name), []byte("fixture"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	assertMissing := func(want bool) {
		t.Helper()
		missing, err := GetMissingModels([]ModelSpec{spec})
		if err != nil || (len(missing) != 0) != want {
			t.Fatalf("missing=%v err=%v wantMissing=%v", missing, err, want)
		}
	}
	assertMissing(true) // Present bytes with no provenance do not prove a release.
	if err := recordModelRevision(spec); err == nil {
		t.Fatal("HF command success cannot turn arbitrary existing bytes into pinned provenance")
	}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors", "fixture", true)
	assertMissing(false) // An ordinary HF CLI snapshot needs no router receipt.
	if err := recordModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	assertMissing(false)
	spec.Revision = "1123456789abcdef0123456789abcdef01234567"
	assertMissing(true)
	if err := invalidateModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	assertMissing(true) // Failed/interrupted refresh must not leave an old receipt.
	if err := recordModelRevision(spec); err == nil {
		t.Fatal("old HF file metadata was accepted for a new pin")
	}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors", "new fixture", true)
	if err := recordModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	assertMissing(false)
	spec.RepoID = "user/custom"
	assertMissing(false) // Local-dir metadata proves the commit/bytes, not repo identity.
	spec.Revision = "main"
	assertMissing(false) // Mutable custom overrides keep their established policy.
}

func TestHFPreloadedSnapshotIsReadOnlyAndReceiptsCannotHideChanges(t *testing.T) {
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "tokenizer.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors", "old weight", true)
	if err := os.Chmod(spec.LocalPath, 0o500); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chmod(spec.LocalPath, 0o700); err != nil {
			t.Error(err)
		}
	})
	if current, err := hasCurrentModelRevision(spec); err != nil || !current {
		t.Fatalf("read-only HF snapshot rejected: current=%v err=%v", current, err)
	}
	if _, err := os.Stat(filepath.Join(spec.LocalPath, modelRevisionReceipt)); !os.IsNotExist(err) {
		t.Fatalf("verification unexpectedly wrote a receipt: %v", err)
	}
	if err := os.Chmod(spec.LocalPath, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := recordModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	weight := filepath.Join(spec.LocalPath, "model.safetensors")
	info, err := os.Stat(weight)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(weight, []byte("bad weight"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(weight, info.ModTime(), info.ModTime()); err != nil {
		t.Fatal(err)
	}
	if current, err := hasCurrentModelRevision(spec); err != nil || current {
		t.Fatalf("same-size/mtime corruption bypassed content check: current=%v err=%v", current, err)
	}
}

func TestHFMetadataRequiresAllRuntimeArtifacts(t *testing.T) {
	for _, name := range []string{"tokenizer.json", "category_mapping.json", "onnx/layer-22/model.onnx.data", "classification_heads.pt"} {
		t.Run(name, func(t *testing.T) {
			spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			writeHFRevisionArtifact(t, spec, "model.safetensors", "weights", true)
			writeHFRevisionArtifact(t, spec, name, "companion", true)
			metadata := filepath.Join(spec.LocalPath, ".cache/huggingface/download", name+".metadata")
			if err := os.Remove(metadata); err != nil {
				t.Fatal(err)
			}
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("unverified runtime artifact accepted: current=%v err=%v", current, err)
			}
		})
	}
}

func TestHFMetadataMalformedAndMixedRevision(t *testing.T) {
	for _, value := range []string{"broken", "other\netag\n1", "0123456789abcdef0123456789abcdef01234567\ninvalid\n1", "0123456789abcdef0123456789abcdef01234567\n1111111111111111111111111111111111111111\nNaN"} {
		t.Run(value, func(t *testing.T) {
			spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
			writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
			writeHFRevisionArtifact(t, spec, "model.safetensors", "weights", true)
			if err := os.WriteFile(filepath.Join(spec.LocalPath, ".cache/huggingface/download/model.safetensors.metadata"), []byte(value), 0o600); err != nil {
				t.Fatal(err)
			}
			if current, err := hasCurrentModelRevision(spec); err != nil || current {
				t.Fatalf("bad metadata accepted: %v %v", current, err)
			}
		})
	}
}

func TestHFIndexedShardsAndExcludedExports(t *testing.T) {
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567", ExcludePatterns: []string{"*.onnx", "*.onnx.data"}}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors.index.json", `{"weight_map":{"weight":"shards/part.safetensors"}}`, false)
	if current, err := hasCurrentModelRevision(spec); err != nil || current {
		t.Fatalf("missing shard accepted: %v %v", current, err)
	}
	writeHFRevisionArtifact(t, spec, "shards/part.safetensors", "weights", true)
	if err := os.MkdirAll(filepath.Join(spec.LocalPath, "onnx/layer-22"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(spec.LocalPath, "onnx/layer-22/model.onnx"), []byte("excluded stale export"), 0o600); err != nil {
		t.Fatal(err)
	}
	if current, err := hasCurrentModelRevision(spec); err != nil || !current {
		t.Fatalf("excluded export invalidated native model: %v %v", current, err)
	}
	spec.ExcludePatterns = nil
	if current, err := hasCurrentModelRevision(spec); err != nil || current {
		t.Fatalf("runtime export without provenance accepted: %v %v", current, err)
	}
}

func TestHFGlobalCacheSnapshotChecksRepoPinAndBlobBytes(t *testing.T) {
	root := t.TempDir()
	spec := ModelSpec{RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	repo := filepath.Join(root, "models--example--release")
	spec.LocalPath = filepath.Join(repo, "snapshots", spec.Revision)
	if err := os.MkdirAll(spec.LocalPath, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(repo, "blobs"), 0o700); err != nil {
		t.Fatal(err)
	}
	for name, value := range map[string]string{"config.json": "{}", "model.safetensors": "weight"} {
		etag := fmt.Sprintf("%x", sha256.Sum256([]byte(value)))
		blob := filepath.Join(repo, "blobs", etag)
		if err := os.WriteFile(blob, []byte(value), 0o600); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(blob, filepath.Join(spec.LocalPath, name)); err != nil {
			t.Fatal(err)
		}
	}
	if current, err := hasCurrentModelRevision(spec); err != nil || !current {
		t.Fatalf("standard cache rejected: %v %v", current, err)
	}
	spec.RepoID = "other/release"
	if current, err := hasCurrentModelRevision(spec); err != nil || current {
		t.Fatalf("wrong cache repository accepted: %v %v", current, err)
	}
}

func TestIncompleteDownloadCannotProduceReleaseReceipt(t *testing.T) {
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	if err := os.WriteFile(filepath.Join(spec.LocalPath, "config.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := recordModelRevision(spec); err == nil {
		t.Fatal("incomplete download recorded as complete")
	}
	if _, err := os.Stat(filepath.Join(spec.LocalPath, modelRevisionReceipt)); !os.IsNotExist(err) {
		t.Fatalf("unexpected receipt: %v", err)
	}
}
