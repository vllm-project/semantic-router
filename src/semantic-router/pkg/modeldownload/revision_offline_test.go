//go:build !windows

package modeldownload

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOfflineCLISuccessCannotPromoteOldModelBytes(t *testing.T) {
	spec := ModelSpec{LocalPath: t.TempDir(), RepoID: "example/release", Revision: "0123456789abcdef0123456789abcdef01234567"}
	writeHFRevisionArtifact(t, spec, "config.json", "{}", false)
	writeHFRevisionArtifact(t, spec, "model.safetensors", "old weights", true)
	if err := recordModelRevision(spec); err != nil {
		t.Fatal(err)
	}
	// This reproduces HF snapshot_download's offline nonempty-local-dir fallback:
	// exit success without replacing a single old file or its download metadata.
	command := filepath.Join(t.TempDir(), "hf")
	if err := os.WriteFile(command, []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	previous := hfCommand
	hfCommand = command
	t.Cleanup(func() { hfCommand = previous })
	spec.Revision = "1123456789abcdef0123456789abcdef01234567"
	err := DownloadModelWithProgressContext(context.Background(), spec, DownloadConfig{})
	if err == nil || !strings.Contains(err.Error(), "pinned HF revision") {
		t.Fatalf("offline old directory promoted: %v", err)
	}
	if _, err := os.Stat(filepath.Join(spec.LocalPath, modelRevisionReceipt)); !os.IsNotExist(err) {
		t.Fatalf("stale receipt survived: %v", err)
	}
	if err := os.WriteFile(command, []byte("#!/bin/sh\nexit 1\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := DownloadModelWithProgressContext(context.Background(), spec, DownloadConfig{}); err == nil || errors.Is(err, ErrGatedModelSkipped) {
		t.Fatalf("failed pinned refresh was silently treated as optional: %v", err)
	}
	if err := os.WriteFile(command, []byte("#!/bin/sh\nexit 0\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	// A mutable override still invalidates prior pinned provenance.
	if err := os.WriteFile(filepath.Join(spec.LocalPath, modelRevisionReceipt), []byte("stale"), 0o600); err != nil {
		t.Fatal(err)
	}
	spec.Revision = "main"
	if err := DownloadModelWithProgressContext(context.Background(), spec, DownloadConfig{}); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(spec.LocalPath, modelRevisionReceipt)); !os.IsNotExist(err) {
		t.Fatalf("mutable download retained pinned provenance: %v", err)
	}
}
