//go:build !windows

package modeldownload

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestHuggingFaceCommandsStopAfterStartupCancellation(t *testing.T) {
	commandPath := filepath.Join(t.TempDir(), "hf")
	if err := os.WriteFile(commandPath, []byte("#!/bin/sh\nexec /bin/sleep 1\n"), 0o700); err != nil {
		t.Fatal(err)
	}

	t.Run("download", func(t *testing.T) {
		previousCommand := hfCommand
		hfCommand = commandPath
		t.Cleanup(func() { hfCommand = previousCommand })
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		defer cancel()
		err := DownloadModelWithProgressContext(ctx, ModelSpec{
			LocalPath: t.TempDir(),
			RepoID:    "test/model",
		}, DownloadConfig{})
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("DownloadModelWithProgressContext() error = %v, want deadline exceeded", err)
		}
	})

	t.Run("cli check", func(t *testing.T) {
		t.Setenv("PATH", filepath.Dir(commandPath))
		ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
		defer cancel()
		if err := CheckHuggingFaceCLIContext(ctx); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("CheckHuggingFaceCLIContext() error = %v, want deadline exceeded", err)
		}
	})
}
