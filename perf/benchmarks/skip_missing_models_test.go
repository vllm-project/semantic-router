package benchmarks

import (
	"errors"
	"fmt"
	"io/fs"
	"testing"
)

func TestMissingBenchModels(t *testing.T) {
	if missingBenchModels(nil) {
		t.Fatal("nil should not skip")
	}
	if missingBenchModels(errors.New("candle init failed")) {
		t.Fatal("real init errors should still fail the bench")
	}
	if !missingBenchModels(fs.ErrNotExist) {
		t.Fatal("ErrNotExist should skip")
	}
	if !missingBenchModels(fmt.Errorf("stat models directory %q: %w", "/tmp/models", fs.ErrNotExist)) {
		t.Fatal("wrapped ErrNotExist should skip")
	}
	if missingBenchModels(errors.New("models directory does not exist: /tmp/models")) {
		t.Fatal("text-only discovery miss should still fail the bench")
	}
	if missingBenchModels(errors.New("embedding model dir not found at models/mom-embedding-pro")) {
		t.Fatal("text-only cache miss should still fail the bench")
	}
	if missingBenchModels(fmt.Errorf("intent model dir not found at %s: %w", "models/x", fs.ErrPermission)) {
		t.Fatal("permission errors should still fail the bench")
	}
	if missingBenchModels(fmt.Errorf("embedding model dir not found at %s: %w", "models/x", errSymlinkLoop)) {
		t.Fatal("symlink-loop errors should still fail the bench")
	}
	if missingBenchModels(fmt.Errorf("label not found: %w", errors.New("io timeout"))) {
		t.Fatal("unrelated not-found text should still fail the bench")
	}
}

var errSymlinkLoop = errors.New("too many levels of symbolic links")
