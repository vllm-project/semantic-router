//go:build !windows && cgo

package benchmarks

import (
	"errors"
	"os"
	"testing"
)

func TestMissingBenchModels(t *testing.T) {
	if missingBenchModels(nil) {
		t.Fatal("nil should not skip")
	}
	if missingBenchModels(errors.New("candle init failed")) {
		t.Fatal("real init errors should still fail the bench")
	}
	if !missingBenchModels(os.ErrNotExist) {
		t.Fatal("ErrNotExist should skip")
	}
	if !missingBenchModels(errors.New("models directory does not exist: /tmp/models")) {
		t.Fatal("discovery miss should skip")
	}
	if !missingBenchModels(errors.New("embedding model dir not found at models/mom-embedding-pro")) {
		t.Fatal("cache miss should skip")
	}
}
