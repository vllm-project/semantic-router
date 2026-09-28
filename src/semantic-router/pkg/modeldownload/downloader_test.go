package modeldownload

import (
	"reflect"
	"strings"
	"testing"
)

// TestBuildDownloadArgsFetchesFullSnapshotByDefault keeps the historical contract for
// models without a narrowed download scope: repo ID plus --local-dir, nothing else.
func TestBuildDownloadArgsFetchesFullSnapshotByDefault(t *testing.T) {
	spec := ModelSpec{
		LocalPath: "models/category_classifier_modernbert-base_model",
		RepoID:    "llm-semantic-router/category_classifier_modernbert-base_model",
	}

	want := []string{
		"download",
		spec.RepoID,
		"--local-dir", spec.LocalPath,
	}
	for _, revision := range []string{"", "main"} {
		spec.Revision = revision
		if got := buildDownloadArgs(spec); !reflect.DeepEqual(got, want) {
			t.Fatalf("revision %q: buildDownloadArgs() = %#v, want %#v", revision, got, want)
		}
	}
}

// TestBuildDownloadArgsRepeatsExcludeFlagPerPattern guards the CLI contract: the
// typer-based `hf download` takes `--exclude` as a repeatable single-value option, so
// one flag followed by several patterns silently drops all but the first and passes
// the rest as positional filenames. Each pattern must carry its own flag.
func TestBuildDownloadArgsRepeatsExcludeFlagPerPattern(t *testing.T) {
	spec := ModelSpec{
		LocalPath:       testEmbeddingModelPath,
		RepoID:          testEmbeddingRepoID,
		Revision:        "abc123",
		ExcludePatterns: []string{"*.onnx", "*.onnx.data", "*.onnx_data"},
	}

	got := buildDownloadArgs(spec)
	want := []string{
		"download",
		testEmbeddingRepoID,
		"--local-dir", testEmbeddingModelPath,
		"--revision", "abc123",
		"--exclude", "*.onnx",
		"--exclude", "*.onnx.data",
		"--exclude", "*.onnx_data",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildDownloadArgs() = %#v, want %#v", got, want)
	}

	// No two patterns may ever share one flag: with the typer CLI the second would be
	// parsed as a positional filename instead of a filter.
	flags := 0
	for i, arg := range got {
		if arg != "--exclude" {
			continue
		}
		flags++
		if i+1 >= len(got) || strings.HasPrefix(got[i+1], "-") {
			t.Fatalf("buildDownloadArgs() = %#v: --exclude at %d carries no pattern", got, i)
		}
	}
	if flags != len(spec.ExcludePatterns) {
		t.Fatalf("buildDownloadArgs() = %#v: %d --exclude flags for %d patterns", got, flags, len(spec.ExcludePatterns))
	}
}

// TestBuildDownloadArgsSkipsEmptyExcludePatterns keeps a stray empty entry from
// producing a bare `--exclude` that would swallow nothing or error out.
func TestBuildDownloadArgsSkipsEmptyExcludePatterns(t *testing.T) {
	spec := ModelSpec{
		LocalPath:       testEmbeddingModelPath,
		RepoID:          testEmbeddingRepoID,
		Revision:        "main",
		ExcludePatterns: []string{"", "*.onnx", ""},
	}

	got := buildDownloadArgs(spec)
	want := []string{
		"download",
		testEmbeddingRepoID,
		"--local-dir", testEmbeddingModelPath,
		"--exclude", "*.onnx",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildDownloadArgs() = %#v, want %#v", got, want)
	}
}
