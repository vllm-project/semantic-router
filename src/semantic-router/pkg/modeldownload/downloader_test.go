package modeldownload

import (
	"reflect"
	"testing"
)

// TestBuildDownloadArgsFetchesFullSnapshotByDefault keeps the historical contract for
// models without a narrowed download scope: repo ID plus --local-dir, nothing else.
func TestBuildDownloadArgsFetchesFullSnapshotByDefault(t *testing.T) {
	spec := ModelSpec{
		LocalPath: "models/category_classifier_modernbert-base_model",
		RepoID:    "llm-semantic-router/category_classifier_modernbert-base_model",
		Revision:  "main",
	}

	got := buildDownloadArgs(spec)
	want := []string{
		"download",
		spec.RepoID,
		"--local-dir", spec.LocalPath,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildDownloadArgs() = %#v, want %#v", got, want)
	}
}

// TestBuildDownloadArgsAppendsExcludePatternsLast guards the CLI contract:
// `--exclude` is variadic, so it must come after every other flag or it would
// swallow them as patterns.
func TestBuildDownloadArgsAppendsExcludePatternsLast(t *testing.T) {
	spec := ModelSpec{
		LocalPath:       testEmbeddingModelPath,
		RepoID:          testEmbeddingRepoID,
		Revision:        "abc123",
		ExcludePatterns: []string{"*.onnx", "*.onnx.data"},
	}

	got := buildDownloadArgs(spec)
	want := []string{
		"download",
		testEmbeddingRepoID,
		"--local-dir", testEmbeddingModelPath,
		"--revision", "abc123",
		"--exclude", "*.onnx", "*.onnx.data",
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("buildDownloadArgs() = %#v, want %#v", got, want)
	}
}
