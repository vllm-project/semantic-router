//go:build !windows

package modeldownload

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// fakeHFCommand puts first on PATH an hf command that answers `hf env` and,
// for `hf download`, records its arguments and writes a tokenizer.json into
// --local-dir, except for the repo org/empty. It returns the record's path.
func fakeHFCommand(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	record := filepath.Join(dir, "downloads")
	script := `#!/bin/sh
if [ "$1" = env ]; then echo "- huggingface_hub version: 1.33.0"; exit 0; fi
echo "$@" >> "` + record + `"
repo="$2"
while [ $# -gt 0 ]; do
  if [ "$1" = --local-dir ]; then dir="$2"; fi
  shift
done
mkdir -p "$dir"
if [ "$repo" != org/empty ]; then printf '{}' > "$dir/tokenizer.json"; fi
`
	if err := os.WriteFile(filepath.Join(dir, "hf"), []byte(script), 0o700); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
	hfCommandMu.Lock()
	previous := hfCommand
	hfCommandMu.Unlock()
	t.Cleanup(func() { setHFCommand(previous) })
	return record
}

func recordedDownloads(t *testing.T, record string) []string {
	t.Helper()
	data, err := os.ReadFile(record)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(data)), "\n")
}

func TestTokenizersLiveBesideTheModels(t *testing.T) {
	commit := strings.Repeat("ab", 20)
	for _, tc := range []struct{ repo, revision, dir string }{
		{"deepseek-ai/DeepSeek-V4.1-Flash", "main", "models/tokenizers/deepseek-ai/DeepSeek-V4.1-Flash"},
		{"gpt2", "", "models/tokenizers/gpt2"},
		{"org/model", commit, "models/tokenizers/org/model@" + commit},
		{"org/model", "refs/pr/1", "models/tokenizers/org/model@refs/pr/1"},
	} {
		if got := TokenizerDir(tc.repo, tc.revision); got != filepath.FromSlash(tc.dir) {
			t.Errorf("TokenizerDir(%q, %q) = %q, want %q", tc.repo, tc.revision, got, tc.dir)
		}
	}
	got := buildDownloadArgs(ModelSpec{
		LocalPath: TokenizerDir("org/model", commit), RepoID: "org/model", Revision: commit,
		IncludePatterns: []string{encodingFile},
	})
	want := []string{"download", "org/model", "--local-dir", filepath.FromSlash("models/tokenizers/org/model@" + commit), "--revision", commit, "--include", "tokenizer.json"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("buildDownloadArgs() = %q, want %q", got, want)
	}
}

// EnsureTokenizers downloads only the tokenizer.json of each repo that is
// not on disk, at its revision, through the Hugging Face CLI, and tells its
// listeners; offline, or for a repo without one, it fails.
func TestEnsureTokenizersDownloadsOnlyMissingTokenizerFiles(t *testing.T) {
	t.Chdir(t.TempDir())
	record := fakeHFCommand(t)
	t.Setenv("HF_HUB_OFFLINE", "")
	downloads := 0
	OnTokenizerDownloaded(func() { downloads++ })
	t.Cleanup(func() { tokenizerListeners.fns = nil })

	tokenizers := []string{"o200k_base", "./tokenizer.json", "org/model", "org/model@main", "org/tagged@v1"}
	if err := EnsureTokenizers(context.Background(), tokenizers); err != nil {
		t.Fatal(err)
	}
	want := []string{
		"download org/model --local-dir models/tokenizers/org/model --include tokenizer.json",
		"download org/tagged --local-dir models/tokenizers/org/tagged@v1 --revision v1 --include tokenizer.json",
	}
	if got := recordedDownloads(t, record); !reflect.DeepEqual(got, want) {
		t.Fatalf("downloads = %q, want %q", got, want)
	}
	if downloads != 2 {
		t.Errorf("the listeners heard of %d downloads, want 2", downloads)
	}
	for _, repo := range [][2]string{{"org/model", "main"}, {"org/tagged", "v1"}} {
		if _, ok := TokenizerPath(repo[0], repo[1]); !ok {
			t.Errorf("%s@%s has no tokenizer.json on disk", repo[0], repo[1])
		}
	}

	if err := EnsureTokenizers(context.Background(), tokenizers); err != nil {
		t.Fatal(err)
	}
	if got := recordedDownloads(t, record); len(got) != 2 {
		t.Errorf("a tokenizer on disk was downloaded again: %q", got)
	}

	t.Setenv("HF_HUB_OFFLINE", "1")
	if err := EnsureTokenizers(context.Background(), []string{"org/other"}); err == nil || !strings.Contains(err.Error(), "HF_HUB_OFFLINE") {
		t.Errorf("an offline download should fail, got %v", err)
	}
	t.Setenv("HF_HUB_OFFLINE", "")
	if err := EnsureTokenizers(context.Background(), []string{"org/empty"}); err == nil || !strings.Contains(err.Error(), "has no tokenizer.json") {
		t.Errorf("a repo without a tokenizer.json should fail, got %v", err)
	}
	if got := recordedDownloads(t, record); len(got) != 3 {
		t.Errorf("downloads = %q, want only org/empty's after the first two", got)
	}
}
