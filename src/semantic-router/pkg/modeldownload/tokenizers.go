package modeldownload

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// encodingFile is the one file the Router downloads from a model's Hugging
// Face repo: the tokenizer the context window check counts its prompts with.
const encodingFile = "tokenizer.json"

const tokenizerDownloadTimeout = 5 * time.Minute

var (
	// tokenizerDownloads runs one tokenizer download at a time, so that a
	// second caller finds the file the first fetched.
	tokenizerDownloads sync.Mutex
	tokenizerListeners struct {
		sync.Mutex
		fns []func()
	}
)

// TokenizerDir is where the Router keeps a Hugging Face repo's tokenizer,
// beside its other models: models/tokenizers/<repo>, or
// models/tokenizers/<repo>@<revision> for a revision other than main. An
// air-gapped cluster puts the file there itself.
func TokenizerDir(repo, revision string) string {
	if revision != "" && revision != "main" {
		repo += "@" + revision
	}
	return filepath.Join("models", "tokenizers", filepath.FromSlash(repo))
}

// TokenizerPath is the path of repo's tokenizer.json at revision, and whether
// a regular file is there.
func TokenizerPath(repo, revision string) (string, bool) {
	path := filepath.Join(TokenizerDir(repo, revision), encodingFile)
	info, err := os.Stat(path)
	return path, err == nil && info.Mode().IsRegular()
}

// OnTokenizerDownloaded calls fn after each tokenizer download.
func OnTokenizerDownloaded(fn func()) {
	tokenizerListeners.Lock()
	defer tokenizerListeners.Unlock()
	tokenizerListeners.fns = append(tokenizerListeners.fns, fn)
}

// PrefetchTokenizers runs EnsureTokenizers in the background when one of
// tokenizers is a Hugging Face repo whose file is missing, so that neither
// startup nor a request waits for a download: the repo's models count with
// the proxy tokenizer until it arrives.
func PrefetchTokenizers(tokenizers []string) {
	missing := missingTokenizers(tokenizers)
	if len(missing) == 0 {
		return
	}
	go func() {
		_ = ensureTokenizers(context.Background(), missing)
	}()
}

// EnsureTokenizers downloads the tokenizer.json of each Hugging Face repo
// among tokenizers that is not on disk yet, one at a time, through the
// Hugging Face CLI with the settings of the Router's other model downloads:
// HF_ENDPOINT, HF_TOKEN, HF_HOME and the repo's revision. With HF_HUB_OFFLINE
// set it downloads nothing. It skips built-in encodings and local files, and
// returns the failures together; a repo that fails leaves its models counting
// with the proxy tokenizer.
func EnsureTokenizers(ctx context.Context, tokenizers []string) error {
	return ensureTokenizers(ctx, missingTokenizers(tokenizers))
}

func ensureTokenizers(ctx context.Context, repos []tokenizerspec.Spec) error {
	var failures error
	for _, repo := range repos {
		name := repo.Value + "@" + repo.Revision
		started := time.Now()
		path, err := ensureTokenizer(ctx, repo)
		if err != nil {
			failures = errors.Join(failures, fmt.Errorf("%s: %w", name, err))
			logging.ComponentWarnEvent("router", "tokenizer_download_failed", map[string]interface{}{
				"tokenizer": name, "counts_with": "proxy_tokenizer", "error": err.Error(),
			})
			continue
		}
		logging.ComponentEvent("router", "tokenizer_downloaded", map[string]interface{}{
			"tokenizer": name, "path": path, "elapsed_ms": time.Since(started).Milliseconds(),
		})
		tokenizerListeners.Lock()
		listeners := append([]func(){}, tokenizerListeners.fns...)
		tokenizerListeners.Unlock()
		for _, listener := range listeners {
			listener()
		}
	}
	return failures
}

func ensureTokenizer(ctx context.Context, repo tokenizerspec.Spec) (string, error) {
	tokenizerDownloads.Lock()
	defer tokenizerDownloads.Unlock()
	if path, ok := TokenizerPath(repo.Value, repo.Revision); ok {
		return path, nil
	}
	if hubOffline() {
		return "", errors.New("it is not on disk, and HF_HUB_OFFLINE is set")
	}
	ctx, cancel := context.WithTimeout(ctx, tokenizerDownloadTimeout)
	defer cancel()
	if err := CheckHuggingFaceCLIContext(ctx); err != nil {
		return "", err
	}
	revision := repo.Revision
	if revision == "main" {
		revision = ""
	}
	spec := ModelSpec{
		LocalPath:       TokenizerDir(repo.Value, repo.Revision),
		RepoID:          repo.Value,
		Revision:        revision,
		RequiredFiles:   []string{encodingFile},
		FilesOnly:       true,
		IncludePatterns: []string{encodingFile},
	}
	if err := DownloadModelWithProgressContext(ctx, spec, GetDownloadConfig()); err != nil {
		return "", err
	}
	path, ok := TokenizerPath(repo.Value, repo.Revision)
	if !ok {
		return "", fmt.Errorf("the repo has no %s", encodingFile)
	}
	return path, nil
}

// missingTokenizers are the Hugging Face repos among tokenizers whose file
// is not on disk, each once.
func missingTokenizers(tokenizers []string) []tokenizerspec.Spec {
	var missing []tokenizerspec.Spec
	seen := map[tokenizerspec.Spec]bool{}
	for _, tokenizer := range tokenizers {
		spec, err := tokenizerspec.Parse(tokenizer)
		if err != nil || spec.Kind != tokenizerspec.Hub || seen[spec] {
			continue
		}
		seen[spec] = true
		if _, ok := TokenizerPath(spec.Value, spec.Revision); !ok {
			missing = append(missing, spec)
		}
	}
	return missing
}

func hubOffline() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("HF_HUB_OFFLINE"))) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}
