package tokencount

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

const (
	hubFile            = "tokenizer.json"
	hubDownloadTimeout = 2 * time.Minute
	maxTokenizerBytes  = 256 << 20
)

var commitHash = regexp.MustCompile(`^[0-9a-f]{40}$`)

// Hub reads a repo's tokenizer.json from the Hugging Face cache, in the
// layout huggingface_hub writes, and downloads it into that cache when it is
// missing and the hub is not offline.
type Hub struct {
	Endpoint string
	Token    string
	CacheDir string
	Offline  bool
	Client   *http.Client
}

// HubFromEnv follows the environment huggingface_hub reads: HF_ENDPOINT,
// HF_TOKEN, HF_HUB_CACHE or HF_HOME, and HF_HUB_OFFLINE.
func HubFromEnv() Hub {
	cache := os.Getenv("HF_HUB_CACHE")
	if cache == "" {
		home := os.Getenv("HF_HOME")
		if home == "" {
			if dir, err := os.UserHomeDir(); err == nil {
				home = filepath.Join(dir, ".cache", "huggingface")
			}
		}
		cache = filepath.Join(home, "hub")
	}
	endpoint := os.Getenv("HF_ENDPOINT")
	if endpoint == "" {
		endpoint = "https://huggingface.co"
	}
	token := os.Getenv("HF_TOKEN")
	if token == "" {
		token = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	return Hub{
		Endpoint: endpoint, Token: token, CacheDir: cache,
		Offline: truthy(os.Getenv("HF_HUB_OFFLINE")) || truthy(os.Getenv("TRANSFORMERS_OFFLINE")),
	}
}

func truthy(value string) bool {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

func (h Hub) repoDir(repo string) string {
	return filepath.Join(h.CacheDir, "models--"+strings.ReplaceAll(repo, "/", "--"))
}

// cachedPath finds the repo's tokenizer.json at revision in the cache. The
// default revision also takes any snapshot when no ref names one.
func (h Hub) cachedPath(repo, revision string) (string, bool) {
	dir := h.repoDir(repo)
	snapshot := revision
	if !commitHash.MatchString(revision) {
		ref, err := os.ReadFile(filepath.Join(dir, "refs", revision))
		snapshot = strings.TrimSpace(string(ref))
		if err != nil || !commitHash.MatchString(snapshot) {
			if revision != "main" {
				return "", false
			}
			matches, _ := filepath.Glob(filepath.Join(dir, "snapshots", "*", hubFile))
			if len(matches) == 0 {
				return "", false
			}
			return matches[0], true
		}
	}
	path := filepath.Join(dir, "snapshots", snapshot, hubFile)
	if info, err := os.Stat(path); err != nil || info.IsDir() {
		return "", false
	}
	return path, true
}

// download fetches the repo's tokenizer.json at revision into the cache, at
// the commit the hub resolves the revision to.
func (h Hub) download(ctx context.Context, repo, revision string) (string, error) {
	if h.Offline {
		return "", errors.New("the tokenizer is not in the Hugging Face cache, and the hub is offline")
	}
	source := strings.TrimSuffix(h.Endpoint, "/") + "/" + repo + "/resolve/" + url.PathEscape(revision) + "/" + hubFile
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, source, nil)
	if err != nil {
		return "", err
	}
	if h.Token != "" {
		request.Header.Set(headers.Authorization, "Bearer "+h.Token)
	}
	commit := ""
	client := http.Client{Timeout: hubDownloadTimeout}
	if h.Client != nil {
		client = *h.Client
	}
	client.CheckRedirect = func(next *http.Request, via []*http.Request) error {
		if commit == "" && next.Response != nil {
			commit = next.Response.Header.Get(headers.HubRepoCommit)
		}
		if len(via) >= 10 {
			return errors.New("too many redirects")
		}
		return nil
	}
	response, err := client.Do(request)
	if err != nil {
		return "", err
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: %s", source, response.Status)
	}
	if commit == "" {
		commit = response.Header.Get(headers.HubRepoCommit)
	}
	if !commitHash.MatchString(commit) {
		commit = revision
	}
	data, err := io.ReadAll(io.LimitReader(response.Body, maxTokenizerBytes+1))
	if err != nil {
		return "", err
	}
	if len(data) > maxTokenizerBytes {
		return "", fmt.Errorf("download %s: larger than %d bytes", source, maxTokenizerBytes)
	}
	dir := h.repoDir(repo)
	path := filepath.Join(dir, "snapshots", commit, hubFile)
	if err := writeFileAtomic(path, data); err != nil {
		return "", err
	}
	if commit != revision {
		if err := writeFileAtomic(filepath.Join(dir, "refs", revision), []byte(commit)); err != nil {
			return "", err
		}
	}
	return path, nil
}

func writeFileAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), ".tokenizer-*")
	if err != nil {
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmp.Name())
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmp.Name())
		return err
	}
	return os.Rename(tmp.Name(), path)
}
