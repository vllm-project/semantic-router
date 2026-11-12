package tokencount

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

const (
	encodingFile      = "tokenizer.json"
	maxTokenizerBytes = 256 << 20
)

// RepoPath finds a Hugging Face repo's tokenizer.json at a revision on disk,
// and reports whether it is there.
type RepoPath func(repo, revision string) (string, bool)

// Registry loads tokenizers by spec and keeps them for the life of the
// process. A built-in encoding, a local file (a tokenizer.json, or the
// directory that holds one) or a Hugging Face repo whose tokenizer.json is on
// disk loads when first asked for. A repo whose file is not there yet is
// pending, and its models count with the proxy, until a Preload or Refresh
// finds the file; the Registry never downloads.
type Registry struct {
	repoPath RepoPath
	notify   func(tokenizer string, loaded bool)
	mu       sync.Mutex
	entries  map[string]*entry
	pending  map[string]bool
}

type entry struct {
	ready chan struct{}
	enc   *Encoding
	err   error
}

// NewRegistry returns a registry that finds repos with repoPath and, when
// notify is not nil, tells it whether each tokenizer it is asked for is
// loaded: false while it is pending, loading or failed, true once loaded.
func NewRegistry(repoPath RepoPath, notify func(tokenizer string, loaded bool)) *Registry {
	return &Registry{repoPath: repoPath, notify: notify, entries: map[string]*entry{}, pending: map[string]bool{}}
}

// Lookup returns the encoding a spec names, waiting for a load from disk the
// first time. A pending repo finds nothing.
func (r *Registry) Lookup(tokenizer string) (*Encoding, bool) {
	e := r.start(tokenizer, false)
	if e == nil {
		return nil, false
	}
	<-e.ready
	return e.enc, e.err == nil
}

// Load loads a spec, waiting until it is loaded or ctx ends. A pending repo
// fails at once.
func (r *Registry) Load(ctx context.Context, tokenizer string) (*Encoding, error) {
	e := r.start(tokenizer, true)
	if e == nil {
		return nil, fmt.Errorf("the tokenizer.json of %s is not on disk yet", tokenizer)
	}
	select {
	case <-e.ready:
		return e.enc, e.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Preload starts loading every spec in the background, looking again for
// the files of pending repos.
func (r *Registry) Preload(tokenizers ...string) {
	for _, tokenizer := range tokenizers {
		if strings.TrimSpace(tokenizer) != "" {
			r.start(tokenizer, true)
		}
	}
}

// Refresh looks again for the files of pending repos and loads those it
// finds.
func (r *Registry) Refresh() {
	r.mu.Lock()
	pending := make([]string, 0, len(r.pending))
	for tokenizer := range r.pending {
		pending = append(pending, tokenizer)
	}
	r.mu.Unlock()
	r.Preload(pending...)
}

// start returns the spec's entry, starting its load the first time, or nil
// while the spec is a repo whose file is not on disk. The file of a repo
// already pending is looked for again only when look is set.
func (r *Registry) start(tokenizer string, look bool) *entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.entries[tokenizer]; ok {
		return e
	}
	if r.pending[tokenizer] && !look {
		return nil
	}
	parsed, err := tokenizerspec.Parse(tokenizer)
	path := ""
	if err == nil && parsed.Kind == tokenizerspec.Hub {
		found := false
		if r.repoPath != nil {
			path, found = r.repoPath(parsed.Value, parsed.Revision)
		}
		if !found {
			if !r.pending[tokenizer] {
				r.pending[tokenizer] = true
				r.tell(tokenizer, false)
				logging.ComponentWarnEvent("tokencount", "tokenizer_pending", map[string]interface{}{
					"tokenizer": tokenizer, "counts_with": "proxy_tokenizer",
					"reason": "its tokenizer.json is not on disk yet",
				})
			}
			return nil
		}
		delete(r.pending, tokenizer)
	}
	e := &entry{ready: make(chan struct{})}
	r.entries[tokenizer] = e
	r.tell(tokenizer, false)
	go func() {
		started := time.Now()
		if err != nil {
			e.err = err
		} else {
			e.enc, e.err = loadRecovered(parsed, path)
		}
		r.tell(tokenizer, e.err == nil)
		close(e.ready)
		if e.err != nil {
			logging.ComponentWarnEvent("tokencount", "tokenizer_unavailable", map[string]interface{}{
				"tokenizer": tokenizer, "counts_with": "proxy_tokenizer", "error": e.err.Error(),
			})
			return
		}
		logging.ComponentEvent("tokencount", "tokenizer_loaded", map[string]interface{}{
			"tokenizer": tokenizer, "vocabulary": len(e.enc.ranks), "elapsed_ms": time.Since(started).Milliseconds(),
		})
	}()
	return e
}

func (r *Registry) tell(tokenizer string, loaded bool) {
	if r.notify != nil {
		r.notify(tokenizer, loaded)
	}
}

// loadRecovered is load with a panic turned into its error, so that a
// malformed tokenizer leaves its models counting with the proxy instead of
// stopping the Router.
func loadRecovered(tokenizer tokenizerspec.Spec, repoFile string) (enc *Encoding, err error) {
	defer func() {
		if recovered := recover(); recovered != nil {
			enc, err = nil, fmt.Errorf("the tokenizer failed to load: %v", recovered)
		}
	}()
	return load(tokenizer, repoFile)
}

// load loads a spec; repoFile is where a repo's tokenizer.json is.
func load(tokenizer tokenizerspec.Spec, repoFile string) (*Encoding, error) {
	switch tokenizer.Kind {
	case tokenizerspec.Builtin:
		return Builtin(tokenizer.Value)
	case tokenizerspec.File:
		path := tokenizer.Value
		if info, err := os.Stat(path); err == nil && info.IsDir() {
			path = filepath.Join(path, encodingFile)
		}
		data, err := readTokenizerFile(path)
		if err != nil {
			return nil, err
		}
		return ParseHFTokenizer(tokenizer.Value, data)
	default:
		data, err := readTokenizerFile(repoFile)
		if err != nil {
			return nil, err
		}
		return ParseHFTokenizer(tokenizer.Value+"@"+tokenizer.Revision, data)
	}
}

// readTokenizerFile reads a tokenizer.json that is a regular file of at most
// maxTokenizerBytes, so that a path to a device, a pipe or a huge file fails
// to load instead of blocking or exhausting memory. It checks before
// opening, since opening a pipe waits for a writer.
func readTokenizerFile(path string) ([]byte, error) {
	info, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s is not a regular file", path)
	}
	if info.Size() > maxTokenizerBytes {
		return nil, fmt.Errorf("%s is larger than %d bytes", path, maxTokenizerBytes)
	}
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(io.LimitReader(file, maxTokenizerBytes+1))
	if err != nil {
		return nil, err
	}
	if len(data) > maxTokenizerBytes {
		return nil, fmt.Errorf("%s is larger than %d bytes", path, maxTokenizerBytes)
	}
	return data, nil
}
