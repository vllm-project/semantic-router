package tokencount

import (
	"context"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tokencount/tokenizerspec"
)

// Registry loads tokenizers by spec and keeps them for the life of the
// process. Built-in encodings and local files load when first asked for, as
// does a repo already in the Hugging Face cache; a repo that is not there is
// downloaded in the background, and its callers count with a proxy until it
// arrives.
type Registry struct {
	hub     Hub
	mu      sync.Mutex
	entries map[string]*entry
}

type entry struct {
	ready chan struct{}
	local bool
	enc   *Encoding
	err   error
}

// NewRegistry returns a registry that fetches repos through hub.
func NewRegistry(hub Hub) *Registry {
	return &Registry{hub: hub, entries: map[string]*entry{}}
}

var defaultRegistry = sync.OnceValue(func() *Registry { return NewRegistry(HubFromEnv()) })

// Default is the process's registry, whose hub follows the HF_* environment.
func Default() *Registry { return defaultRegistry() }

// Lookup returns the encoding a spec names, once it is loaded.
func (r *Registry) Lookup(tokenizer string) (*Encoding, bool) {
	e := r.start(tokenizer)
	if e.local {
		<-e.ready
	}
	select {
	case <-e.ready:
		return e.enc, e.err == nil
	default:
		return nil, false
	}
}

// Load loads a spec, waiting until it is loaded or ctx ends.
func (r *Registry) Load(ctx context.Context, tokenizer string) (*Encoding, error) {
	e := r.start(tokenizer)
	select {
	case <-e.ready:
		return e.enc, e.err
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// Preload starts loading every spec in the background.
func (r *Registry) Preload(tokenizers ...string) {
	for _, tokenizer := range tokenizers {
		if strings.TrimSpace(tokenizer) != "" {
			r.start(tokenizer)
		}
	}
}

func (r *Registry) start(tokenizer string) *entry {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.entries[tokenizer]; ok {
		return e
	}
	e := &entry{ready: make(chan struct{})}
	parsed, err := tokenizerspec.Parse(tokenizer)
	if err == nil && parsed.Kind == tokenizerspec.Hub {
		_, e.local = r.hub.cachedPath(parsed.Value, parsed.Revision)
	} else {
		e.local = true
	}
	r.entries[tokenizer] = e
	go func() {
		started := time.Now()
		if err != nil {
			e.err = err
		} else {
			e.enc, e.err = r.load(parsed)
		}
		close(e.ready)
		if e.err != nil {
			logging.ComponentWarnEvent("tokencount", "tokenizer_unavailable", map[string]interface{}{
				"tokenizer": tokenizer, "error": e.err.Error(),
			})
			return
		}
		logging.ComponentEvent("tokencount", "tokenizer_loaded", map[string]interface{}{
			"tokenizer": tokenizer, "vocabulary": len(e.enc.ranks), "elapsed_ms": time.Since(started).Milliseconds(),
		})
	}()
	return e
}

func (r *Registry) load(tokenizer tokenizerspec.Spec) (*Encoding, error) {
	switch tokenizer.Kind {
	case tokenizerspec.Builtin:
		return Builtin(tokenizer.Value)
	case tokenizerspec.File:
		data, err := os.ReadFile(tokenizer.Value)
		if err != nil {
			return nil, err
		}
		return ParseHFTokenizer(tokenizer.Value, data)
	default:
		path, ok := r.hub.cachedPath(tokenizer.Value, tokenizer.Revision)
		if !ok {
			ctx, cancel := context.WithTimeout(context.Background(), hubDownloadTimeout)
			defer cancel()
			var err error
			if path, err = r.hub.download(ctx, tokenizer.Value, tokenizer.Revision); err != nil {
				return nil, err
			}
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return ParseHFTokenizer(tokenizer.Value+"@"+tokenizer.Revision, data)
	}
}
