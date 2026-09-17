package extproc

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// Parsing is stubbed in lifecycle tests; retain a real source document identity
// so publication still exercises the production stale-candidate check.
func writeReloadTestDocument(t *testing.T, path, document string, cfg *config.RouterConfig) {
	t.Helper()
	data := []byte(document)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256(data)
	cfg.DocumentHash = hex.EncodeToString(digest[:])
}

func TestFileReloadDoesNotPublishSupersededWarmup(t *testing.T) {
	for _, tc := range []struct {
		name         string
		removeSource bool
		closeFails   bool
	}{
		{name: "restore-active-document"},
		{name: "source-unreadable", removeSource: true},
		{name: "discard-close-failure", closeFails: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			restore := stubReloadSeams(t)
			defer restore()
			path := filepath.Join(t.TempDir(), "router.yaml")
			active := &config.RouterConfig{}
			writeReloadTestDocument(t, path, "A", active)
			registry := routerruntime.NewRegistry(active)
			previous := &OpenAIRouter{Config: active}
			server := &Server{runtime: registry, service: NewRouterService(previous)}
			t.Cleanup(func() { _ = server.service.Close() })
			candidate := &config.RouterConfig{}
			writeReloadTestDocument(t, path, "B", candidate)
			parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
			ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
			prepareReloadRuntime = func(*config.RouterConfig) (modelruntime.EmbeddingRuntimeState, error) {
				return modelruntime.EmbeddingRuntimeState{}, nil
			}
			var candidateCloses atomic.Int32
			closeFailure := errors.New("test resource close failed")
			buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
				resources := newResourceScope()
				resources.add(func() error {
					candidateCloses.Add(1)
					if tc.closeFails {
						return closeFailure
					}
					return nil
				})
				return &OpenAIRouter{Config: cfg, resources: resources}, nil
			}
			started, finish := make(chan struct{}), make(chan struct{})
			var once sync.Once
			releaseWarmup := func() { once.Do(func() { close(finish) }) }
			t.Cleanup(releaseWarmup)
			warmupReloadRouter = func(*OpenAIRouter, modelruntime.EmbeddingRuntimeState) error {
				close(started)
				<-finish
				return nil
			}
			result := make(chan error, 1)
			go func() { result <- server.reloadRouterFromFile(path) }()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("candidate warmup did not start")
			}
			// This is the same per-Router transaction used by API persistence.
			// It must remain available while an older generation is warming.
			release := registry.LockConfigPublication()
			if tc.removeSource {
				if err := os.Remove(path); err != nil {
					t.Fatal(err)
				}
			} else {
				writeReloadTestDocument(t, path, "A", active)
			}
			release()
			releaseWarmup()
			select {
			case err := <-result:
				if err == nil || (!tc.removeSource && !tc.closeFails && !errors.Is(err, errConfigReloadSuperseded)) {
					t.Fatalf("reload error = %v", err)
				}
				if errors.Is(err, closeFailure) != tc.closeFails || (tc.closeFails && errors.Is(err, errConfigReloadSuperseded)) {
					t.Fatalf("discard cleanup failure was lost: %v", err)
				}
			case <-time.After(5 * time.Second):
				t.Fatal("superseded reload did not finish")
			}
			if registry.CurrentConfig() != active || server.service.GetRouter() != previous {
				t.Fatal("old warmup replaced the active A generation")
			}
			if got := candidateCloses.Load(); got != 1 {
				t.Fatalf("superseded candidate close count = %d, want 1", got)
			}
		})
	}
}
