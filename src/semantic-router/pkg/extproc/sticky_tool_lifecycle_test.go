package extproc

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontools"
)

// stickyCorpusDocument enables supported sticky filter-mode selection on the
// corpus recipe's decision.
func stickyCorpusDocument(t *testing.T) string {
	t.Helper()
	document := parityCorpusDocument(t)
	anchor := "            - model: general-small\n              use_reasoning: false\n\nentrypoints:"
	require.Contains(t, document, anchor)
	return strings.Replace(document, anchor, `            - model: general-small
              use_reasoning: false
          plugins:
            - type: tool_selection
              configuration:
                enabled: true
                mode: filter
                sticky:
                  enabled: true
            - type: tools
              configuration:
                enabled: true
                mode: passthrough
                trusted_facts:
                  enabled: true
                  enforcement: authoritative
                  trust_sources: [operator-policy]
                  stage_roles: [candidate, final]

entrypoints:`, 1)
}

// stickyCorpusConfig parses document and binds the generation's embeddings
// to a local OpenAI-compatible endpoint, so filter mode builds its tool
// embedder without the model runtime.
func stickyCorpusConfig(t *testing.T, document string) *config.RouterConfig {
	t.Helper()
	endpoint := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var input struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&input)
		data := make([]map[string]any, len(input.Input))
		for i := range data {
			data[i] = map[string]any{"index": i, "embedding": []float32{1, 0}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"data": data})
	}))
	t.Cleanup(endpoint.Close)
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	cfg.EmbeddingConfig = config.HNSWConfig{ModelType: "bert", TargetDimension: 2}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"global": {Provider: "http", ExternalModel: "global"}}
	cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: "global", Adapter: "openai_compatible", Contract: "embedding.v1"}}
	cfg.ExternalModels = []config.ExternalModelConfig{{Name: "global", ModelName: "global", ModelEndpoint: config.ClassifierVLLMEndpoint{Address: endpoint.URL}}}
	return cfg
}

func newStickyRouterServer(t *testing.T, cfg *config.RouterConfig) *Server {
	t.Helper()
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	server := &Server{configPath: filepath.Join(t.TempDir(), "config.yaml"), runtime: routerruntime.NewRegistry(cfg)}
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	require.NoError(t, err)
	router.nameSignals(snapshot.ComponentKey(configsnapshot.ComponentSignals))
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	t.Cleanup(func() { _ = server.service.Close() })
	return server
}

func seedStickyState(t *testing.T, runtime *stickyToolRuntime) error {
	t.Helper()
	identity := sessiontools.ToolIdentity{Name: "weather", DefinitionFingerprint: "definition"}
	_, err := runtime.manager.Update(context.Background(), sessiontools.UpdateRequest{
		Key:   "lifecycle-key",
		Quota: sessiontools.QuotaKey{Principal: "principal", Namespace: "support"},
		Selection: sessiontools.SelectionInput{
			Eligible: []sessiontools.ToolIdentity{identity},
			Ranked:   []sessiontools.RankedTool{{ToolIdentity: identity, Score: 1}},
			Bounds: sessiontools.SelectionBounds{
				MaxTools: 4, MaxNewToolsPerTurn: 2, PinCalledTools: true, MaxStateBytes: runtime.maxStateBytes,
			},
			Fingerprints: sessiontools.Fingerprints{Policy: "policy", Catalog: "catalog", Capability: "capability"},
		},
	})
	return err
}

// A configuration without sticky decisions builds no sticky store.
func TestStickyRuntimeAbsentWhenDisabled(t *testing.T) {
	_, stores := installStickyTestRuntime(t)
	server := newRealRouterServer(t, parityCorpusDocument(t))
	require.Nil(t, server.service.GetRouter().stickyTools)
	require.Empty(t, *stores, "a disabled configuration must allocate no sticky resources")
}

// The store belongs to its generation: a reload starts an empty store while
// an in-flight request keeps the old one, the old store closes once that
// request drains, and shutdown closes the current one.
func TestStickyRuntimeFollowsGenerationLifecycle(t *testing.T) {
	_, stores := installStickyTestRuntime(t)
	document := stickyCorpusDocument(t)
	server := newStickyRouterServer(t, stickyCorpusConfig(t, document))
	first := server.service.GetRouter()
	require.NotNil(t, first.stickyTools)
	require.Len(t, *stores, 1)
	require.NoError(t, seedStickyState(t, first.stickyTools))

	lease, err := server.service.Pin()
	require.NoError(t, err)
	reloaded := stickyCorpusConfig(t, strings.Replace(document, "127.0.0.1:18000", "127.0.0.1:18100", 1))
	require.NoError(t, server.reloadRouterFromConfig("kubernetes", server.configPath, reloaded))
	second := server.service.GetRouter()
	require.NotSame(t, first, second)
	require.Len(t, *stores, 2, "a reload builds its own store")
	fresh, err := (*stores)[1].Load(context.Background(), "lifecycle-key")
	require.NoError(t, err)
	require.False(t, fresh.Found, "a new generation starts with empty sticky state")

	require.NoError(t, seedStickyState(t, first.stickyTools), "an in-flight request keeps its generation's store")
	lease.Release()
	waitFor(t, func() bool {
		_, loadErr := (*stores)[0].Load(context.Background(), "lifecycle-key")
		return errors.Is(loadErr, sessiontools.ErrStoreClosed)
	})
	require.ErrorIs(t, seedStickyState(t, first.stickyTools), sessiontools.ErrStoreClosed)

	require.NoError(t, server.service.Close())
	_, err = (*stores)[1].Load(context.Background(), "lifecycle-key")
	require.ErrorIs(t, err, sessiontools.ErrStoreClosed, "shutdown closes the current generation's store")
}

// A generation whose build fails after the store exists closes it, and an
// unsupported configuration fails before allocating one.
func TestStickyRuntimeConstructionFailureReleasesStore(t *testing.T) {
	_, stores := installStickyTestRuntime(t)
	cfg := stickyCorpusConfig(t, stickyCorpusDocument(t))
	var err error

	invalidTimeout := 0
	cfg.ToolSessions = &config.ToolSessionStoreConfig{TimeoutMs: &invalidTimeout}
	_, err = buildOpenAIRouterFromConfig(cfg)
	require.Error(t, err)
	require.Len(t, *stores, 1)
	_, err = (*stores)[0].Load(context.Background(), "any")
	require.ErrorIs(t, err, sessiontools.ErrStoreClosed, "a failed build must close the store it allocated")

	cfg.ToolSessions = &config.ToolSessionStoreConfig{
		Backend: config.ToolSessionStoreBackendRedis,
		Redis:   &config.ToolSessionRedisConfig{Address: "127.0.0.1:6379"},
	}
	_, err = buildOpenAIRouterFromConfig(cfg)
	require.ErrorIs(t, err, config.ErrToolSelectionStickyUnsupported)
	require.Len(t, *stores, 1, "an unsupported configuration must not allocate a store")
}

// After its generation closes, a request still holding it uses the current
// stateless selection instead of failing.
func TestStickyClosedStoreFallsBackToStatelessSelection(t *testing.T) {
	h := newStickyHarness(t, "openai.chat.v1", stickyAddDecision(t, &config.StickyToolSelectionConfig{MaxTools: intPtr(3)}), nil)
	h.run(stickyTurn{query: stickyTestQueries["weather"]})
	require.NoError(t, h.router.Close())
	result := h.run(stickyTurn{query: stickyTestQueries["email"]})
	require.Equal(t, []string{"email"}, result.tools)
	require.Equal(t, stickyToolOutcomeStateless, result.receipt.Outcome)
	require.Equal(t, stickyToolReasonStoreClosed, result.receipt.Reason)
}
