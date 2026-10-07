package extproc

import (
	"context"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
)

// TestRouterStartupValidatesReferenceMemoryConfig runs a copy of the reference
// config through the router's loader and then router construction. An unknown
// hybrid_mode fails at load. An unknown reflection algorithm passes the loader
// and fails construction, which must roll back what it had already built.
func TestRouterStartupValidatesReferenceMemoryConfig(t *testing.T) {
	cases := []struct {
		name         string
		from, to     string
		wantLoadErr  string
		wantBuildErr string
	}{
		{name: "unchanged"},
		{
			name: "hybrid_mode_rerank",
			from: "hybrid_mode: weighted", to: "hybrid_mode: rerank",
			wantLoadErr: `global.stores.memory.hybrid_mode "rerank" is not supported`,
		},
		{
			name: "algorithm_recency_semantic",
			from: "algorithm: heuristic", to: "algorithm: recency_semantic",
			wantBuildErr: `global.stores.memory.reflection.algorithm "recency_semantic" is not a registered memory filter`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			loaded, err := loadRouterConfig(writeReferenceConfigCopy(t, tc.from, tc.to))
			if tc.wantLoadErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantLoadErr)
				assert.Contains(t, err.Error(), `use "weighted"`)
				return
			}
			require.NoError(t, err)

			sessionStore := stubRouterSessionStateStore(t)
			router, err := buildOpenAIRouterFromConfig(memoryStartupBuildConfig(loaded))
			if tc.wantBuildErr != "" {
				require.Error(t, err)
				assert.Nil(t, router)
				assert.Contains(t, err.Error(), tc.wantBuildErr)
				assert.Contains(t, err.Error(), `use "heuristic"`)
				assert.Equal(t, int32(1), sessionStore.closeCalls.Load(),
					"construction must roll back the resources it built before the memory runtime")
				return
			}
			require.NoError(t, err)
			require.NotNil(t, router)
			assert.Equal(t, int32(0), sessionStore.closeCalls.Load())
			require.NoError(t, router.Close())
			assert.Equal(t, int32(1), sessionStore.closeCalls.Load())
		})
	}
}

// writeReferenceConfigCopy writes config/config.yaml to a temp file, replacing
// every occurrence of from with to. The reference config sets each value in the
// global memory block and in a decision's memory plugin.
func writeReferenceConfigCopy(t *testing.T, from, to string) string {
	t.Helper()
	_, sourceFile, _, ok := runtime.Caller(0)
	require.True(t, ok)
	data, err := os.ReadFile(filepath.Join(filepath.Dir(sourceFile), "../../../../config/config.yaml"))
	require.NoError(t, err)
	text := string(data)
	if from != "" {
		require.Equal(t, 2, strings.Count(text, from), "reference config should set %q globally and in one decision", from)
		text = strings.ReplaceAll(text, from, to)
	}
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(text), 0o600))
	return path
}

// memoryStartupBuildConfig keeps the loaded memory settings on a config the
// test can construct without model assets or a reachable memory backend.
// Memory stays disabled because the algorithm check runs before enablement.
// A stubbed Redis session store is built before the memory runtime, so its
// close count shows whether a failed construction rolled back.
func memoryStartupBuildConfig(loaded *config.RouterConfig) *config.RouterConfig {
	cfg := &config.RouterConfig{}
	cfg.Memory = loaded.Memory
	cfg.Memory.Enabled = false
	for _, decision := range loaded.AllRoutingDecisions() {
		plugin := decision.GetMemoryConfig()
		if plugin == nil {
			continue
		}
		payload := map[string]interface{}{"enabled": false, "hybrid_mode": plugin.HybridMode}
		if plugin.Reflection != nil {
			payload["reflection"] = map[string]interface{}{"algorithm": plugin.Reflection.Algorithm}
		}
		cfg.Decisions = append(cfg.Decisions, config.Decision{
			Name:    decision.Name,
			Rules:   config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{}},
			Plugins: []config.DecisionPlugin{{Type: config.DecisionPluginMemory, Configuration: config.MustStructuredPayload(payload)}},
		})
	}
	cfg.RouterLearning.StateStore = config.RouterLearningStateStoreConfig{
		Backend: "redis",
		Redis:   config.RouterLearningRedisStateStoreConfig{Address: "candidate.invalid:6379"},
	}
	return cfg
}

func stubRouterSessionStateStore(t *testing.T) *trackingRouterSessionStateStore {
	t.Helper()
	sessiontelemetry.ResetRouterSessionMemoryForTesting()
	store := &trackingRouterSessionStateStore{}
	original := newRouterSessionStateStore
	newRouterSessionStateStore = func(sessiontelemetry.RedisRouterSessionStoreConfig) (sessiontelemetry.RouterSessionStateStore, error) {
		return store, nil
	}
	t.Cleanup(func() {
		newRouterSessionStateStore = original
		sessiontelemetry.SetRouterSessionStateStore(nil)
		sessiontelemetry.ResetRouterSessionMemoryForTesting()
	})
	return store
}

// velaEmbeddingViews are the widths Vela Embedding's card advertises
// (MATRYOSHKA_DIMENSIONS in the runtime's heads/pooled.py), largest first.
var velaEmbeddingViews = []int{768, 512, 256, 128, 64}

// servedViews is a prepared provider whose card advertises views.
type servedViews struct {
	embedding.Provider
	views []int
}

func (p servedViews) EmbeddingInfo() embedding.ModelInfo {
	return embedding.ModelInfo{Dimension: p.Dimension(), Dimensions: p.views}
}

// TestReferenceMemoryStoresUseAWidthVelaEmbeddingServes sizes every memory
// backend of the reference config against Vela Embedding's card. A width the
// model doesn't serve fails the store's sizing, and the router then starts
// with memory disabled and only a warning in its log.
func TestReferenceMemoryStoresUseAWidthVelaEmbeddingServes(t *testing.T) {
	loaded, err := loadRouterConfig(writeReferenceConfigCopy(t, "", ""))
	require.NoError(t, err)
	require.True(t, loaded.Memory.Enabled)
	require.Equal(t, string(memory.EmbeddingModelMMBERT), detectMemoryEmbeddingModel(loaded))
	require.NotNil(t, loaded.Memory.Valkey)
	require.NotNil(t, loaded.Memory.Qdrant)

	base, err := embedding.NewFuncProvider(config.EmbeddingBackendModelRuntime, velaEmbeddingViews[0], func(context.Context, string) ([]float32, error) {
		return nil, assert.AnError
	})
	require.NoError(t, err)
	prepared := memory.EmbeddingConfig{Model: memory.EmbeddingModelMMBERT, Provider: servedViews{base, velaEmbeddingViews}}
	for backend, configured := range map[string]int{
		"milvus": loaded.Memory.Milvus.Dimension,
		"valkey": loaded.Memory.Valkey.Dimension,
		"qdrant": loaded.Memory.Qdrant.Dimension,
	} {
		width, err := memory.StorageDimension(configured, prepared)
		require.NoError(t, err, backend)
		assert.Equal(t, configured, width, backend)
	}

	var found bool
	for _, requirement := range config.EmbeddingRequirements(loaded, string(memory.EmbeddingModelMMBERT), true) {
		if requirement.Consumer == "memory" {
			found = true
			assert.Equal(t, string(memory.EmbeddingModelMMBERT), requirement.Model)
			assert.Contains(t, velaEmbeddingViews, requirement.Dimension)
		}
	}
	assert.True(t, found, "the reference config's memory store needs a prepared embedding")
}
