package extproc

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// A Router started from a configuration file without a router_replay plugin
// or a replay store keeps its routing records in .vllm-sr/router-replay.jsonl
// beside that file, owner-only and without the conversation's text, and the
// next Router started from it lists them. Where a plugin keeps text, or that
// directory cannot be written, the records stay in memory, as before.
func TestReplayDefaultRecordSurvivesARestart(t *testing.T) {
	dir := t.TempDir()
	cfg := parseReplayConfig(t, dir, replayDefaultConfig)
	require.Equal(t, "file", cfg.RouterReplay.StoreBackend)
	restarted, id := restartReplayRouter(t, cfg)
	record, found := restarted.ReplayRecorder.GetRecord(id)
	require.True(t, found, "the default record did not survive the restart")
	require.Equal(t, routerreplay.LifecycleCompleted, record.LifecycleState)
	require.NotNil(t, record.TotalTokens)
	require.Equal(t, 3, *record.TotalTokens)
	path := filepath.Join(dir, ".vllm-sr", "router-replay.jsonl")
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0o600), info.Mode().Perm())
	stored, err := os.ReadFile(path)
	require.NoError(t, err)
	for _, text := range append(append([]string{}, replayRequestTexts...), replayResponseTexts...) {
		require.NotContains(t, string(stored), text)
	}

	raw := t.TempDir()
	cfg = parseReplayConfig(t, raw, `version: v0.5
global:
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        prompt_capture: raw
`)
	restarted, id = restartReplayRouter(t, cfg)
	_, found = restarted.ReplayRecorder.GetRecord(id)
	require.False(t, found, "a record that keeps text was kept on disk")
	require.NoFileExists(t, filepath.Join(raw, ".vllm-sr", "router-replay.jsonl"))

	readOnly := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(readOnly, ".vllm-sr"), nil, 0o600))
	restarted, id = restartReplayRouter(t, parseReplayConfig(t, readOnly, replayDefaultConfig))
	_, found = restarted.ReplayRecorder.GetRecord(id)
	require.False(t, found)
}

// The file store keeps at most replayFileMaxRecords records, a file of about
// 8 MB, while max_records keeps its default of 10,000 for the memory store.
func TestReplayFileStoreKeepsAtMostAThousandRecords(t *testing.T) {
	cfg := parseReplayConfig(t, t.TempDir(), replayDefaultConfig)
	policy := cfg.EffectiveRouterReplayConfig(nil)
	require.Equal(t, 10000, policy.MaxRecords)
	storage := createReplayFileStore(cfg, policy)
	t.Cleanup(func() { _ = storage.Close() })
	ctx := context.Background()
	for i := 0; i <= 1000; i++ {
		_, err := storage.Add(ctx, store.Record{ID: strconv.Itoa(i), Timestamp: time.Now(), LifecycleState: store.LifecycleCompleted})
		require.NoError(t, err)
	}
	records, err := storage.List(ctx)
	require.NoError(t, err)
	require.Len(t, records, 1000)
}

// A passthrough request runs global.plugins, so where they keep raw text the
// records stay in memory, though a recipe sets prompt_capture: none.
func TestReplayFileStoreHoldsNoTextOfAPassthroughRequest(t *testing.T) {
	dir := t.TempDir()
	cfg := parseReplayConfig(t, dir, `version: v0.5
global:
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        prompt_capture: raw
recipes:
  - name: main
    plugins:
      - type: router_replay
        configuration:
          prompt_capture: none
`)
	router := replayRuntimeRouter(t, cfg)
	ctx := replayConversationRequest(false)
	router.startRouterReplay(ctx, "public-model", "public-model", "")
	replayConversationReply(t, router, ctx, false)
	require.NotEmpty(t, ctx.RouterReplayID)
	require.NoFileExists(t, filepath.Join(dir, ".vllm-sr", "router-replay.jsonl"))
}

// With replay off in global.plugins and on in a recipe, a request of that
// recipe that matches no decision is recorded on the file store, as on the
// memory store.
func TestReplayFileStoreRecordsARecipeThatEnablesReplay(t *testing.T) {
	cfg := parseReplayConfig(t, t.TempDir(), `version: v0.5
providers:
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
global:
  plugins:
    - type: router_replay
      configuration:
        enabled: false
recipes:
  - name: main
  - name: audited
    plugins:
      - type: router_replay
        configuration:
          enabled: true
entrypoints:
  - model_names: [main-model]
    recipe: main
    default: true
  - model_names: [audited-model]
    recipe: audited
`)
	require.Equal(t, "file", cfg.RouterReplay.StoreBackend)
	router := replayRuntimeRouter(t, cfg)
	audited, ok := cfg.RecipeByName("audited")
	require.True(t, ok)
	ctx := replayPolicyRequest(t)
	ctx.Routing.SelectRecipe(audited)
	router.respondDecisionUnresolved(ctx, "audited-model", &decision.DecisionUnresolvedError{Decision: "unknown"})
	require.NotEmpty(t, ctx.RouterReplayID, "the audited recipe's request was not recorded")
}

// A Router that reads a generated .vllm-sr/runtime-config.yaml keeps the file
// in that directory, the project's state directory, not in one nested in it.
func TestReplayFileOfAGeneratedRuntimeConfigIsInTheProjectStateDirectory(t *testing.T) {
	t.Setenv(config.ConfigBaseDirEnv, "")
	state := filepath.Join(t.TempDir(), ".vllm-sr")
	require.NoError(t, os.MkdirAll(state, 0o700))
	path := filepath.Join(state, "runtime-config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(replayDefaultConfig), 0o600))
	cfg, err := config.Parse(path)
	require.NoError(t, err)
	storage := createReplayFileStore(cfg, cfg.EffectiveRouterReplayConfig(nil))
	t.Cleanup(func() { _ = storage.Close() })
	require.FileExists(t, filepath.Join(state, "router-replay.jsonl"))
	require.NoDirExists(t, filepath.Join(state, ".vllm-sr"))
}

// replayDefaultConfig declares no router_replay plugin and no replay store.
const replayDefaultConfig = "version: v0.5\nglobal: {}\n"

func parseReplayConfig(t *testing.T, dir, document string) *config.RouterConfig {
	t.Helper()
	path := filepath.Join(dir, "config.yaml")
	require.NoError(t, os.WriteFile(path, []byte(document), 0o600))
	cfg, err := config.Parse(path)
	require.NoError(t, err)
	return cfg
}

// restartReplayRouter records a conversation on a Router built from cfg,
// closes that Router, and builds the next one from the same configuration.
func restartReplayRouter(t *testing.T, cfg *config.RouterConfig) (*OpenAIRouter, string) {
	t.Helper()
	router := replayRuntimeRouter(t, cfg)
	ctx := replayConversationRequest(false)
	ctx.RouterReplayPluginConfig = cfg.EffectiveRouterReplayConfig(nil)
	router.startRouterReplay(ctx, "public-model", "public-model", "")
	replayConversationReply(t, router, ctx, false)
	_, found := router.ReplayRecorder.GetRecord(ctx.RouterReplayID)
	require.True(t, found)
	require.NoError(t, closeReplayRecorders(router.ReplayRecorder, router.ReplayRecorders, router.ReplayStoreShared))
	return replayRuntimeRouter(t, cfg), ctx.RouterReplayID
}
