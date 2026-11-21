package extproc

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

const memoryReplayDocument = `version: v0.5
providers:
  models:
    - name: m1
      backend_refs:
        - endpoint: 127.0.0.1:8000
global:
  services:
    router_replay:
      store_backend: memory
      ttl_seconds: 3600
`

// A reload of the configuration file, as a Dashboard save makes, keeps the
// records of the memory Replay store; one that changes ttl_seconds, and a
// restart, start it empty.
func TestAReloadKeepsTheMemoryReplayRecords(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	cfg := parseReplayConfig(t, dir, memoryReplayDocument)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	server := &Server{configPath: path, runtime: routerruntime.NewRegistry(cfg)}
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	require.NoError(t, err)
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	id, err := router.ReplayRecorder.AddRecord(routerreplay.RoutingRecord{ID: "before-the-save"})
	require.NoError(t, err)

	reload := func(document string) *OpenAIRouter {
		t.Helper()
		require.NoError(t, os.WriteFile(path, []byte(document), 0o600))
		require.NoError(t, server.reloadRouterFromFile(path))
		return server.service.GetRouter()
	}
	saved := reload(strings.Replace(memoryReplayDocument, "127.0.0.1:8000", "127.0.0.1:8001", 1))
	require.NotSame(t, router, saved, "the save built no new generation")
	_, found := saved.ReplayRecorder.GetRecord(id)
	require.True(t, found, "the save lost the memory store's records")

	retimed := reload(strings.Replace(memoryReplayDocument, "ttl_seconds: 3600", "ttl_seconds: 600", 1))
	_, found = retimed.ReplayRecorder.GetRecord(id)
	require.False(t, found, "a store with another TTL kept the old records")
	_, err = retimed.ReplayRecorder.AddRecord(routerreplay.RoutingRecord{ID: "before-the-restart"})
	require.NoError(t, err)

	require.NoError(t, server.service.Close())
	restarted := replayRuntimeRouter(t, parseReplayConfig(t, dir, strings.Replace(memoryReplayDocument, "ttl_seconds: 3600", "ttl_seconds: 600", 1)))
	_, found = restarted.ReplayRecorder.GetRecord("before-the-restart")
	require.False(t, found, "the records outlived every generation that held them")
}

// A decision named global keeps its memory store apart from the store of the
// requests that match no decision, so each keeps its own max_records.
func TestADecisionNamedGlobalKeepsItsOwnMemoryStore(t *testing.T) {
	cfg := withGlobalPlugins(t, &config.RouterConfig{
		RouterReplay: config.RouterReplayConfig{StoreBackend: "memory"},
		IntelligentRouting: config.IntelligentRouting{Decisions: []config.Decision{{
			Name: "global", ModelRefs: []config.ModelRef{{Model: "m"}},
			Plugins: []config.DecisionPlugin{replayTestPlugin(map[string]interface{}{"enabled": true, "max_records": 1})},
		}}},
	}, replayTestPlugin(map[string]interface{}{"enabled": true, "max_records": 3}))
	cfg.ConfigBaseDir = t.TempDir()
	router := replayRuntimeRouter(t, cfg)
	first, err := router.ReplayRecorder.AddRecord(routerreplay.RoutingRecord{ID: "first"})
	require.NoError(t, err)
	_, err = router.ReplayRecorders["global"].AddRecord(routerreplay.RoutingRecord{ID: "decision"})
	require.NoError(t, err)
	_, found := router.ReplayRecorder.GetRecord(first)
	require.True(t, found, "the decision named global evicted the fallback store's record")
}

// A reload that is rejected leaves the serving memory store as it was: the
// candidate's smaller max_records takes effect only when it activates, as an
// accepted reload shows.
func TestARejectedReloadKeepsTheMemoryStoreCapacity(t *testing.T) {
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	document := memoryReplayDocument + `  plugins:
    - type: router_replay
      configuration:
        enabled: true
        max_records: 3
`
	cfg := parseReplayConfig(t, dir, document)
	router, err := buildOpenAIRouterFromConfig(cfg)
	require.NoError(t, err)
	server := &Server{configPath: path, runtime: routerruntime.NewRegistry(cfg)}
	snapshot, err := server.configManager().Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	require.NoError(t, err)
	server.service = newRouterServiceWithSnapshot(router, snapshot)
	var ids []string
	for _, id := range []string{"one", "two", "three"} {
		kept, err := router.ReplayRecorder.AddRecord(routerreplay.RoutingRecord{ID: id})
		require.NoError(t, err)
		ids = append(ids, kept)
	}

	rejected := strings.Replace(document, "max_records: 3", "max_records: 1", 1) + `  stores:
    memory:
      reflection:
        algorithm: not-a-registered-filter
`
	require.NoError(t, os.WriteFile(path, []byte(rejected), 0o600))
	require.ErrorContains(t, server.reloadRouterFromFile(path), "not-a-registered-filter")
	require.Same(t, router, server.service.GetRouter())
	for _, id := range ids {
		_, found := router.ReplayRecorder.GetRecord(id)
		require.True(t, found, "the rejected reload dropped record %s", id)
	}

	accepted := strings.Replace(document, "max_records: 3", "max_records: 1", 1)
	require.NoError(t, os.WriteFile(path, []byte(accepted), 0o600))
	require.NoError(t, server.reloadRouterFromFile(path))
	resized := server.service.GetRouter()
	require.NotSame(t, router, resized)
	_, found := resized.ReplayRecorder.GetRecord(ids[0])
	require.False(t, found, "the accepted reload kept the old max_records")
	_, found = resized.ReplayRecorder.GetRecord(ids[2])
	require.True(t, found, "the accepted reload dropped the newest record")
}
