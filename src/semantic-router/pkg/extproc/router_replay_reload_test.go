package extproc

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestReplayRetentionFollowsActivatedReload(t *testing.T) {
	for _, tc := range []struct {
		name         string
		limit        int
		superseded   bool
		unopenedFile bool
		wantRecords  []string
	}{
		{"same-limit-superseded", 5, true, false, []string{"5", "4", "3", "2", "1"}},
		{"lower-limit-activated", 2, false, false, []string{"5", "4"}},
		{"lower-limit-superseded/shared-file", 2, true, false, []string{"5", "4", "3", "2", "1"}},
		{"lower-limit-superseded/unopened-file", 2, true, true, []string{"5", "4", "3", "2", "1"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			defer stubReloadSeams(t)()
			dir := t.TempDir()
			configPath := filepath.Join(dir, "router.yaml")
			replayPath := filepath.Join(replayStateDir(dir), "router-replay.jsonl")
			saved, err := store.OpenFileStore(replayPath, 5, 0)
			require.NoError(t, err)
			t.Cleanup(func() { _ = saved.Close() })
			now := time.Now()
			for i, id := range []string{"1", "2", "3", "4", "5"} {
				_, addErr := saved.Add(t.Context(), store.Record{
					ID: id, Timestamp: now.Add(time.Duration(i) * time.Millisecond), LifecycleState: store.LifecycleCompleted,
				})
				require.NoError(t, addErr)
			}
			require.NoError(t, saved.Close())

			makeConfig := func(backend string, limit int) *config.RouterConfig {
				return withGlobalPlugins(t, &config.RouterConfig{
					ConfigBaseDir: dir,
					RouterReplay:  config.RouterReplayConfig{StoreBackend: backend},
				}, replayTestPlugin(map[string]interface{}{
					"enabled": true, "prompt_capture": "none", "max_records": limit,
				}))
			}
			makeRouter := func(cfg *config.RouterConfig) *OpenAIRouter {
				recorders, fallback, shared, runtimeErr := createReplayRuntime(cfg)
				require.NoError(t, runtimeErr)
				require.NotNil(t, fallback)
				resources := newResourceScope()
				resources.add(func() error { return closeReplayRecorders(fallback, recorders, shared) })
				return &OpenAIRouter{Config: cfg, ReplayRecorders: recorders, ReplayRecorder: fallback, ReplayStoreShared: shared, resources: resources}
			}

			backend := "file"
			if tc.unopenedFile {
				backend = "memory"
			}
			active := makeConfig(backend, 5)
			writeReloadTestDocument(t, configPath, "active", active)
			previous := makeRouter(active)
			server := &Server{runtime: routerruntime.NewRegistry(active), service: NewRouterService(previous)}
			t.Cleanup(func() { _ = server.service.Close() })
			if !tc.unopenedFile {
				require.Equal(t, []string{"5", "4", "3", "2", "1"}, replayHistoryIDs(previous.ReplayRecorder.ListAllRecords()))
			}

			candidate := makeConfig("file", tc.limit)
			writeReloadTestDocument(t, configPath, "candidate", candidate)
			parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
			ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
			buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
				router := makeRouter(cfg)
				require.True(t, router.ReplayStoreShared)
				if tc.superseded {
					writeReloadTestDocument(t, configPath, "active", active)
				}
				return router, nil
			}

			err = server.reloadRouterFromFile(configPath)
			serving := server.service.GetRouter()
			if tc.superseded {
				require.ErrorIs(t, err, errConfigReloadSuperseded)
				require.Same(t, previous, serving)
			} else {
				require.NoError(t, err)
				require.Same(t, candidate, serving.Config)
			}
			if !tc.unopenedFile {
				liveIDs := replayHistoryIDs(serving.ReplayRecorder.ListAllRecords())
				require.Equal(t, tc.wantRecords, liveIDs, "history in the serving router")
			}
			require.NoError(t, server.service.Close())

			reopened, err := store.OpenFileStore(replayPath, 5, 0)
			require.NoError(t, err)
			t.Cleanup(func() { _ = reopened.Close() })
			records, err := reopened.List(t.Context())
			require.NoError(t, err)
			require.Equal(t, tc.wantRecords, replayHistoryIDs(records), "history after reopening the file")
		})
	}
}

func replayHistoryIDs(records []store.Record) []string {
	ids := make([]string, len(records))
	for i, record := range records {
		ids[i] = record.ID
	}
	return ids
}
