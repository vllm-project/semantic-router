//go:build !windows

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestStartupRedisPersistenceDoesNotControlLocalReadiness(t *testing.T) {
	store := miniredis.RunT(t)
	for _, ready := range []bool{false, true} {
		registry := routerruntime.NewRegistry(nil)
		backend, err := startupstatus.NewRedisWriter(startupstatus.RedisWriterConfig{Address: store.Addr(), InstanceID: registry.InstanceID()})
		require.NoError(t, err)
		t.Cleanup(func() { require.NoError(t, backend.Close()) })
		writer := registry.StartupStatusWriter(backend)
		store.SetError("ERR unavailable")
		require.Error(t, writer.Write(startupstatus.State{Ready: ready, Phase: "test"}))
		server := &ClassificationAPIServer{runtimeRegistry: registry}
		for _, endpoint := range []string{"/ready", "/startup-status"} {
			response := httptest.NewRecorder()
			req := httptest.NewRequest(http.MethodGet, endpoint, nil)
			if endpoint == "/ready" {
				server.handleReady(response, req)
			} else {
				server.handleStartupStatus(response, req)
			}
			want := http.StatusServiceUnavailable
			if ready {
				want = http.StatusOK
			}
			require.Equal(t, want, response.Code)
			var body startupstatus.State
			require.NoError(t, json.Unmarshal(response.Body.Bytes(), &body))
			require.Equal(t, ready, body.Ready)
			if endpoint == "/startup-status" {
				require.Equal(t, registry.InstanceID(), body.InstanceID)
			}
		}
		store.SetError("")
	}
}

func TestStartupRedisWithoutIdentityDoesNotReadLegacyOrFileState(t *testing.T) {
	store := miniredis.RunT(t)
	require.NoError(t, store.Set("vllm-sr:startup-status", `{"ready":true,"phase":"ready"}`))
	path := filepath.Join(t.TempDir(), "config.yaml")
	require.NoError(t, startupstatus.NewFileWriter(path).Write(startupstatus.State{Ready: true}))
	server := &ClassificationAPIServer{configPath: path, startupStatusConfig: &config.StartupStatusConfig{
		StoreBackend: "redis", Redis: &config.StartupStatusRedisConfig{Address: store.Addr()},
	}}
	response := httptest.NewRecorder()
	server.handleStartupStatus(response, httptest.NewRequest(http.MethodGet, "/startup-status", nil))
	require.Equal(t, http.StatusServiceUnavailable, response.Code)
	require.Contains(t, response.Body.String(), `"phase":"unknown"`)
}
