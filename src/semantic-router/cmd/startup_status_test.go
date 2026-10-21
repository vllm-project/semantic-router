//go:build !windows

package main

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestStartupPublisherUsesRegistryIdentityAndStops(t *testing.T) {
	store := miniredis.RunT(t)
	cfg := &config.RouterConfig{}
	cfg.StartupStatus = config.StartupStatusConfig{StoreBackend: "redis", Redis: &config.StartupStatusRedisConfig{Address: store.Addr()}}
	registry := routerruntime.NewRegistry(cfg)
	writer, closeWriter := newStartupWriter(cfg, filepath.Join(t.TempDir(), "config.yaml"), registry)
	t.Cleanup(func() { require.NoError(t, closeWriter()) })
	client := redis.NewClient(&redis.Options{Addr: store.Addr()})
	t.Cleanup(func() { _ = client.Close() })
	got, err := startupstatus.LoadFromRedis(client, registry.InstanceID())
	require.NoError(t, err)
	require.Equal(t, "starting", got.Phase)
	require.False(t, got.Ready)
	require.Equal(t, registry.Status().InstanceID, got.InstanceID)
	require.Equal(t, got.InstanceID, registry.StartupState().InstanceID)
	require.NoError(t, writer.Write(startupstatus.State{Ready: true, Phase: "ready"}))
	require.NoError(t, closeWriter())
	store.FastForward(startupstatus.DefaultRedisTTL + time.Second)
	_, err = startupstatus.LoadFromRedis(client, registry.InstanceID())
	require.ErrorIs(t, err, redis.Nil)
	require.ErrorIs(t, writer.Write(startupstatus.State{}), context.Canceled)
}
