package memory

import (
	"context"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

func TestValkeyReplaceCurrentGroupScriptReplayReportsCommittedMerge(t *testing.T) {
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { _ = client.Close() })

	createdAt := time.UnixMilli(1_700_000_000_000).UTC()
	versions := []memoryVersion{
		versionFixture("a", "  alpha  ", createdAt),
		versionFixture("b", "beta", createdAt.Add(time.Second)),
	}
	keys := []string{"mem:a", "mem:b", "mem:summary"}
	summary := map[string]string{
		"id":      "summary",
		"content": "  merged  ",
	}
	ctx := context.Background()
	for i, want := range versions {
		seedValkeyVersionHash(t, client, keys[i], want)
	}

	require.Equal(t, int64(2), evalReplaceScript(t, client, keys, versions, summary))
	require.Equal(t, int64(0), client.Exists(ctx, "mem:a", "mem:b").Val())
	require.Equal(t, "  merged  ", client.HGet(ctx, "mem:summary", "content").Val())

	// The first reply was lost. The retry must still report the committed merge.
	require.Equal(t, int64(2), evalReplaceScript(t, client, keys, versions, summary))
	require.Equal(t, "  merged  ", client.HGet(ctx, "mem:summary", "content").Val())
}

func TestValkeyReplaceCurrentGroupScriptRejectsStaleAndCollidingWrites(t *testing.T) {
	createdAt := time.UnixMilli(1_700_000_000_000).UTC()
	versions := []memoryVersion{
		versionFixture("a", "alpha", createdAt),
		versionFixture("b", "beta", createdAt.Add(time.Second)),
	}
	keys := []string{"mem:a", "mem:b", "mem:summary"}
	summary := map[string]string{"id": "summary", "content": "merged"}

	t.Run("stale source", func(t *testing.T) {
		mr := miniredis.RunT(t)
		client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
		t.Cleanup(func() { _ = client.Close() })
		for i, want := range versions {
			seedValkeyVersionHash(t, client, keys[i], want)
		}
		require.NoError(t, client.HSet(context.Background(), "mem:a", "content", "changed").Err())

		require.Equal(t, int64(0), evalReplaceScript(t, client, keys, versions, summary))
		require.Equal(t, int64(2), client.Exists(context.Background(), "mem:a", "mem:b").Val())
		require.Equal(t, int64(0), client.Exists(context.Background(), "mem:summary").Val())
	})

	t.Run("summary id already exists", func(t *testing.T) {
		mr := miniredis.RunT(t)
		client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
		t.Cleanup(func() { _ = client.Close() })
		for i, want := range versions {
			seedValkeyVersionHash(t, client, keys[i], want)
		}
		require.NoError(t, client.HSet(context.Background(), "mem:summary", "id", "other").Err())

		require.Equal(t, int64(-1), evalReplaceScript(t, client, keys, versions, summary))
		require.Equal(t, int64(2), client.Exists(context.Background(), "mem:a", "mem:b").Val())
		require.Equal(t, "other", client.HGet(context.Background(), "mem:summary", "id").Val())
	})

	t.Run("deleted sources without this summary", func(t *testing.T) {
		mr := miniredis.RunT(t)
		client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
		t.Cleanup(func() { _ = client.Close() })

		require.Equal(t, int64(0), evalReplaceScript(t, client, keys, versions, summary))
		require.Equal(t, int64(0), client.Exists(context.Background(), "mem:summary").Val())
	})
}

func versionFixture(id, content string, createdAt time.Time) memoryVersion {
	return memoryVersion{
		id:         id,
		userID:     "user-1",
		projectID:  "project-1",
		typ:        MemoryTypeSemantic,
		content:    content,
		createdAt:  createdAt,
		updatedAt:  createdAt.Add(time.Millisecond),
		importance: 0.5,
	}
}

func seedValkeyVersionHash(t *testing.T, client *redis.Client, key string, want memoryVersion) {
	t.Helper()
	args := valkeySourceVersionArgs(want)
	names := []string{"id", "user_id", "project_id", "memory_type", "content", "created_at", "updated_at", "importance"}
	values := make([]any, 0, len(names)*2)
	for i, name := range names {
		values = append(values, name, args[i])
	}
	require.NoError(t, client.HSet(context.Background(), key, values...).Err())
}

func evalReplaceScript(t *testing.T, client *redis.Client, keys []string, versions []memoryVersion, summary map[string]string) int64 {
	t.Helper()
	args := valkeyReplaceCurrentGroupArgs(versions, summary)
	cmdArgs := make([]any, len(args))
	for i, arg := range args {
		cmdArgs[i] = arg
	}
	result, err := client.Eval(context.Background(), valkeyReplaceCurrentGroupScriptSource, keys, cmdArgs...).Result()
	require.NoError(t, err)
	n, ok := result.(int64)
	require.True(t, ok, "script result type %T (%v)", result, result)
	return n
}
