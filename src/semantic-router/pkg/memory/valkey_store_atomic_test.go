package memory

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newValkeyAtomicStoreTestClient(t *testing.T) (*miniredis.Miniredis, *redis.Client) {
	t.Helper()

	server := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: server.Addr()})
	t.Cleanup(func() { require.NoError(t, client.Close()) })
	return server, client
}

func runValkeyAtomicStoreTestCommand(ctx context.Context, client *redis.Client, command []string) (any, error) {
	args := make([]any, len(command))
	for i, arg := range command {
		args[i] = arg
	}
	return client.Do(ctx, args...).Result()
}

func TestValkeyAtomicStoreRetryIsIdempotentAndDoesNotOverwrite(t *testing.T) {
	_, client := newValkeyAtomicStoreTestClient(t)
	ctx := context.Background()
	key := "memory:retry"
	fields := map[string]string{
		"id":      "retry",
		"content": "original content",
		"user_id": "user-1",
	}
	command := valkeyBuildAtomicStoreCommand(key, fields, "operation-1")

	// The first result stands in for a successful write whose response is lost.
	result, err := runValkeyAtomicStoreTestCommand(ctx, client, command)
	require.NoError(t, err)
	assert.Equal(t, "stored", result)

	// Retrying the same command must confirm success instead of reporting a duplicate.
	result, err = runValkeyAtomicStoreTestCommand(ctx, client, command)
	require.NoError(t, err)
	require.NoError(t, valkeyAtomicStoreResult(result, "retry"))

	// A separate create attempt with the same ID is still rejected and cannot overwrite.
	duplicateFields := map[string]string{
		"id":      "retry",
		"content": "replacement content",
		"user_id": "user-1",
	}
	result, err = runValkeyAtomicStoreTestCommand(ctx, client,
		valkeyBuildAtomicStoreCommand(key, duplicateFields, "operation-2"))
	require.NoError(t, err)
	assert.ErrorIs(t, valkeyAtomicStoreResult(result, "retry"), errValkeyMemoryAlreadyExists)

	stored, err := client.HGetAll(ctx, key).Result()
	require.NoError(t, err)
	assert.Equal(t, "original content", stored["content"])
	assert.Equal(t, "operation-1", stored[valkeyStoreOperationIDField])
}

func TestValkeyAtomicStoreRejectedWriteLeavesNoReservation(t *testing.T) {
	server, client := newValkeyAtomicStoreTestClient(t)
	ctx := context.Background()
	key := "memory:malformed"
	command := valkeyBuildAtomicStoreCommand(key, map[string]string{
		"id":      "malformed",
		"content": "content",
	}, "operation-1")

	// Remove one value to force the Lua preflight to reject the command before HSET.
	command = command[:len(command)-1]
	_, err := runValkeyAtomicStoreTestCommand(ctx, client, command)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "invalid atomic memory store arguments")
	assert.False(t, server.Exists(key), "a rejected write must not leave an ID-only hash")
}

func TestValkeyAtomicStoreConcurrentSameIDHasOneCompleteWinner(t *testing.T) {
	_, client := newValkeyAtomicStoreTestClient(t)
	ctx := context.Background()
	const writers = 12
	key := "memory:concurrent"

	results := make(chan error, writers)
	var wg sync.WaitGroup
	for i := 0; i < writers; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			fields := map[string]string{
				"id":      "concurrent",
				"content": fmt.Sprintf("complete content %d", i),
				"user_id": fmt.Sprintf("user-%d", i),
			}
			result, err := runValkeyAtomicStoreTestCommand(ctx, client,
				valkeyBuildAtomicStoreCommand(key, fields, fmt.Sprintf("operation-%d", i)))
			if err != nil {
				results <- err
				return
			}
			results <- valkeyAtomicStoreResult(result, "concurrent")
		}(i)
	}
	wg.Wait()
	close(results)

	successes := 0
	duplicates := 0
	for err := range results {
		switch {
		case err == nil:
			successes++
		case assert.ErrorIs(t, err, errValkeyMemoryAlreadyExists):
			duplicates++
		default:
			t.Errorf("unexpected store error: %v", err)
		}
	}
	assert.Equal(t, 1, successes)
	assert.Equal(t, writers-1, duplicates)

	stored, err := client.HGetAll(ctx, key).Result()
	require.NoError(t, err)
	assert.Equal(t, "concurrent", stored["id"])
	assert.Contains(t, stored["content"], "complete content ")
	assert.NotEmpty(t, stored["user_id"])
	assert.NotEmpty(t, stored[valkeyStoreOperationIDField])
}
