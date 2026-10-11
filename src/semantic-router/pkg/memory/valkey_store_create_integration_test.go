//go:build !windows

package memory

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	glide "github.com/valkey-io/valkey-glide/go/v2"
)

// Store's create must be all-or-nothing server-side: a failed or ambiguous
// write may not strand an ID-only partial record, a retry of the same call
// must recognize its own completed write, and a separate create attempt for
// an existing ID must fail as a duplicate without overwriting. These tests
// drive the create path with explicit operation IDs (what a Store call
// generates and reuses across its internal retries).

func valkeyMemoryFixture(id, content string) *Memory {
	return &Memory{
		ID:      id,
		Content: content,
		UserID:  "user-a",
		Type:    MemoryTypeSemantic,
	}
}

// StorageIntegration: valkey
func TestValkeyStoreInteg_StoreAmbiguousRetryRecognizesOwnWrite(t *testing.T) {
	store, client := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	fields, err := valkeyBuildHashFields(valkeyMemoryFixture("retry-own", "first attempt"), make([]float32, 384))
	require.NoError(t, err)

	// The first attempt applies (the response may or may not reach the
	// caller), then the same call retries with its own operation ID.
	require.NoError(t, store.createMemory(ctx, "retry-own", fields, "op-a"))
	require.NoError(t, store.createMemory(ctx, "retry-own", fields, "op-a"))

	stored, err := store.Get(ctx, "retry-own")
	require.NoError(t, err)
	require.Equal(t, "first attempt", stored.Content)
	require.Equal(t, "op-a", hgetField(t, client, store.hashKey("retry-own"), valkeyCreateOperationField))
}

// StorageIntegration: valkey
func TestValkeyStoreInteg_StoreDuplicateIDFailsWithoutOverwrite(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	fields, err := valkeyBuildHashFields(valkeyMemoryFixture("dup-id", "original content"), make([]float32, 384))
	require.NoError(t, err)
	require.NoError(t, store.createMemory(ctx, "dup-id", fields, "op-a"))

	// A separate create attempt for the existing ID fails and leaves the
	// original record intact.
	other, err := valkeyBuildHashFields(valkeyMemoryFixture("dup-id", "other caller's content"), make([]float32, 384))
	require.NoError(t, err)
	err = store.createMemory(ctx, "dup-id", other, "op-b")
	require.ErrorIs(t, err, errValkeyMemoryAlreadyExists)

	stored, err := store.Get(ctx, "dup-id")
	require.NoError(t, err)
	require.Equal(t, "original content", stored.Content)
}

// StorageIntegration: valkey
func TestValkeyStoreInteg_StoreConcurrentCreators(t *testing.T) {
	store, client := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	const creators = 8
	errs := make([]error, creators)
	var wg sync.WaitGroup
	wg.Add(creators)
	for i := range creators {
		go func(i int) {
			defer wg.Done()
			fields, err := valkeyBuildHashFields(
				valkeyMemoryFixture("race-id", fmt.Sprintf("creator-%d content", i)), make([]float32, 384))
			if err != nil {
				errs[i] = err
				return
			}
			errs[i] = store.createMemory(ctx, "race-id", fields, fmt.Sprintf("op-%d", i))
		}(i)
	}
	wg.Wait()

	wins, duplicates := 0, 0
	for _, err := range errs {
		switch {
		case err == nil:
			wins++
		case errors.Is(err, errValkeyMemoryAlreadyExists):
			duplicates++
		default:
			t.Fatalf("unexpected create error: %v", err)
		}
	}
	require.Equal(t, 1, wins, "exactly one creator may win")
	require.Equal(t, creators-1, duplicates)

	// The winner's record is complete, not an ID-only partial.
	stored, err := store.Get(ctx, "race-id")
	require.NoError(t, err)
	require.Equal(t, "user-a", stored.UserID)
	require.NotEmpty(t, stored.Content)
	require.NotZero(t, stored.CreatedAt)
	require.NotEmpty(t, hgetField(t, client, store.hashKey("race-id"), "content"))
}

// StorageIntegration: valkey
func TestValkeyStoreInteg_StoreLegacyPartialRecordStaysDuplicate(t *testing.T) {
	store, client := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	// A hash left by a pre-atomic create (reservation applied, full write
	// lost) is an existing record: a new create reports a duplicate instead
	// of silently overwriting it. Removing it is an explicit Forget.
	_, err := client.HSet(ctx, store.hashKey("legacy-partial"), map[string]string{"id": "legacy-partial"})
	require.NoError(t, err)

	fields, err := valkeyBuildHashFields(valkeyMemoryFixture("legacy-partial", "new content"), make([]float32, 384))
	require.NoError(t, err)
	err = store.createMemory(ctx, "legacy-partial", fields, "op-new")
	require.ErrorIs(t, err, errValkeyMemoryAlreadyExists)

	stored, err := store.Get(ctx, "legacy-partial")
	require.NoError(t, err)
	require.Empty(t, stored.Content, "the partial record must not gain content")
}

// TestValkeyStoreInteg_StoreAmbiguousRetryEndToEnd drives the public Store
// path so the integration covers validation, embedding, and the create script
// together; the second call is a separate create for an existing ID.
// StorageIntegration: valkey
func TestValkeyStoreInteg_StoreAmbiguousRetryEndToEnd(t *testing.T) {
	store, _ := setupValkeyMemoryIntegration(t)
	ctx := context.Background()

	mem := valkeyMemoryFixture("e2e-create", "the full Store path")
	require.NoError(t, store.Store(ctx, mem))
	err := store.Store(ctx, valkeyMemoryFixture("e2e-create", "another attempt"))
	require.ErrorIs(t, err, errValkeyMemoryAlreadyExists)

	stored, err := store.Get(ctx, "e2e-create")
	require.NoError(t, err)
	require.Equal(t, "the full Store path", stored.Content)
}

func hgetField(t *testing.T, client *glide.Client, key, field string) string {
	t.Helper()
	value, err := client.HGet(context.Background(), key, field)
	require.NoError(t, err)
	return value.Value()
}
