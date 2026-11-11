package extproc

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responsestore"
)

func TestGetInputItemsStableIDs(t *testing.T) {
	for _, instructions := range []string{"be helpful", ""} {
		name := "with_instructions"
		if instructions == "" {
			name = "without_instructions"
		}
		t.Run(name, func(t *testing.T) {
			store, err := responsestore.NewMemoryStore(responsestore.StoreConfig{Enabled: true})
			require.NoError(t, err)
			t.Cleanup(func() { require.NoError(t, store.Close()) })
			stored := &responseapi.StoredResponse{
				ID:           "resp_stored",
				Object:       "response",
				Status:       "completed",
				Instructions: instructions,
				Input: []responseapi.InputItem{{
					ID: "msg_probe", Type: "message", Role: "user", Content: json.RawMessage(`"hello"`),
				}},
			}
			require.NoError(t, store.StoreResponse(t.Context(), stored))
			before, err := json.Marshal(stored)
			require.NoError(t, err)
			readOnly := &inputItemsReadOnlyStore{ResponseStore: store, t: t}
			filter := NewResponseAPIFilter(readOnly)
			first := getInputItemsList(t, filter, stored.ID)
			second := getInputItemsList(t, filter, stored.ID)
			recreated := getInputItemsList(t, NewResponseAPIFilter(readOnly), stored.ID)
			after, err := store.GetResponse(t.Context(), stored.ID)
			require.NoError(t, err)
			afterJSON, err := json.Marshal(after)
			require.NoError(t, err)
			require.JSONEq(t, string(before), string(afterJSON), "GET must not mutate the stored response")
			require.Equal(t, first.FirstID, second.FirstID, "first_id must remain stable across GETs")
			require.Equal(t, first, second, "unchanged input items must retain their IDs across GETs")
			require.Equal(t, first, recreated, "item identity must survive filter recreation")
			require.Equal(t, "list", first.Object)
			require.False(t, first.HasMore)
			require.Equal(t, "msg_probe", first.LastID)
			if instructions == "" {
				require.Equal(t, stored.Input, first.Data)
				require.Equal(t, "msg_probe", first.FirstID)
				return
			}
			require.Len(t, first.Data, 2)
			require.Equal(t, stored.Input, first.Data[1:])
			instruction := first.Data[0]
			require.NotEmpty(t, instruction.ID)
			require.NotEqual(t, "msg_probe", instruction.ID)
			require.Equal(t, instruction.ID, first.FirstID)
			require.Equal(t, "message", instruction.Type)
			require.Equal(t, "system", instruction.Role)
			require.Equal(t, "completed", instruction.Status)
			require.JSONEq(t, `[{"type":"input_text","text":"be helpful"}]`, string(instruction.Content))
		})
	}
}

func TestGetInputItemsLegacyRedisResponse(t *testing.T) {
	redis := miniredis.RunT(t)
	store, err := responsestore.NewRedisStore(responsestore.StoreConfig{
		Enabled: true,
		Redis: responsestore.RedisStoreConfig{
			Address: redis.Addr(), KeyPrefix: "input-items:",
		},
	})
	require.NoError(t, err)
	t.Cleanup(func() { require.NoError(t, store.Close()) })
	readOnly := &inputItemsReadOnlyStore{ResponseStore: store, t: t}
	var instructionIDs []string
	for _, id := range []string{"resp_legacy_one", "resp_legacy_two"} {
		// Seed the existing JSON schema directly, bypassing current ingestion and
		// store writes. Old records contain no persisted instruction item ID.
		raw := `{"id":"` + id + `","object":"response","status":"completed","instructions":"be helpful","input":[]}`
		key := "input-items:" + responsestore.ResponseKeyPrefix + id
		require.NoError(t, redis.Set(key, raw))
		first := getInputItemsList(t, NewResponseAPIFilter(readOnly), id)
		second := getInputItemsList(t, NewResponseAPIFilter(readOnly), id)
		require.Equal(t, first.FirstID, second.FirstID, "old records must have stable instruction IDs without migration")
		require.Equal(t, first, second)
		require.Len(t, first.Data, 1, "an instructions-only response must retain its system message")
		require.NotEmpty(t, first.Data[0].ID)
		require.Equal(t, "system", first.Data[0].Role)
		require.JSONEq(t, `[{"type":"input_text","text":"be helpful"}]`, string(first.Data[0].Content))
		require.Equal(t, first.Data[0].ID, first.FirstID)
		require.Equal(t, first.FirstID, first.LastID)
		instructionIDs = append(instructionIDs, first.FirstID)
		after, err := redis.Get(key)
		require.NoError(t, err)
		require.Equal(t, raw, after, "GET must leave persisted JSON unchanged")
	}
	require.NotEqual(t, instructionIDs[0], instructionIDs[1], "identical instructions in different responses need distinct IDs")
}

// Reject writes even if they would leave the stored value unchanged.
type inputItemsReadOnlyStore struct {
	responsestore.ResponseStore
	t *testing.T
}

func (s *inputItemsReadOnlyStore) StoreResponse(context.Context, *responseapi.StoredResponse) error {
	s.t.Fatal("GET input_items must not call StoreResponse")
	return nil
}

func (s *inputItemsReadOnlyStore) UpdateResponse(context.Context, *responseapi.StoredResponse) error {
	s.t.Fatal("GET input_items must not call UpdateResponse")
	return nil
}

func (s *inputItemsReadOnlyStore) DeleteResponse(context.Context, string) error {
	s.t.Fatal("GET input_items must not call DeleteResponse")
	return nil
}

func getInputItemsList(t *testing.T, filter *ResponseAPIFilter, id string) responseapi.InputItemsListResponse {
	t.Helper()
	response, err := filter.HandleGetInputItems(t.Context(), id)
	require.NoError(t, err)
	require.EqualValues(t, 200, response.GetImmediateResponse().GetStatus().GetCode())
	var list responseapi.InputItemsListResponse
	require.NoError(t, json.Unmarshal(response.GetImmediateResponse().GetBody(), &list))
	return list
}
