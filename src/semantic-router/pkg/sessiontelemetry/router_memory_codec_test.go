package sessiontelemetry

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Intercept Redis commands at the client boundary without opening a connection.
type snapshotRedisHook struct {
	values map[string]string
}

func (h *snapshotRedisHook) DialHook(next redis.DialHook) redis.DialHook { return next }

func (h *snapshotRedisHook) ProcessPipelineHook(next redis.ProcessPipelineHook) redis.ProcessPipelineHook {
	return next
}

func (h *snapshotRedisHook) ProcessHook(_ redis.ProcessHook) redis.ProcessHook {
	return func(_ context.Context, cmd redis.Cmder) error {
		key := cmd.Args()[1].(string)
		switch cmd.Name() {
		case "get":
			value, ok := h.values[key]
			if !ok {
				cmd.SetErr(redis.Nil)
				return redis.Nil
			}
			cmd.(*redis.StringCmd).SetVal(value)
		case "set":
			h.values[key] = string(cmd.Args()[2].([]byte))
			cmd.(*redis.StatusCmd).SetVal("OK")
		default:
			return fmt.Errorf("unexpected Redis command %q", cmd.Name())
		}
		return nil
	}
}

func newSnapshotRedisTestStore(t *testing.T) (*redisRouterSessionStore, *snapshotRedisHook) {
	t.Helper()
	hook := &snapshotRedisHook{values: make(map[string]string)}
	client := redis.NewClient(&redis.Options{Addr: "unused.invalid:6379"})
	client.AddHook(hook)
	t.Cleanup(func() { _ = client.Close() })
	return &redisRouterSessionStore{client: client, timeout: time.Second, keyPrefix: "test:"}, hook
}

func TestRedisSessionSnapshotRejectsLegacyIdentityEncoding(t *testing.T) {
	store, hook := newSnapshotRedisTestStore(t)
	// The old codec persisted literal percent-encoded client text unchanged.
	// The new codec gives this same key to a different raw identity, "a/b".
	legacy := RouterSessionSnapshot{SessionID: "a%2Fb", CurrentModel: "another-owner", LastSeen: time.Now()}
	payload, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	key := RoutingSessionKey(config.DefaultRecipeName, "a/b")
	if key != legacy.SessionID {
		t.Fatal("fixture does not represent the historical encoding collision")
	}
	hook.values[store.keyPrefix+key] = string(payload)
	snapshot, found, err := store.Load(key)
	if err != nil || found {
		t.Fatalf("legacy identity must be a cache miss: snapshot=%+v found=%t err=%v", snapshot, found, err)
	}
	if hook.values[store.keyPrefix+key] != string(payload) {
		t.Fatal("rejecting legacy identity changed the persisted bytes")
	}
}

func TestRedisSessionSnapshotRoundTrip(t *testing.T) {
	store, hook := newSnapshotRedisTestStore(t)
	snapshot := RouterSessionSnapshot{
		SessionID: RoutingSessionKey("speed", "team/run", "job%2Fa"), UserID: "user-a",
		CurrentModel: "frontier", LastSeen: time.Date(2026, time.January, 2, 3, 4, 5, 0, time.UTC),
		TurnCount: 3, SwitchCount: 1, ModelTurns: map[string]int{"cheap": 1, "frontier": 2},
		CumulativePromptTokens: 30, CumulativeCompletionTokens: 6, CumulativeCost: .04,
		ActiveToolLoop: true, LastDecisionName: "reasoning", LastDecisionReason: "stay",
	}
	if err := store.Save(snapshot, time.Minute); err != nil {
		t.Fatal(err)
	}
	var envelope struct {
		Version  int             `json:"version"`
		Snapshot json.RawMessage `json:"snapshot"`
	}
	if err := json.Unmarshal([]byte(hook.values[store.keyPrefix+snapshot.SessionID]), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Version != 2 || len(envelope.Snapshot) == 0 {
		t.Fatalf("Save did not bind the escaped identity codec: %+v", envelope)
	}
	restored, found, err := store.Load(snapshot.SessionID)
	if err != nil || !found || !reflect.DeepEqual(restored, snapshot) {
		t.Fatalf("same-version restore lost ownership or usage: snapshot=%+v found=%t err=%v", restored, found, err)
	}
}

func TestRedisSessionSnapshotLoadRejectsUnsupportedOrInvalidPayloads(t *testing.T) {
	for _, tc := range []struct {
		name    string
		payload string
		wantErr bool
	}{
		{name: "unknown_version", payload: `{"version":3,"snapshot":{"SessionID":"session-a","CurrentModel":"other"}}`},
		{name: "unknown_version_shape", payload: `{"version":3,"snapshot":["future-format"]}`},
		{name: "missing_snapshot", payload: `{"version":2}`},
		{name: "mismatched_identity", payload: `{"version":2,"snapshot":{"SessionID":"other","CurrentModel":"other"}}`},
		{name: "invalid_json", payload: `{"version":2,`, wantErr: true},
		{name: "invalid_snapshot", payload: `{"version":2,"snapshot":"invalid"}`, wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			store, hook := newSnapshotRedisTestStore(t)
			const key = "session-a"
			hook.values[store.keyPrefix+key] = tc.payload
			snapshot, found, err := store.Load(key)
			if found || (err != nil) != tc.wantErr {
				t.Fatalf("unsupported payload restored: snapshot=%+v found=%t err=%v", snapshot, found, err)
			}
			if hook.values[store.keyPrefix+key] != tc.payload {
				t.Fatal("Load changed rejected persisted data")
			}
		})
	}
}
