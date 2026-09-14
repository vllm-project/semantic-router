package sessiontools

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func redisStoreTestConfig(address string) config.ToolSessionStoreConfig {
	ttlSeconds := 60
	maxSessions := 10
	maxSessionsByIdentity := 2
	maxStateBytes := 4096
	timeoutMillis := 10
	return config.ToolSessionStoreConfig{
		Backend:               config.ToolSessionStoreBackendRedis,
		TTLSeconds:            &ttlSeconds,
		MaxSessions:           &maxSessions,
		MaxSessionsByIdentity: &maxSessionsByIdentity,
		MaxStateBytes:         &maxStateBytes,
		TimeoutMs:             &timeoutMillis,
		Redis: &config.ToolSessionRedisConfig{
			Address:   address,
			KeyPrefix: "vsr:test:session-tools",
		},
	}
}

func TestNewRedisStoreRequiresRedisBackend(t *testing.T) {
	if _, err := NewRedisStore(config.ToolSessionStoreConfig{}); err == nil {
		t.Fatal("expected the Redis constructor to reject the local backend")
	}
}

func TestRedisStoreKeysDoNotExposeSessionOrQuotaIdentities(t *testing.T) {
	store, err := NewRedisStore(redisStoreTestConfig("127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	const session = "raw-session-value"
	quota := QuotaKey{Principal: "raw-principal-value", Namespace: "raw-namespace-value"}
	stateKey := store.stateKey(session)
	quotaLRU, quotaExpiry := store.quotaIndexKeys(quota)
	for _, key := range []string{stateKey, quotaLRU, quotaExpiry} {
		if strings.Contains(key, session) || strings.Contains(key, quota.Principal) || strings.Contains(key, quota.Namespace) {
			t.Fatalf("Redis key exposes a raw identity: %q", key)
		}
	}
	if !strings.HasPrefix(stateKey, "vsr:test:session-tools:state:") {
		t.Fatalf("state key = %q, want the normalized configured prefix", stateKey)
	}
	if stateKey != store.stateKey(session) {
		t.Fatal("state-key hashing must be deterministic")
	}
	otherLRU, _ := store.quotaIndexKeys(QuotaKey{Principal: quota.Principal, Namespace: "other"})
	if quotaLRU == otherLRU {
		t.Fatal("quota namespace must participate in the opaque identity index key")
	}
}

func TestRedisStoreLuaContractsKeepCASAndInvalidationAtomic(t *testing.T) {
	casFragments := []string{
		`redis.call("HGET", state_key, "revision")`,
		`revision ~= expected_revision`,
		`redis.call("INCR", generation_key)`,
		`ensure_slot(requested_quota_lru`,
		`ensure_slot(global_lru`,
		`redis.call("PEXPIRE", state_key, ttl_ms)`,
	}
	for _, fragment := range casFragments {
		if !strings.Contains(redisCompareAndSwapScript, fragment) {
			t.Fatalf("CAS script is missing %q", fragment)
		}
	}
	if strings.Contains(redisCompareAndSwapScript, "tonumber(expected_revision)") {
		t.Fatal("revision comparison must stay string-based to preserve uint64 precision")
	}
	for _, fragment := range []string{
		`revision ~= ARGV[1]`,
		`generation ~= ARGV[2]`,
		`redis.call("DEL", state_key)`,
	} {
		if !strings.Contains(redisDeleteIfTokenScript, fragment) {
			t.Fatalf("conditional-delete script is missing %q", fragment)
		}
	}
	for _, fragment := range []string{
		`local function ensure_zset`,
		`local function remove_index_member`,
		`redis.call("TYPE", key)`,
	} {
		if !strings.Contains(redisLoadScript, fragment) ||
			!strings.Contains(redisCompareAndSwapScript, fragment) ||
			!strings.Contains(redisDeleteScript, fragment) {
			t.Fatalf("Redis scripts must guard index operations with %q", fragment)
		}
	}
	for _, fragment := range []string{
		`local function remove_slot_member`,
		`member_lru ~= lru_key`,
		`member_expiry ~= expiry_key`,
		`member_lru == member`,
		`remove_index_member(lru_key, member, state_key)`,
		`string.match(generation_value, "^%d+$")`,
	} {
		if !strings.Contains(redisCompareAndSwapScript, fragment) {
			t.Fatalf("CAS script is missing corruption/ABA guard %q", fragment)
		}
	}
	for _, fragment := range []string{
		`quota_lru == state_key`,
		`quota_expiry == state_key`,
		`quota_lru == quota_expiry`,
	} {
		if !strings.Contains(redisLoadScript, fragment) ||
			!strings.Contains(redisCompareAndSwapScript, fragment) {
			t.Fatalf("Redis scripts must reject malformed quota pointers %q", fragment)
		}
	}
}

func TestRedisStoreUnavailableReturnsAnOperationError(t *testing.T) {
	store, err := NewRedisStore(redisStoreTestConfig("127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()
	if _, err := store.Load(ctx, "session"); err == nil {
		t.Fatal("an unavailable Redis endpoint must return a store operation error")
	}
}

func TestRedisStoreUnavailableMakesManagerFallBackStateless(t *testing.T) {
	store, err := NewRedisStore(redisStoreTestConfig("127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = store.Close() }()

	options := DefaultManagerOptions()
	options.OperationTimeout = 100 * time.Millisecond
	manager, err := NewManager(store, options)
	if err != nil {
		t.Fatal(err)
	}
	candidate := ToolCandidate{Name: "search", DefinitionFingerprint: "definition"}
	result := manager.Select(context.Background(), SelectionInput{
		Enabled:               true,
		Trusted:               true,
		Key:                   "session",
		Quota:                 QuotaKey{Principal: "principal", Namespace: "recipe"},
		PolicyFingerprint:     "policy",
		CatalogFingerprint:    "catalog",
		CapabilityFingerprint: "capability",
		Authorized:            []ToolCandidate{candidate},
		Selected:              []ToolCandidate{candidate},
		MaxTools:              2,
		MaxNewToolsPerTurn:    1,
	})
	if !result.Receipt.Fallback || result.Receipt.Reason != SelectionReasonStoreUnavailable {
		t.Fatalf("receipt = %+v, want store-unavailable fallback", result.Receipt)
	}
	if len(result.Selected) != 1 || result.Selected[0] != candidate {
		t.Fatalf("fallback selection = %+v, want the ordinary request-time selection", result.Selected)
	}
}

func TestRedisStoreCloseIsIdempotentAndRejectsOperations(t *testing.T) {
	store, err := NewRedisStore(redisStoreTestConfig("127.0.0.1:1"))
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Load(context.Background(), "session"); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("Load after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := store.CompareAndSwap(
		context.Background(),
		"session",
		0,
		State{},
		time.Minute,
		QuotaKey{},
	); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("CompareAndSwap after Close: err = %v, want ErrStoreClosed", err)
	}
	if err := store.Delete(context.Background(), "session"); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("Delete after Close: err = %v, want ErrStoreClosed", err)
	}
	if _, err := store.DeleteIfToken(context.Background(), "session", StateToken{}); !errors.Is(err, ErrStoreClosed) {
		t.Fatalf("DeleteIfToken after Close: err = %v, want ErrStoreClosed", err)
	}
}

func TestRedisReplyUint64PreservesLargeRevisions(t *testing.T) {
	const encoded = "18446744073709551615"
	got, err := redisReplyUint64(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if got != ^uint64(0) {
		t.Fatalf("revision = %d, want %s", got, encoded)
	}
}
