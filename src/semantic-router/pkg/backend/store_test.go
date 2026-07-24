package backend

import (
	"testing"
	"time"
)

func TestStoreGetFreshHonorsDefaultTTL(t *testing.T) {
	base := time.Unix(100, 0)
	now := base
	store := newStoreWithClock(5*time.Second, func() time.Time { return now })
	identity := BackendIdentity{
		BackendID: "qwen-primary",
		ModelName: "qwen3-8b",
		ReplicaID: "replica-a",
	}

	if err := store.Upsert(BackendTelemetry{Identity: identity}); err != nil {
		t.Fatalf("Upsert returned error: %v", err)
	}
	if _, ok := store.GetFresh(identity); !ok {
		t.Fatal("expected telemetry to be fresh immediately after upsert")
	}

	now = base.Add(6 * time.Second)
	if _, ok := store.GetFresh(identity); ok {
		t.Fatal("expected telemetry to be stale after default TTL")
	}
	if telemetry, ok := store.Get(identity); !ok || telemetry.Identity.BackendID != identity.BackendID {
		t.Fatalf("expected raw stale telemetry to remain available, got %#v ok=%v", telemetry, ok)
	}
}

func TestStoreTelemetryTTLOverridesDefault(t *testing.T) {
	base := time.Unix(200, 0)
	now := base
	store := newStoreWithClock(5*time.Second, func() time.Time { return now })
	identity := BackendIdentity{BackendID: "slow-ttl", ModelName: "qwen3-8b"}

	if err := store.Upsert(BackendTelemetry{
		Identity:    identity,
		CollectedAt: base,
		TTL:         30 * time.Second,
	}); err != nil {
		t.Fatalf("Upsert returned error: %v", err)
	}

	now = base.Add(20 * time.Second)
	if _, ok := store.GetFresh(identity); !ok {
		t.Fatal("expected telemetry-specific TTL to keep sample fresh")
	}
}

func TestStoreGetFreshRejectsFutureCollectedAt(t *testing.T) {
	base := time.Unix(300, 0)
	store := newStoreWithClock(5*time.Second, func() time.Time { return base })
	identity := BackendIdentity{BackendID: "future-sample", ModelName: "qwen3-8b"}

	if err := store.Upsert(BackendTelemetry{
		Identity:    identity,
		CollectedAt: base.Add(time.Second),
	}); err != nil {
		t.Fatalf("Upsert returned error: %v", err)
	}

	if _, ok := store.GetFresh(identity); ok {
		t.Fatal("expected future-dated telemetry to be treated as not fresh")
	}
}

func TestStoreRejectsIncompleteIdentity(t *testing.T) {
	store := NewStore(DefaultTelemetryTTL)
	if err := store.Upsert(BackendTelemetry{Identity: BackendIdentity{ModelName: "qwen3-8b"}}); err == nil {
		t.Fatal("expected missing backend id to be rejected")
	}
	if err := store.Upsert(BackendTelemetry{Identity: BackendIdentity{BackendID: "backend"}}); err == nil {
		t.Fatal("expected missing model name to be rejected")
	}
}
