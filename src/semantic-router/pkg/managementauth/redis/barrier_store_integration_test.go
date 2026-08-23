package redis

import (
	"context"
	"errors"
	"net/url"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	redisclient "github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	testSessionID   = "11111111-1111-4111-8111-111111111111"
	testPrincipalID = "22222222-2222-4222-8222-222222222222"
	testSourceID    = "33333333-3333-4333-8333-333333333333"
	testNamespaceID = "44444444-4444-4444-8444-444444444444"
)

type snapshotLoader struct {
	mu       sync.Mutex
	barriers []managementauth.RevocationBarrier
	err      error
}

func (loader *snapshotLoader) LoadRevocationBarriers(context.Context) ([]managementauth.RevocationBarrier, error) {
	loader.mu.Lock()
	defer loader.mu.Unlock()
	return append([]managementauth.RevocationBarrier(nil), loader.barriers...), loader.err
}

func TestStoreRebuildInstallRemoveAndFailClosed(t *testing.T) {
	client := testRedisClient(t)
	prefix := "vsr:test:management-revocation:" + uuid.NewString()
	loader := &snapshotLoader{barriers: []managementauth.RevocationBarrier{
		{Kind: managementauth.BarrierManagementPrincipal, ID: testPrincipalID},
		{Kind: managementauth.BarrierAuthenticationSource, ID: "issuer:" + testSourceID},
	}}
	store, testStoreRebuildInstallRemoveAndFailClosedErr := New(Options{Client: client, KeyPrefix: prefix, Loader: loader})
	if testStoreRebuildInstallRemoveAndFailClosedErr != nil {
		t.Fatal(testStoreRebuildInstallRemoveAndFailClosedErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	if err := store.Ready(ctx); !errors.Is(err, ErrNotReady) {
		t.Fatalf("uninitialized Ready() = %v", err)
	}
	if state, err := store.Check(ctx, testBarrierCheck()); err != nil || state.Ready {
		t.Fatalf("uninitialized Check() = %+v, %v", state, err)
	}
	if err := store.Rebuild(ctx); err != nil {
		t.Fatal(err)
	}
	if err := store.Ready(ctx); err != nil {
		t.Fatal(err)
	}
	state, testStoreRebuildInstallRemoveAndFailClosedErr := store.Check(ctx, testBarrierCheck())
	if testStoreRebuildInstallRemoveAndFailClosedErr != nil || !state.Ready || !state.PrincipalDenied || !state.AuthSourceDenied || state.SessionDenied {
		t.Fatalf("reconstructed Check() = %+v, %v", state, testStoreRebuildInstallRemoveAndFailClosedErr)
	}
	if err := store.InstallDeny(ctx, managementauth.BarrierManagementSession, testSessionID); err != nil {
		t.Fatal(err)
	}
	state, _ = store.Check(ctx, testBarrierCheck())
	if !state.SessionDenied {
		t.Fatalf("installed Check() = %+v", state)
	}
	if err := store.RemoveDeny(ctx, managementauth.BarrierManagementSession, testSessionID); err != nil {
		t.Fatal(err)
	}
	state, _ = store.Check(ctx, testBarrierCheck())
	if state.SessionDenied {
		t.Fatalf("removed Check() = %+v", state)
	}

	loader.mu.Lock()
	loader.err = errors.New("postgres unavailable")
	loader.mu.Unlock()
	if err := store.Rebuild(ctx); err == nil {
		t.Fatal("failed durable snapshot unexpectedly rebuilt")
	}
	// A failed rebuild never replaces the last complete generation.
	if err := store.Ready(ctx); err != nil {
		t.Fatalf("previous generation lost after rebuild failure: %v", err)
	}
	state, _ = store.Check(ctx, testBarrierCheck())
	if !state.PrincipalDenied {
		t.Fatal("previous reconstructed barrier was lost")
	}
}

func TestStoreReplicaRebuildIsAtomic(t *testing.T) {
	client := testRedisClient(t)
	prefix := "vsr:test:management-revocation:" + uuid.NewString()
	loader := &snapshotLoader{barriers: []managementauth.RevocationBarrier{
		{Kind: managementauth.BarrierManagementSession, ID: testSessionID},
	}}
	first, testStoreReplicaRebuildIsAtomicErr := New(Options{Client: client, KeyPrefix: prefix, Loader: loader})
	if testStoreReplicaRebuildIsAtomicErr != nil {
		t.Fatal(testStoreReplicaRebuildIsAtomicErr)
	}
	second, testStoreReplicaRebuildIsAtomicErr := New(Options{Client: client, KeyPrefix: prefix, Loader: loader})
	if testStoreReplicaRebuildIsAtomicErr != nil {
		t.Fatal(testStoreReplicaRebuildIsAtomicErr)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	results := make(chan error, 2)
	go func() { results <- first.Rebuild(ctx) }()
	go func() { results <- second.Rebuild(ctx) }()
	for range 2 {
		if err := <-results; err != nil {
			t.Fatalf("replica rebuild failed: %v", err)
		}
	}
	state, testStoreReplicaRebuildIsAtomicErr := first.Check(ctx, testBarrierCheck())
	if testStoreReplicaRebuildIsAtomicErr != nil || !state.Ready || !state.SessionDenied {
		t.Fatalf("atomic replica state = %+v, %v", state, testStoreReplicaRebuildIsAtomicErr)
	}
}

func testBarrierCheck() managementauth.BarrierCheck {
	return managementauth.BarrierCheck{
		SessionID: testSessionID, PrincipalID: testPrincipalID,
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: testSourceID,
		NamespaceID: testNamespaceID,
	}
}

func testRedisClient(t *testing.T) *redisclient.Client {
	t.Helper()
	raw := os.Getenv("VLLM_SR_MANAGEMENT_AUTH_TEST_REDIS_URL")
	if raw == "" {
		t.Skip("VLLM_SR_MANAGEMENT_AUTH_TEST_REDIS_URL is not configured")
	}
	parsed, err := url.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	options, err := redisclient.ParseURL(parsed.String())
	if err != nil {
		t.Fatal(err)
	}
	client := redisclient.NewClient(options)
	t.Cleanup(func() { _ = client.Close() })
	return client
}
