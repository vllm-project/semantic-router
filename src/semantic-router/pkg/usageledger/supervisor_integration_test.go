package usageledger

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

func TestUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollups(t *testing.T) {
	db, client := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	const (
		namespaceTwo = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
		partitionOne = "supervisor-partition-one"
		partitionTwo = "supervisor-partition-two"
	)
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'supervisor-one',$2,'USD','active')`, testNamespaceID, partitionOne); err != nil {
		t.Fatal(err)
	}

	prefix := "usage-supervisor:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	t.Cleanup(func() { deleteRedisPrefix(client, prefix) })
	namespaces, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := NewPostgresNamespaceSource(db)
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil {
		t.Fatal(testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	streams, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := NewRedisStreamFactory(client, prefix)
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil {
		t.Fatal(testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	newSupervisor := func(replica string) *Supervisor {
		// A production replica owns its own processor. Keeping that separation
		// here exercises the PostgreSQL advisory-lock boundary instead of
		// accidentally serializing both replicas through one in-process mutex.
		rollups, err := NewPostgresRollupProcessor(db, PostgresRollupProcessorOptions{DirtyBucketLimit: 10})
		if err != nil {
			t.Fatal(err)
		}
		storage, err := NewPostgresStorageLifecycle(db, StorageLifecycleOptions{
			CreateAheadMonths: 1, MaintenanceInterval: time.Minute,
		})
		if err != nil {
			t.Fatal(err)
		}
		supervisor, err := NewSupervisor(SupervisorOptions{
			Namespaces: namespaces, Streams: streams,
			Store: PostgresStore{DB: db, Partitions: storage}, Rollups: rollups, Storage: storage,
			ReplicaID: replica, BatchSize: 10, Block: 10 * time.Millisecond,
			ReclaimIdle: 10 * time.Millisecond, ReconcileInterval: 20 * time.Millisecond,
			RollupInterval: 10 * time.Millisecond, MinBackoff: time.Millisecond, MaxBackoff: 20 * time.Millisecond,
		})
		if err != nil {
			t.Fatal(err)
		}
		return supervisor
	}
	assertSupervisorDiscovery(t, ctx, db, client, prefix, partitionOne, partitionTwo, namespaceTwo, newSupervisor)
}

func assertSupervisorDiscovery(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partitionOne string,
	partitionTwo string,
	namespaceTwo string,
	newSupervisor func(string) *Supervisor,
) {
	t.Helper()
	supervisorA := newSupervisor("replica-a")
	runA := make(chan error, 1)
	go func() { runA <- supervisorA.Run(ctx) }()
	select {
	case <-supervisorA.Started():
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	if err := supervisorA.Ready(ctx); err != nil {
		t.Fatal(err)
	}

	minute := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	streamOne, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := NewRedisStream(client, RedisStreamOptions{
		KeyPrefix: prefix, Partition: partitionOne, Group: ConsumerGroupName, Consumer: "test-producer-one",
	})
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil {
		t.Fatal(testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	first := testTerminalEvent("supervisor-first", minute.Add(10*time.Second))
	addTerminalEvent(t, ctx, client, streamOne.key, first)
	waitUsageCondition(t, ctx, func() bool { return namespaceEventCount(t, ctx, db, testNamespaceID) == 1 })
	waitUsageCondition(t, ctx, func() bool { return namespaceRollupRequests(t, ctx, db, "usage_rollup_1m", testNamespaceID) == 1 })

	// A late event for an already materialized minute must make that historical
	// bucket dirty by ledger watermark; no fixed lookback is involved.
	late := testTerminalEvent("supervisor-late", minute.Add(20*time.Second))
	addTerminalEvent(t, ctx, client, streamOne.key, late)
	waitUsageCondition(t, ctx, func() bool { return namespaceEventCount(t, ctx, db, testNamespaceID) == 2 })
	waitUsageCondition(t, ctx, func() bool {
		return namespaceRollupRequests(t, ctx, db, "usage_rollup_1m", testNamespaceID) == 2 &&
			namespaceRollupRequests(t, ctx, db, "usage_rollup_1h", testNamespaceID) == 2 &&
			namespaceRollupRequests(t, ctx, db, "usage_rollup_1d", testNamespaceID) == 2
	})

	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'supervisor-two',$2,'USD','active')`, namespaceTwo, partitionTwo); err != nil {
		t.Fatal(err)
	}
	if err := supervisorA.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	streamTwo, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := NewRedisStream(client, RedisStreamOptions{
		KeyPrefix: prefix, Partition: partitionTwo, Group: ConsumerGroupName, Consumer: "test-producer-two",
	})
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil {
		t.Fatal(testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	secondNamespaceEvent := testTerminalEvent("supervisor-second-namespace", minute.Add(time.Minute))
	secondNamespaceEvent.NamespaceID = namespaceTwo
	addTerminalEvent(t, ctx, client, streamTwo.key, secondNamespaceEvent)
	waitUsageCondition(t, ctx, func() bool { return namespaceEventCount(t, ctx, db, namespaceTwo) == 1 })
	assertSupervisorConcurrency(t, ctx, db, client, namespaceTwo, minute, streamOne, streamTwo, supervisorA, runA, newSupervisor)
}

func assertSupervisorConcurrency(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	namespaceTwo string,
	minute time.Time,
	streamOne *RedisStream,
	streamTwo *RedisStream,
	supervisorA *Supervisor,
	runA <-chan error,
	newSupervisor func(string) *Supervisor,
) {
	t.Helper()
	// Two live Router replicas use the same group and distinct consumers. Their
	// concurrent ingestion and rollup passes must preserve exact cardinality.
	peer := newSupervisor("replica-peer")
	peerRun := make(chan error, 1)
	go func() { peerRun <- peer.Run(ctx) }()
	select {
	case <-peer.Started():
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	for index := 0; index < 12; index++ {
		event := testTerminalEvent(fmt.Sprintf("supervisor-shared-%d", index), minute.Add(time.Duration(70+index)*time.Second))
		event.NamespaceID = namespaceTwo
		addTerminalEvent(t, ctx, client, streamTwo.key, event)
	}
	waitUsageCondition(t, ctx, func() bool { return namespaceEventCount(t, ctx, db, namespaceTwo) == 13 })
	waitUsageCondition(t, ctx, func() bool {
		consumers, err := client.XInfoConsumers(ctx, streamTwo.key, ConsumerGroupName).Result()
		if err != nil {
			return false
		}
		names := make(map[string]struct{}, len(consumers))
		for _, consumer := range consumers {
			names[consumer.Name] = struct{}{}
		}
		_, replicaA := names["replica-a."+namespaceTwo]
		_, replicaPeer := names["replica-peer."+namespaceTwo]
		return replicaA && replicaPeer
	})
	if err := peer.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-peerRun:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("peer supervisor Run() = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("peer supervisor did not stop")
	}

	if _, err := db.ExecContext(ctx, `UPDATE access_namespaces SET status = 'disabled' WHERE id = $1`, testNamespaceID); err != nil {
		t.Fatal(err)
	}
	if err := supervisorA.Reconcile(ctx); err != nil {
		t.Fatal(err)
	}
	disabledEvent := testTerminalEvent("supervisor-disabled", minute.Add(2*time.Minute))
	addTerminalEvent(t, ctx, client, streamOne.key, disabledEvent)
	time.Sleep(50 * time.Millisecond)
	if got := namespaceEventCount(t, ctx, db, testNamespaceID); got != 2 {
		t.Fatalf("disabled namespace event count = %d, want 2", got)
	}

	if err := supervisorA.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-runA:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("first supervisor Run() = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("first supervisor did not stop")
	}
	assertSupervisorReclaim(t, ctx, db, client, namespaceTwo, minute, streamTwo, newSupervisor)
}

func assertSupervisorReclaim(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	namespaceTwo string,
	minute time.Time,
	streamTwo *RedisStream,
	newSupervisor func(string) *Supervisor,
) {
	t.Helper()
	// Deliver an item to a consumer that then disappears. A new replica in the
	// same group must XAUTOCLAIM it after the bounded idle period.
	reclaimed := testTerminalEvent("supervisor-reclaimed", minute.Add(3*time.Minute))
	reclaimed.NamespaceID = namespaceTwo
	addTerminalEvent(t, ctx, client, streamTwo.key, reclaimed)
	read, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := client.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group: ConsumerGroupName, Consumer: "crashed-replica", Streams: []string{streamTwo.key, ">"}, Count: 1,
	}).Result()
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil || len(read) != 1 || len(read[0].Messages) != 1 {
		t.Fatalf("seed crashed consumer = (%+v, %v)", read, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	time.Sleep(15 * time.Millisecond)
	supervisorB := newSupervisor("replica-b")
	runB := make(chan error, 1)
	go func() { runB <- supervisorB.Run(ctx) }()
	select {
	case <-supervisorB.Started():
	case <-ctx.Done():
		t.Fatal(ctx.Err())
	}
	waitUsageCondition(t, ctx, func() bool { return namespaceEventCount(t, ctx, db, namespaceTwo) == 14 })
	waitUsageCondition(t, ctx, func() bool {
		pending, err := client.XPending(ctx, streamTwo.key, ConsumerGroupName).Result()
		return err == nil && pending.Count == 0
	})
	pending, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := client.XPending(ctx, streamTwo.key, ConsumerGroupName).Result()
	if testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr != nil || pending.Count != 0 {
		t.Fatalf("pending after supervisor reclaim = (%+v, %v)", pending, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr)
	}
	if err := supervisorB.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-runB:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("second supervisor Run() = %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("second supervisor did not stop")
	}
}

func TestRequestRollupUsesRequestBucketForIncompleteDispatches(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if _, err := db.ExecContext(ctx, `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'cross-minute','cross-minute-partition','USD','active')`, testNamespaceID); err != nil {
		t.Fatal(err)
	}

	minute := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	event := testTerminalEvent("cross-minute-unknown", minute.Add(59*time.Second+900*time.Millisecond))
	event.CompletedAt = minute.Add(time.Minute + 400*time.Millisecond)
	event.LatencyMilliseconds = 500
	event.EvidenceState = EvidenceUnknown
	event.Dispatches[0].StartedAt = minute.Add(time.Minute + 100*time.Millisecond)
	event.Dispatches[0].CompletedAt = event.CompletedAt
	event.Dispatches[0].InputTokens = "0"
	event.Dispatches[0].OutputTokens = "0"
	event.Dispatches[0].UsageState = UsageUnknown
	event.Dispatches[0].UnknownReason = "provider_usage_missing"
	event.Dispatches[0].Cost = DispatchCost{
		Currency: "USD", State: CostUnknown, Numerator: "0", Reason: "usage_missing",
	}
	event.Dispatches[0].Attempts[0].StartedAt = event.Dispatches[0].StartedAt
	event.Dispatches[0].Attempts[0].CompletedAt = event.CompletedAt
	event.Dispatches[0].Attempts[0].State = UsageUnknown
	event.Dispatches[0].Attempts[0].StatusCode = 0
	if result, err := (PostgresStore{DB: db}).PersistBatch(ctx, []TerminalEvent{event}); err != nil || result.Inserted != 1 {
		t.Fatalf("PersistBatch() = (%+v, %v)", result, err)
	}

	rollups := PostgresRollups{DB: db}
	if err := rollups.Refresh1m(ctx, testNamespaceID, minute, minute.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if err := rollups.Refresh1m(ctx, testNamespaceID, minute.Add(time.Minute), minute.Add(2*time.Minute)); err != nil {
		t.Fatal(err)
	}
	queries := PostgresQueries{DB: db}
	requestSummary, err := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute,
		Visibility: QueryVisibility{All: true},
	})
	if err != nil || requestSummary.Totals.Requests != "1" || requestSummary.Totals.IncompleteDispatches != "1" {
		t.Fatalf("request summary = (%+v, %v), want the cross-minute dispatch on its request bucket", requestSummary, err)
	}
	dispatchSummary, err := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute.Add(time.Minute), End: minute.Add(2 * time.Minute), Grain: GrainMinute,
		Filters:    UsageFilters{LogicalModelID: testModelID},
		Visibility: QueryVisibility{All: true},
	})
	if err != nil || dispatchSummary.Totals.Requests != "1" || dispatchSummary.Totals.IncompleteDispatches != "1" {
		t.Fatalf("dispatch summary = (%+v, %v), want the dispatch on its actual start bucket", dispatchSummary, err)
	}
}
