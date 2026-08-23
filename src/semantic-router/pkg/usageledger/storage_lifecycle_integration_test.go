package usageledger

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstone(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)

	now := time.Now().UTC()
	oldMonth := usageMonth(now).AddDate(0, -18, 0)
	event := testTerminalEvent("retained-tombstone", oldMonth.Add(7*24*time.Hour+9*time.Hour))
	suffix := oldMonth.Format("200601")
	storage := lifecycleForTest(t, db, now, 90*24*time.Hour)
	store := PostgresStore{DB: db, Partitions: storage}
	result, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr := store.PersistBatch(ctx, []TerminalEvent{event})
	if testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr != nil || result.Inserted != 1 || len(result.projectionEvents) != 1 {
		t.Fatalf("PersistBatch() = (%+v, %v), want one projectable insert", result, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr)
	}
	assertUsagePhysicalPartition(t, ctx, db, "usage_events", "usage_events_"+suffix, event.AdmissionID)
	assertUsagePhysicalPartition(t, ctx, db, "usage_dispatches", "usage_dispatches_"+suffix, event.AdmissionID)
	assertUsagePhysicalPartition(t, ctx, db, "usage_dispatch_attempts", "usage_dispatch_attempts_"+suffix, event.AdmissionID)
	processAllDirtyRollups(t, ctx, db, testNamespaceID)
	assertUsageDirtyCount(t, ctx, db, 0)

	maintenance, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr := storage.Reconcile(ctx)
	if testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr != nil || maintenance.RetiredMonths != 1 || maintenance.BlockedMonths != 0 {
		t.Fatalf("Reconcile() = (%+v, %v), want one retired month", maintenance, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr)
	}
	for _, relation := range []string{
		"usage_events_" + suffix, "usage_dispatches_" + suffix,
		"usage_dispatch_attempts_" + suffix,
	} {
		if usageRelationExists(t, ctx, db, relation) {
			t.Fatalf("retired raw partition %q still exists", relation)
		}
	}
	var retained bool
	var retiredAt sql.NullTime
	if err := db.QueryRowContext(ctx, `SELECT event_retained,raw_retired_at
FROM usage_settlements WHERE namespace_id=$1 AND admission_id=$2`,
		testNamespaceID, event.AdmissionID).Scan(&retained, &retiredAt); err != nil {
		t.Fatal(err)
	}
	if retained || !retiredAt.Valid {
		t.Fatalf("settlement raw state = retained:%v retiredAt:%v, want durable tombstone", retained, retiredAt)
	}
	if _, err := (PostgresQueries{DB: db}).RequestDetail(
		ctx, testNamespaceID, event.AdmissionID, QueryVisibility{All: true},
	); !errors.Is(err, ErrNotFound) {
		t.Fatalf("retired RequestDetail() error = %v, want ErrNotFound", err)
	}

	result, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr = store.PersistBatch(ctx, []TerminalEvent{event})
	if testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr != nil || result.Duplicate != 1 || result.Inserted != 0 || len(result.projectionEvents) != 0 {
		t.Fatalf("retired duplicate = (%+v, %v), want a non-projectable digest match", result, testUsageStorageLifecycleRetiresRawFactsAndKeepsIdempotencyTombstoneErr)
	}
	if usageRelationExists(t, ctx, db, "usage_events_"+suffix) {
		t.Fatal("retired duplicate recreated its raw partition")
	}
	assertUsageDirtyCount(t, ctx, db, 0)
	if got := namespaceRollupRequests(t, ctx, db, "usage_rollup_1d", testNamespaceID); got != 1 {
		t.Fatalf("daily requests after retired duplicate = %d, want 1", got)
	}
	conflicting := event
	conflicting.StatusCode = 201
	if _, err := store.PersistBatch(ctx, []TerminalEvent{conflicting}); !errors.Is(err, ErrConflict) {
		t.Fatalf("retired conflicting digest error = %v, want ErrConflict", err)
	}
}

func TestUsageStorageRetentionBlocksDurableRecoveryReferences(t *testing.T) {
	for _, test := range []struct {
		name       string
		before     func(*TerminalEvent)
		after      func(*testing.T, context.Context, *sql.DB, TerminalEvent)
		leaveDirty bool
	}{
		{name: "pending rollup", leaveDirty: true},
		{
			name: "inference replay",
			before: func(event *TerminalEvent) {
				event.ReplayID = "retention-replay"
			},
		},
		{
			name: "unresolved usage fence",
			after: func(t *testing.T, ctx context.Context, db *sql.DB, event TerminalEvent) {
				if _, err := db.ExecContext(ctx, `INSERT INTO unknown_usage_fences(
  id,namespace_id,admission_id,reason,evidence,state
) VALUES ('aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa',$1,$2,'usage_missing','{}','open')`,
					testNamespaceID, event.AdmissionID); err != nil {
					t.Fatal(err)
				}
			},
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			db, _ := integrationStores(t)
			ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
			defer cancel()
			seedNamespaceAndFencePolicy(t, ctx, db)
			now := time.Now().UTC()
			oldMonth := usageMonth(now).AddDate(0, -18, 0)
			event := testTerminalEvent(
				"retention-block-"+strings.ReplaceAll(test.name, " ", "-"),
				oldMonth.Add(3*24*time.Hour+10*time.Hour),
			)
			storage := lifecycleForTest(t, db, now, 90*24*time.Hour)
			if test.before != nil {
				test.before(&event)
			}
			if _, err := (PostgresStore{DB: db, Partitions: storage}).PersistBatch(
				ctx, []TerminalEvent{event},
			); err != nil {
				t.Fatal(err)
			}
			if test.after != nil {
				test.after(t, ctx, db, event)
			}
			if !test.leaveDirty {
				processAllDirtyRollups(t, ctx, db, testNamespaceID)
			}
			maintenance, err := storage.Reconcile(ctx)
			if err != nil || maintenance.RetiredMonths != 0 || maintenance.BlockedMonths != 1 {
				t.Fatalf("Reconcile() = (%+v, %v), want referenced month blocked", maintenance, err)
			}
			if !usageRelationExists(t, ctx, db, "usage_events_"+oldMonth.Format("200601")) {
				t.Fatal("retention removed a partition with an active recovery reference")
			}
		})
	}
}

func TestUsageRollupDirtyTransitionIsAtomicAtClearFailure(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)
	event := testTerminalEvent("dirty-clear-fault", time.Now().UTC().Truncate(time.Minute).Add(-time.Minute+5*time.Second))
	if _, err := (PostgresStore{DB: db}).PersistBatch(ctx, []TerminalEvent{event}); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `CREATE FUNCTION fail_usage_dirty_clear()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'injected dirty clear failure';
END;
$$;
CREATE TRIGGER fail_usage_dirty_clear
BEFORE DELETE ON usage_rollup_dirty_minutes
FOR EACH STATEMENT EXECUTE FUNCTION fail_usage_dirty_clear()`); err != nil {
		t.Fatal(err)
	}
	minute := event.OccurredAt.Truncate(time.Minute)
	if err := (PostgresRollups{DB: db}).Refresh1m(ctx, testNamespaceID, minute, minute.Add(time.Minute)); err == nil {
		t.Fatal("fault-injected one-minute refresh unexpectedly succeeded")
	}
	if got := namespaceRollupRequests(t, ctx, db, "usage_rollup_1m", testNamespaceID); got != 0 {
		t.Fatalf("rolled-back one-minute requests = %d, want 0", got)
	}
	assertUsageQueueCount(t, ctx, db, "usage_rollup_dirty_minutes", 1)
	assertUsageQueueCount(t, ctx, db, "usage_rollup_dirty_hours", 0)
	if _, err := db.ExecContext(ctx, `DROP TRIGGER fail_usage_dirty_clear ON usage_rollup_dirty_minutes;
DROP FUNCTION fail_usage_dirty_clear()`); err != nil {
		t.Fatal(err)
	}
	if err := (PostgresRollups{DB: db}).Refresh1m(ctx, testNamespaceID, minute, minute.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	assertUsageQueueCount(t, ctx, db, "usage_rollup_dirty_minutes", 0)
	assertUsageQueueCount(t, ctx, db, "usage_rollup_dirty_hours", 1)
}

func TestUsageStorageMaintenanceIsReplicaSafeAndBoundedAtScale(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)
	now := time.Now().UTC()
	storageA := lifecycleForTest(t, db, now, 0)
	storageB := lifecycleForTest(t, db, now, 0)

	events := make([]TerminalEvent, 1000)
	minute := now.Truncate(time.Minute).Add(-time.Minute)
	for index := range events {
		events[index] = testTerminalEvent(fmt.Sprintf("scale-%04d", index), minute.Add(time.Duration(index%50)*time.Millisecond))
	}
	result, err := (PostgresStore{DB: db, Partitions: storageA}).PersistBatch(ctx, events)
	if err != nil || result.Inserted != len(events) {
		t.Fatalf("scale PersistBatch() = (%+v, %v), want %d inserts", result, err, len(events))
	}
	assertUsageQueueCount(t, ctx, db, "usage_rollup_dirty_minutes", 1)

	var wait sync.WaitGroup
	errorsByReplica := make(chan error, 2)
	for _, lifecycle := range []*PostgresStorageLifecycle{storageA, storageB} {
		wait.Add(1)
		go func(candidate *PostgresStorageLifecycle) {
			defer wait.Done()
			_, err := candidate.Reconcile(ctx)
			errorsByReplica <- err
		}(lifecycle)
	}
	wait.Wait()
	close(errorsByReplica)
	for err := range errorsByReplica {
		if err != nil {
			t.Fatalf("concurrent Reconcile() error = %v", err)
		}
	}
	var active int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM usage_partition_months
WHERE state='active' AND month_start >= $1::date AND month_start <= $2::date`,
		usageMonth(now), usageMonth(now).AddDate(0, 2, 0)).Scan(&active); err != nil {
		t.Fatal(err)
	}
	if active != 3 {
		t.Fatalf("active current/lookahead partitions = %d, want exactly 3", active)
	}
}

func TestUsageStorageMaintenanceRetiresAtMostOneMonthPerPass(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)
	now := time.Now().UTC()
	clock := now
	storage, err := NewPostgresStorageLifecycle(db, StorageLifecycleOptions{
		CreateAheadMonths: 2, MaintenanceInterval: time.Minute,
		RawRetention: 90 * 24 * time.Hour, Now: func() time.Time { return clock },
	})
	if err != nil {
		t.Fatal(err)
	}
	events := make([]TerminalEvent, 0, 3)
	for offset := -18; offset <= -16; offset++ {
		month := usageMonth(now).AddDate(0, offset, 0)
		events = append(events, testTerminalEvent(
			fmt.Sprintf("bounded-retirement-%d", -offset),
			month.Add(7*24*time.Hour+9*time.Hour),
		))
	}
	if _, err := (PostgresStore{DB: db, Partitions: storage}).PersistBatch(ctx, events); err != nil {
		t.Fatal(err)
	}
	processAllDirtyRollups(t, ctx, db, testNamespaceID)

	for pass := 1; pass <= len(events); pass++ {
		result, err := storage.Reconcile(ctx)
		if err != nil {
			t.Fatal(err)
		}
		if result.RetiredMonths != 1 || result.ScannedMonths != 1 {
			t.Fatalf("pass %d maintenance = %+v, want exactly one scanned and retired month", pass, result)
		}
		if result.MoreCandidates != (pass < len(events)) {
			t.Fatalf("pass %d more candidates = %v", pass, result.MoreCandidates)
		}
		var retired int
		if err := db.QueryRowContext(ctx, `SELECT count(*) FROM usage_partition_months
WHERE state='retired'`).Scan(&retired); err != nil {
			t.Fatal(err)
		}
		if retired != pass {
			t.Fatalf("retired months after pass %d = %d", pass, retired)
		}
		clock = clock.Add(2 * time.Minute)
	}
}

func TestUsageStorageMaintenanceWaitsForReplicaWriterTransaction(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)
	now := time.Now().UTC()
	writerStorage := lifecycleForTest(t, db, now, 0)
	maintenanceStorage := lifecycleForTest(t, db, now, 90*24*time.Hour)
	event := testTerminalEvent(
		"writer-lock-retirement",
		usageMonth(now).AddDate(0, -18, 0).Add(7*24*time.Hour+9*time.Hour),
	)
	if _, err := (PostgresStore{DB: db, Partitions: writerStorage}).PersistBatch(
		ctx, []TerminalEvent{event},
	); err != nil {
		t.Fatal(err)
	}
	processAllDirtyRollups(t, ctx, db, testNamespaceID)

	writer, err := db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = writer.Rollback() }()
	if err := writerStorage.LockWriterTx(ctx, writer); err != nil {
		t.Fatal(err)
	}
	type maintenanceOutcome struct {
		result StorageMaintenance
		err    error
	}
	finished := make(chan maintenanceOutcome, 1)
	go func() {
		result, err := maintenanceStorage.Reconcile(ctx)
		finished <- maintenanceOutcome{result: result, err: err}
	}()
	select {
	case outcome := <-finished:
		t.Fatalf("maintenance bypassed an active replica writer lock: %+v, %v", outcome.result, outcome.err)
	case <-time.After(100 * time.Millisecond):
	}
	if err := writer.Commit(); err != nil {
		t.Fatal(err)
	}
	select {
	case outcome := <-finished:
		if outcome.err != nil || outcome.result.RetiredMonths != 1 {
			t.Fatalf("maintenance after writer commit = (%+v, %v)", outcome.result, outcome.err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("maintenance did not resume after the replica writer committed")
	}
}

func TestUsageDetailAndLogPlansPruneMonthlyPartitions(t *testing.T) {
	db, _ := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)
	now := time.Now().UTC()
	firstMonth := usageMonth(now).AddDate(0, -18, 0)
	secondMonth := firstMonth.AddDate(0, 1, 0)
	storage := lifecycleForTest(t, db, now, 0)
	first := testTerminalEvent("partition-plan-first", firstMonth.Add(7*24*time.Hour+9*time.Hour))
	second := testTerminalEvent("partition-plan-second", secondMonth.Add(7*24*time.Hour+9*time.Hour))
	if _, err := (PostgresStore{DB: db, Partitions: storage}).PersistBatch(
		ctx, []TerminalEvent{first, second},
	); err != nil {
		t.Fatal(err)
	}

	assertPlanPrunesUsagePartition(t, ctx, db, `SELECT event_id FROM usage_events
WHERE namespace_id=$1 AND event_date=$2::date AND admission_id=$3
  AND event_kind IN ('actual','unknown')`,
		[]any{testNamespaceID, first.OccurredAt, first.AdmissionID},
		"usage_events_"+firstMonth.Format("200601"), "usage_events_"+secondMonth.Format("200601"))
	assertPlanPrunesUsagePartition(t, ctx, db, `SELECT dispatch_id FROM usage_dispatches
WHERE namespace_id=$1 AND event_date=$2::date AND event_id=$3::uuid AND admission_id=$4`,
		[]any{testNamespaceID, first.OccurredAt, first.EventID, first.AdmissionID},
		"usage_dispatches_"+firstMonth.Format("200601"), "usage_dispatches_"+secondMonth.Format("200601"))
	assertPlanPrunesUsagePartition(t, ctx, db, `SELECT attempt_id FROM usage_dispatch_attempts
WHERE namespace_id=$1 AND event_date=$2::date AND event_id=$3::uuid AND admission_id=$4`,
		[]any{testNamespaceID, first.OccurredAt, first.EventID, first.AdmissionID},
		"usage_dispatch_attempts_"+firstMonth.Format("200601"),
		"usage_dispatch_attempts_"+secondMonth.Format("200601"))

	logQuery := LogQuery{
		NamespaceID: testNamespaceID,
		Start:       first.OccurredAt.Truncate(24 * time.Hour),
		End:         first.OccurredAt.Truncate(24 * time.Hour).Add(24 * time.Hour),
		PageSize:    25,
		Visibility:  QueryVisibility{All: true},
	}
	statement, args := rawLogPageQuery(logQuery, &logCursor{
		OccurredAt: logQuery.End.Add(-time.Nanosecond).UnixNano(), EventID: first.EventID,
	})
	assertPlanPrunesUsagePartition(t, ctx, db, statement, args,
		"usage_events_"+firstMonth.Format("200601"), "usage_events_"+secondMonth.Format("200601"))
	detail, err := (PostgresQueries{DB: db}).RequestDetail(
		ctx, testNamespaceID, first.AdmissionID, QueryVisibility{All: true},
	)
	if err != nil || detail.Request.EventID != first.EventID || len(detail.Dispatches) != 1 {
		t.Fatalf("partition-pruned RequestDetail() = (%+v, %v)", detail, err)
	}
}

func lifecycleForTest(
	t *testing.T,
	db *sql.DB,
	now time.Time,
	rawRetention time.Duration,
) *PostgresStorageLifecycle {
	t.Helper()
	lifecycle, err := NewPostgresStorageLifecycle(db, StorageLifecycleOptions{
		CreateAheadMonths: 2, MaintenanceInterval: time.Minute,
		RawRetention: rawRetention, Now: func() time.Time { return now },
	})
	if err != nil {
		t.Fatal(err)
	}
	return lifecycle
}

func processAllDirtyRollups(t *testing.T, ctx context.Context, db *sql.DB, namespaceID string) {
	t.Helper()
	processor, err := NewPostgresRollupProcessor(db, PostgresRollupProcessorOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for pass := 0; pass < 10; pass++ {
		result, err := processor.ProcessDirty(ctx, namespaceID)
		if err != nil {
			t.Fatal(err)
		}
		if !result.More {
			return
		}
	}
	t.Fatal("usage dirty rollups did not converge")
}

func assertUsagePhysicalPartition(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	table, want, admissionID string,
) {
	t.Helper()
	var got string
	statement := "SELECT tableoid::regclass::text FROM " + table +
		" WHERE namespace_id=$1 AND admission_id=$2"
	if err := db.QueryRowContext(ctx, statement, testNamespaceID, admissionID).Scan(&got); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(got, want) {
		t.Fatalf("%s physical partition = %q, want suffix %q", table, got, want)
	}
}

func usageRelationExists(t *testing.T, ctx context.Context, db *sql.DB, relation string) bool {
	t.Helper()
	var exists bool
	if err := db.QueryRowContext(ctx, `SELECT to_regclass($1) IS NOT NULL`, relation).Scan(&exists); err != nil {
		t.Fatal(err)
	}
	return exists
}

func assertUsageDirtyCount(t *testing.T, ctx context.Context, db *sql.DB, want int) {
	t.Helper()
	for _, table := range []string{
		"usage_rollup_dirty_minutes", "usage_rollup_dirty_hours", "usage_rollup_dirty_days",
	} {
		assertUsageQueueCount(t, ctx, db, table, want)
	}
}

func assertUsageQueueCount(t *testing.T, ctx context.Context, db *sql.DB, table string, want int) {
	t.Helper()
	var got int
	if err := db.QueryRowContext(ctx, "SELECT count(*) FROM "+table).Scan(&got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("%s count = %d, want %d", table, got, want)
	}
}

func assertPlanPrunesUsagePartition(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	statement string,
	args []any,
	want, excluded string,
) {
	t.Helper()
	var plan string
	if err := db.QueryRowContext(ctx, "EXPLAIN (FORMAT JSON, COSTS OFF) "+statement, args...).Scan(&plan); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(plan, want) || strings.Contains(plan, excluded) {
		t.Fatalf("query plan does not prune to %q (excluded %q): %s", want, excluded, plan)
	}
}
