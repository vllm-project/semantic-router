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

func TestUsagePipelineWithPostgreSQLAndRedis(t *testing.T) {
	db, client := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	seedNamespaceAndFencePolicy(t, ctx, db)

	prefix := "usageledger:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	partition := "partition-" + strings.ReplaceAll(uuid.NewString(), "-", "")
	stream, testUsagePipelineWithPostgreSQLAndRedisErr := NewRedisStream(client, RedisStreamOptions{
		KeyPrefix: prefix, Partition: partition, Group: "usage-writers", Consumer: "writer-a",
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	t.Cleanup(func() { deleteRedisPrefix(client, prefix) })
	worker, testUsagePipelineWithPostgreSQLAndRedisErr := NewWorker(stream, PostgresStore{DB: db}, WorkerOptions{
		NamespaceID: testNamespaceID, BatchSize: 50, Block: time.Millisecond, ReclaimIdle: time.Millisecond,
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	if err := worker.Ensure(ctx); err != nil {
		t.Fatal(err)
	}
	assertUsageInitialPersistence(t, ctx, db, client, prefix, partition, stream, worker)
}

func assertUsageInitialPersistence(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partition string,
	stream *RedisStream,
	worker *Worker,
) {
	t.Helper()
	minute := time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC)
	known := testTerminalEvent("integration-known", minute.Add(10*time.Second))
	known.ReplayID = "replay-integration-known"
	known.Dispatches[0].DecisionID = "decision-complex"
	known.Dispatches[0].DecisionName = "Complex"
	known.Dispatches[0].DecisionTier = 3
	known.Dispatches[0].ModelName = "integration-model"
	ttft := int64(80)
	known.TTFTMilliseconds = &ttft
	addTerminalEvent(t, ctx, client, stream.key, known)
	result, testUsagePipelineWithPostgreSQLAndRedisErr := worker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Inserted != 1 {
		t.Fatalf("known ProcessOnce() = (%+v, %v), want one insert", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 1, 1, 1)
	var replayKeyID, replayUserID, replayTeamID, replayModelName string
	var replayModelRevision int64
	if err := db.QueryRowContext(ctx, `SELECT api_key_id::text, user_id::text, team_id::text,
  served_models->0->>'name', (served_models->0->>'revision')::bigint
FROM inference_replays WHERE namespace_id=$1 AND replay_id=$2`,
		testNamespaceID, known.ReplayID).Scan(
		&replayKeyID, &replayUserID, &replayTeamID, &replayModelName, &replayModelRevision,
	); err != nil {
		t.Fatalf("read durable inference replay: %v", err)
	}
	if replayKeyID != testKeyID || replayUserID != testUserID || replayTeamID != testTeamID ||
		replayModelName != "integration-model" || replayModelRevision != 7 {
		t.Fatalf("durable replay identity/model = %s/%s/%s %s@%d",
			replayKeyID, replayUserID, replayTeamID, replayModelName, replayModelRevision)
	}
	conflicting := known
	conflicting.StatusCode = 201
	if _, err := (PostgresStore{DB: db}).PersistBatch(ctx, []TerminalEvent{conflicting}); !errors.Is(err, ErrConflict) {
		t.Fatalf("conflicting settlement error = %v, want ErrConflict", err)
	}
	assertLedgerCounts(t, ctx, db, 1, 1, 1)
	if _, err := db.ExecContext(ctx, `UPDATE usage_dispatch_attempts
SET admission_id = 'mismatched-admission'
	WHERE namespace_id = $1 AND admission_id = $2`, testNamespaceID, known.AdmissionID); err == nil {
		t.Fatal("attempt admission unexpectedly diverged from its parent dispatch")
	}
	assertUsageReplaySemantics(t, ctx, db, client, prefix, partition, stream, worker, minute, known)
}

func assertUsageReplaySemantics(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partition string,
	stream *RedisStream,
	worker *Worker,
	minute time.Time,
	known TerminalEvent,
) {
	t.Helper()
	// Simulate the exact process boundary where PostgreSQL committed but the
	// Redis acknowledgement was lost. The pending item must replay through the
	// settlement digest and be acknowledged without a second ledger write.
	committedBeforeAck := testTerminalEvent("integration-commit-before-ack", minute.Add(2*time.Minute))
	addTerminalEvent(t, ctx, client, stream.key, committedBeforeAck)
	failingStream := &failAckOnceStream{Stream: stream}
	failingWorker, testUsagePipelineWithPostgreSQLAndRedisErr := NewWorker(failingStream, PostgresStore{DB: db}, WorkerOptions{
		NamespaceID: testNamespaceID, BatchSize: 50, Block: time.Millisecond, ReclaimIdle: time.Millisecond,
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	result, testUsagePipelineWithPostgreSQLAndRedisErr := failingWorker.ProcessOnce(ctx)
	if !errors.Is(testUsagePipelineWithPostgreSQLAndRedisErr, errInjectedAckFailure) || result.Inserted != 1 {
		t.Fatalf("commit-before-ack ProcessOnce() = (%+v, %v), want committed insert and lost ACK", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 2, 2, 2)
	time.Sleep(5 * time.Millisecond)
	result, testUsagePipelineWithPostgreSQLAndRedisErr = failingWorker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Duplicate != 1 {
		t.Fatalf("commit-before-ack replay = (%+v, %v), want digest duplicate", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 2, 2, 2)

	// A second stream item for the same canonical terminal event takes the
	// digest-based idempotent path and acknowledges the duplicate.
	addTerminalEvent(t, ctx, client, stream.key, known)
	result, testUsagePipelineWithPostgreSQLAndRedisErr = worker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Duplicate != 1 {
		t.Fatalf("duplicate ProcessOnce() = (%+v, %v), want one duplicate", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 2, 2, 2)

	unknown := testTerminalEvent("integration-unknown", minute.Add(20*time.Second))
	unknown.EvidenceState = EvidenceUnknown
	unknown.Dispatches[0].UsageState = UsageUnknown
	unknown.Dispatches[0].UnknownReason = "provider_usage_missing"
	unknown.Dispatches[0].InputTokens = "0"
	unknown.Dispatches[0].OutputTokens = "0"
	unknown.Dispatches[0].Cost = DispatchCost{Currency: "USD", State: CostUnknown, Numerator: "0", Reason: "usage_missing"}
	unknown.Dispatches[0].Attempts[0].State = UsageUnknown
	unknown.Dispatches[0].Attempts[0].StatusCode = 0
	unknown.Fence = &UnknownFence{
		FenceID: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa", Reason: "authoritative_usage_missing",
		Bindings: []FenceBinding{{
			BindingID:      "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
			RuleID:         "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
			AdmissionLimit: "1000", MaximumDebit: "1000",
		}},
	}
	addTerminalEvent(t, ctx, client, stream.key, unknown)
	result, testUsagePipelineWithPostgreSQLAndRedisErr = worker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Inserted != 1 {
		t.Fatalf("unknown ProcessOnce() = (%+v, %v), want one insert", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 3, 3, 3)
	assertCount(t, ctx, db, "unknown_usage_fences", 1)
	assertCount(t, ctx, db, "unknown_usage_fence_bindings", 1)
	assertUsageRollupSummaries(t, ctx, db, client, prefix, partition, stream, minute, known)
}

func assertUsageRollupSummaries(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partition string,
	stream *RedisStream,
	minute time.Time,
	known TerminalEvent,
) {
	t.Helper()
	rollups := PostgresRollups{DB: db}
	if err := rollups.Refresh1m(ctx, testNamespaceID, minute, minute.Add(time.Minute)); err != nil {
		t.Fatalf("Refresh1m(): %v", err)
	}
	var viewRows, pollutedDimensions int
	if err := db.QueryRowContext(ctx, `SELECT count(*),
  count(*) FILTER (WHERE dimensions ? 'view' OR dimensions ? 'requests')
FROM usage_rollup_1m WHERE namespace_id = $1`, testNamespaceID).Scan(&viewRows, &pollutedDimensions); err != nil {
		t.Fatal(err)
	}
	if viewRows != 2 || pollutedDimensions != 0 {
		t.Fatalf("rollup rows = %d, polluted dimensions = %d, want one request and one dispatch view", viewRows, pollutedDimensions)
	}
	queries := PostgresQueries{DB: db}
	summary, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute,
		Visibility: QueryVisibility{All: true},
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	if summary.Totals.Requests != "2" || summary.Totals.InputTokens != "100" ||
		summary.Totals.OutputTokens != "20" || summary.Totals.IncompleteDispatches != "1" ||
		summary.Totals.Completeness != CompletenessPartial {
		t.Fatalf("summary totals = %+v, want exact known lower bound plus incomplete state", summary.Totals)
	}
	if summary.AsOf == nil || summary.LedgerWatermark == nil || summary.Final {
		t.Fatalf("summary freshness = %+v, want database as-of, observed watermark, and conservative non-final state", summary)
	}
	if len(summary.Totals.Costs) != 1 || summary.Totals.Costs[0].KnownAmount != "0.25" ||
		summary.Totals.Costs[0].Completeness != CompletenessPartial {
		t.Fatalf("summary costs = %+v, want partial USD 0.25", summary.Totals.Costs)
	}
	if summary.Totals.Latency.SampleCount != "2" ||
		summary.Totals.Latency.AverageMilliseconds != 250 ||
		summary.Totals.Latency.P95Milliseconds != 250 ||
		summary.Totals.TTFT.SampleCount != "1" || summary.Totals.TTFT.AverageMilliseconds != 80 ||
		summary.Totals.TTFT.P95Milliseconds != 125 {
		t.Fatalf("summary timings = latency %+v, TTFT %+v", summary.Totals.Latency, summary.Totals.TTFT)
	}
	series, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Series(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute,
		Visibility: QueryVisibility{All: true},
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(series.Points) != 1 || series.Points[0].Totals.Requests != "2" {
		t.Fatalf("series = (%+v, %v), want one exact minute", series, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	if series.AsOf == nil || series.LedgerWatermark == nil || series.IngestionLag == nil || series.Final {
		t.Fatalf("series freshness = %+v, want database as-of, watermark, lag, and conservative non-final state", series)
	}
	if err := rollups.Refresh1h(ctx, testNamespaceID, minute.Truncate(time.Hour), minute.Truncate(time.Hour).Add(time.Hour)); err != nil {
		t.Fatalf("Refresh1h(): %v", err)
	}
	hourSummary, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute.Truncate(time.Hour), End: minute.Truncate(time.Hour).Add(time.Hour), Grain: GrainHour,
		Visibility: QueryVisibility{All: true},
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || hourSummary.Totals.Requests != "2" || hourSummary.Totals.Costs[0].KnownAmount != "0.25" ||
		hourSummary.Totals.Latency.P95Milliseconds != 250 {
		t.Fatalf("hour summary = (%+v, %v), want lossless coarse rollup", hourSummary, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	dayStart := minute.Truncate(24 * time.Hour)
	if err := rollups.Refresh1d(ctx, testNamespaceID, dayStart, dayStart.Add(24*time.Hour)); err != nil {
		t.Fatalf("Refresh1d(): %v", err)
	}
	daySummary, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: dayStart, End: dayStart.Add(24 * time.Hour), Grain: GrainDay,
		Visibility: QueryVisibility{All: true},
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || daySummary.Totals.Requests != "2" || daySummary.Totals.Costs[0].KnownAmount != "0.25" ||
		daySummary.Totals.Latency.P95Milliseconds != 250 {
		t.Fatalf("day summary = (%+v, %v), want lossless coarse rollup", daySummary, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	dispatchSummary, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Summary(ctx, UsageQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute,
		Filters:    UsageFilters{LogicalModelID: testModelID},
		Visibility: QueryVisibility{All: true},
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || dispatchSummary.Totals.IncompleteDispatches != "1" {
		t.Fatalf("dispatch summary = (%+v, %v), want internal dispatch view", dispatchSummary, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertUsageBreakdownAndLogs(t, ctx, db, client, prefix, partition, stream, minute, known, queries)
}

func assertUsageBreakdownAndLogs(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partition string,
	stream *RedisStream,
	minute time.Time,
	known TerminalEvent,
	queries PostgresQueries,
) {
	t.Helper()
	breakdown, testUsagePipelineWithPostgreSQLAndRedisErr := queries.Breakdown(ctx, BreakdownQuery{
		UsageQuery: UsageQuery{NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute, Visibility: QueryVisibility{All: true}},
		Dimension:  BreakdownLogicalModel,
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(breakdown.Rows) != 1 || breakdown.Rows[0].Value != testModelID {
		t.Fatalf("model breakdown = (%+v, %v), want one internal Model row", breakdown, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	if breakdown.AsOf == nil || breakdown.LedgerWatermark == nil || breakdown.IngestionLag == nil || breakdown.Final {
		t.Fatalf("breakdown freshness = %+v, want database as-of, watermark, lag, and conservative non-final state", breakdown)
	}

	codec, testUsagePipelineWithPostgreSQLAndRedisErr := NewLogCursorCodec([]byte("0123456789abcdef0123456789abcdef"))
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	page, testUsagePipelineWithPostgreSQLAndRedisErr := queries.ListLogs(ctx, LogQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), PageSize: 1,
		Visibility: QueryVisibility{All: true},
	}, codec)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(page.Items) != 1 || page.NextCursor == "" {
		t.Fatalf("first log page = (%+v, %v), want cursor", page, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	second, testUsagePipelineWithPostgreSQLAndRedisErr := queries.ListLogs(ctx, LogQuery{
		NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), PageSize: 1, Cursor: page.NextCursor,
		Visibility: QueryVisibility{All: true},
	}, codec)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(second.Items) != 1 || second.Items[0].AdmissionID == page.Items[0].AdmissionID {
		t.Fatalf("second log page = (%+v, %v), want distinct second item", second, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	detail, testUsagePipelineWithPostgreSQLAndRedisErr := queries.RequestDetail(ctx, testNamespaceID, known.AdmissionID, QueryVisibility{All: true})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(detail.Dispatches) != 1 || len(detail.Dispatches[0].Attempts) != 1 ||
		detail.Request.TTFTMilliseconds == nil || *detail.Request.TTFTMilliseconds != 80 {
		t.Fatalf("request detail = (%+v, %v), want normalized dispatch and attempt", detail, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertUsageConsumerReclaim(t, ctx, db, client, prefix, partition, stream, minute)
}

func assertUsageConsumerReclaim(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	prefix string,
	partition string,
	stream *RedisStream,
	minute time.Time,
) {
	t.Helper()
	// Consumer-group reclaim proves crash recovery for a delivered but
	// unacknowledged terminal event.
	reclaimed := testTerminalEvent("integration-reclaimed", minute.Add(30*time.Second))
	addTerminalEvent(t, ctx, client, stream.key, reclaimed)
	read, testUsagePipelineWithPostgreSQLAndRedisErr := client.XReadGroup(ctx, &redis.XReadGroupArgs{
		Group: "usage-writers", Consumer: "crashed-writer", Streams: []string{stream.key, ">"}, Count: 1,
	}).Result()
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || len(read) != 1 || len(read[0].Messages) != 1 {
		t.Fatalf("seed pending message = (%+v, %v)", read, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	time.Sleep(5 * time.Millisecond)
	recoveryStream, testUsagePipelineWithPostgreSQLAndRedisErr := NewRedisStream(client, RedisStreamOptions{
		KeyPrefix: prefix, Partition: partition, Group: "usage-writers", Consumer: "writer-recovery",
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	recoveryWorker, _ := NewWorker(recoveryStream, PostgresStore{DB: db}, WorkerOptions{
		NamespaceID: testNamespaceID, BatchSize: 50, Block: time.Millisecond, ReclaimIdle: time.Millisecond,
	})
	result, testUsagePipelineWithPostgreSQLAndRedisErr := recoveryWorker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Inserted != 1 {
		t.Fatalf("reclaim ProcessOnce() = (%+v, %v), want recovered insert", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	pending, testUsagePipelineWithPostgreSQLAndRedisErr := client.XPending(ctx, stream.key, "usage-writers").Result()
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || pending.Count != 0 {
		t.Fatalf("pending = (%+v, %v), want empty after reclaim", pending, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 4, 4, 4)
	assertUsageRollupRecovery(t, ctx, db, client, stream, minute)
}

func assertUsageRollupRecovery(
	t *testing.T,
	ctx context.Context,
	db *sql.DB,
	client *redis.Client,
	stream *RedisStream,
	minute time.Time,
) {
	t.Helper()
	// Crash boundary after the immutable ledger commit but before rollup/XACK:
	// replay must take the duplicate settlement path, finish every grain, and
	// only then remove the pending entry.
	partial := testTerminalEvent("integration-commit-before-rollup", minute.Add(4*time.Minute))
	addTerminalEvent(t, ctx, client, stream.key, partial)
	processor, testUsagePipelineWithPostgreSQLAndRedisErr := NewPostgresRollupProcessor(db, PostgresRollupProcessorOptions{})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	hook := &failCommitHookOnce{delegate: processor}
	partialWorker, testUsagePipelineWithPostgreSQLAndRedisErr := NewWorker(stream, PostgresStore{DB: db}, WorkerOptions{
		NamespaceID: testNamespaceID, BatchSize: 50, Block: time.Millisecond,
		ReclaimIdle: time.Millisecond, AfterCommit: hook,
	})
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil {
		t.Fatal(testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	result, testUsagePipelineWithPostgreSQLAndRedisErr := partialWorker.ProcessOnce(ctx)
	if !errors.Is(testUsagePipelineWithPostgreSQLAndRedisErr, errInjectedRollupFailure) || result.Inserted != 1 {
		t.Fatalf("commit-before-rollup ProcessOnce() = (%+v, %v)", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	time.Sleep(5 * time.Millisecond)
	result, testUsagePipelineWithPostgreSQLAndRedisErr = partialWorker.ProcessOnce(ctx)
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || result.Duplicate != 1 {
		t.Fatalf("commit-before-rollup replay = (%+v, %v)", result, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertLedgerCounts(t, ctx, db, 5, 5, 5)
	pending, testUsagePipelineWithPostgreSQLAndRedisErr := client.XPending(ctx, stream.key, "usage-writers").Result()
	if testUsagePipelineWithPostgreSQLAndRedisErr != nil || pending.Count != 0 {
		t.Fatalf("pending after rollup recovery = (%+v, %v)", pending, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	if got := namespaceRollupRequests(t, ctx, db, "usage_rollup_1d", testNamespaceID); got != 3 {
		// The original test intentionally materialized only two request events
		// before this hook-driven event; other ledger rows are outside refreshed
		// ranges until a historical reconciliation pass.
		t.Fatalf("hook-driven daily request total = %d, want 3", got)
	}
}

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
		KeyPrefix: prefix, Partition: partitionOne, Group: defaultConsumerGroup, Consumer: "test-producer-one",
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
		KeyPrefix: prefix, Partition: partitionTwo, Group: defaultConsumerGroup, Consumer: "test-producer-two",
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
		consumers, err := client.XInfoConsumers(ctx, streamTwo.key, defaultConsumerGroup).Result()
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
		Group: defaultConsumerGroup, Consumer: "crashed-replica", Streams: []string{streamTwo.key, ">"}, Count: 1,
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
		pending, err := client.XPending(ctx, streamTwo.key, defaultConsumerGroup).Result()
		return err == nil && pending.Count == 0
	})
	pending, testUsageSupervisorDiscoversNamespacesReclaimsAndRefreshesLateRollupsErr := client.XPending(ctx, streamTwo.key, defaultConsumerGroup).Result()
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
