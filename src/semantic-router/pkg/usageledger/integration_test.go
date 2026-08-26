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

func TestRedisStreamQuarantinesPoisonWithoutBlockingValidItems(t *testing.T) {
	_, client := integrationStores(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	prefix := "usageledger:" + strings.ReplaceAll(uuid.NewString(), "-", "")
	partition := "partition-" + strings.ReplaceAll(uuid.NewString(), "-", "")
	stream, err := NewRedisStream(client, RedisStreamOptions{
		KeyPrefix: prefix, Partition: partition, Group: "usage-writers", Consumer: "writer-a",
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { deleteRedisPrefix(client, prefix) })
	if err := stream.EnsureGroup(ctx); err != nil {
		t.Fatal(err)
	}

	event := testTerminalEvent("redis-quarantine", time.Date(2026, 8, 22, 12, 0, 10, 0, time.UTC))
	payload, err := EncodeTerminalEvent(event)
	if err != nil {
		t.Fatal(err)
	}
	values := streamValues(event, payload)
	values["evidence_state"] = string(EvidenceUnknown)
	if err := client.XAdd(ctx, &redis.XAddArgs{Stream: stream.key, Values: values}).Err(); err != nil {
		t.Fatal(err)
	}
	items, err := stream.ReadNew(ctx, 1, time.Millisecond)
	if err != nil || len(items) != 1 {
		t.Fatalf("ReadNew() = (%+v, %v), want one item", items, err)
	}
	moved, err := stream.Quarantine(ctx, items[0], "event_envelope_mismatch")
	if err != nil || !moved {
		t.Fatalf("Quarantine() = (%t, %v), want durable move", moved, err)
	}
	if moved, err := stream.Quarantine(ctx, items[0], "event_envelope_mismatch"); err != nil || moved {
		t.Fatalf("duplicate Quarantine() = (%t, %v), want idempotent no-op", moved, err)
	}
	if length, err := client.XLen(ctx, stream.key).Result(); err != nil || length != 0 {
		t.Fatalf("source stream length = (%d, %v), want zero", length, err)
	}
	if count, err := stream.Quarantined(ctx); err != nil || count != 1 {
		t.Fatalf("Quarantined() = (%d, %v), want one", count, err)
	}
	entries, err := client.XRange(ctx, stream.quarantineKey, "-", "+").Result()
	if err != nil || len(entries) != 1 || entries[0].Values["source_id"] != items[0].ID ||
		entries[0].Values["reason"] != "event_envelope_mismatch" || entries[0].Values["payload"] == "" ||
		len(fmt.Sprint(entries[0].Values["payload_digest"])) != 64 {
		t.Fatalf("quarantine entries = (%+v, %v), want one recoverable sanitized record", entries, err)
	}

	valid := testTerminalEvent("redis-ack-delete", event.OccurredAt.Add(time.Second))
	validPayload, err := EncodeTerminalEvent(valid)
	if err != nil {
		t.Fatal(err)
	}
	if err := client.XAdd(ctx, &redis.XAddArgs{Stream: stream.key, Values: streamValues(valid, validPayload)}).Err(); err != nil {
		t.Fatal(err)
	}
	items, err = stream.ReadNew(ctx, 1, time.Millisecond)
	if err != nil || len(items) != 1 {
		t.Fatalf("valid ReadNew() = (%+v, %v), want one item", items, err)
	}
	if err := stream.Ack(ctx, []string{items[0].ID}); err != nil {
		t.Fatal(err)
	}
	if length, err := client.XLen(ctx, stream.key).Result(); err != nil || length != 0 {
		t.Fatalf("acknowledged source stream length = (%d, %v), want zero", length, err)
	}
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
	committedBeforeAck.Dispatches[0].DecisionID = "decision-no-replay"
	committedBeforeAck.Dispatches[0].DecisionName = "No replay"
	committedBeforeAck.Dispatches[0].DecisionTier = 2
	committedBeforeAck.Dispatches[0].ModelName = "no-replay-model"
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
	noReplayDetail, noReplayDetailErr := (PostgresQueries{DB: db}).RequestDetail(
		ctx, testNamespaceID, committedBeforeAck.AdmissionID, QueryVisibility{All: true},
	)
	if noReplayDetailErr != nil || noReplayDetail.Request.DecisionID != "decision-no-replay" ||
		len(noReplayDetail.Request.Models) != 1 || noReplayDetail.Request.Models[0].Name != "no-replay-model" ||
		noReplayDetail.QuotaReceipts == nil {
		t.Fatalf("terminal-snapshot request detail = (%+v, %v), want replay-independent evidence and canonical arrays",
			noReplayDetail, noReplayDetailErr)
	}
	var replayCount int
	if err := db.QueryRowContext(ctx, `SELECT count(*) FROM inference_replays
WHERE namespace_id=$1 AND event_id=$2::uuid`, testNamespaceID, committedBeforeAck.EventID).Scan(&replayCount); err != nil {
		t.Fatal(err)
	}
	if replayCount != 0 {
		t.Fatalf("optional replay rows = %d, want none for terminal-only observability proof", replayCount)
	}
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
	unknown.Dispatches[0].DecisionID = "decision-unknown"
	unknown.Dispatches[0].DecisionName = "Unknown usage"
	unknown.Dispatches[0].DecisionTier = 1
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
	if viewRows != 3 || pollutedDimensions != 0 {
		t.Fatalf("rollup rows = %d, polluted dimensions = %d, want two decision-scoped request rows and one dispatch row", viewRows, pollutedDimensions)
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
	decisionBreakdown, decisionBreakdownErr := queries.Breakdown(ctx, BreakdownQuery{
		UsageQuery: UsageQuery{NamespaceID: testNamespaceID, Start: minute, End: minute.Add(time.Minute), Grain: GrainMinute, Visibility: QueryVisibility{All: true}},
		Dimension:  BreakdownDecision,
	})
	if decisionBreakdownErr != nil || !breakdownContains(decisionBreakdown, "decision-complex") ||
		!breakdownContains(decisionBreakdown, "decision-unknown") {
		t.Fatalf("decision breakdown = (%+v, %v), want terminal decision evidence with and without replay",
			decisionBreakdown, decisionBreakdownErr)
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
	if page.Items[0].AdmissionID == known.AdmissionID &&
		(page.Items[0].DecisionName != "Complex" || len(page.Items[0].Models) != 1 ||
			page.Items[0].Models[0].Name != "integration-model") {
		t.Fatalf("request-log routing snapshots = %+v, want durable decision and Model names", page.Items[0])
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
		detail.Request.TTFTMilliseconds == nil || *detail.Request.TTFTMilliseconds != 80 ||
		!detail.Request.CompletedAt.Equal(known.CompletedAt) ||
		detail.Request.DecisionName != "Complex" || len(detail.Request.Models) != 1 ||
		detail.Request.Models[0].Name != "integration-model" {
		t.Fatalf("request detail = (%+v, %v), want normalized dispatch and attempt", detail, testUsagePipelineWithPostgreSQLAndRedisErr)
	}
	assertUsageConsumerReclaim(t, ctx, db, client, prefix, partition, stream, minute)
}

func breakdownContains(breakdown UsageBreakdown, value string) bool {
	for _, row := range breakdown.Rows {
		if row.Value == value {
			return true
		}
	}
	return false
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
