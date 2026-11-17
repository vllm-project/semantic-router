package store

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/internal/testutil/storagetest"
)

func pendingReviewRecord(id string) Record {
	return Record{
		ID: id, RequestID: id, Timestamp: time.Unix(1700000000, 0).UTC(), SessionID: "agent-session",
		RequestKind: RequestKindCodexApprovalReview,
		Review: &Review{
			SchemaVersion: ReviewSchemaVersion, IdentificationSource: ReviewIdentifiedByEntrypoint,
			ReviewedSessionID: "agent-session", ThreadID: "review-thread", TurnID: "review-turn", ParentTurnID: "agent-turn",
			Status: ReviewStatusPending,
			Action: &ReviewAction{Tool: "exec_command", Summary: "rm -rf build", Cwd: "/work", Arguments: `{"command":["rm","-rf","build"]}`},
		},
	}
}

func completedReview(record Record) Review {
	review := *CloneReview(record.Review)
	review.Status, review.LatencyMS = ReviewStatusCompleted, 420
	review.Assessment = &ReviewAssessment{Outcome: ReviewOutcomeDeny, RiskLevel: "high", UserAuthorization: "low", Rationale: "Deletes build output the user did not ask to remove."}
	return review
}

func newMiniRedisStore(t *testing.T) *RedisStore {
	t.Helper()
	server := miniredis.RunT(t)
	redis, err := NewRedisStore(&RedisConfig{Address: server.Addr()}, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = redis.Close() })
	return redis
}

// The memory and Redis stores keep a review whole and hand out copies.
func TestReviewRoundTripsThroughMemoryAndRedis(t *testing.T) {
	ctx := context.Background()
	for name, storage := range map[string]Storage{"memory": NewMemoryStore(10, 0), "redis": newMiniRedisStore(t)} {
		t.Run(name, func(t *testing.T) {
			input := pendingReviewRecord("review-" + name)
			id, err := storage.Add(ctx, input)
			if err != nil {
				t.Fatal(err)
			}
			pending, _, err := storage.Get(ctx, id)
			if err != nil || pending.RequestKind != RequestKindCodexApprovalReview || !reflect.DeepEqual(pending.Review, input.Review) {
				t.Fatalf("pending review = %+v (%v)", pending.Review, err)
			}
			want := completedReview(input)
			if err = storage.UpdateReview(ctx, id, want); err != nil {
				t.Fatal(err)
			}
			got, _, err := storage.Get(ctx, id)
			if err != nil || !reflect.DeepEqual(*got.Review, want) {
				t.Fatalf("settled review = %+v, want %+v (%v)", got.Review, want, err)
			}
			got.Review.Assessment.Outcome, got.Review.Action.Tool = "allow", "changed"
			again, _, _ := storage.Get(ctx, id)
			if again.Review.Outcome() != ReviewOutcomeDeny || again.Review.Action.Tool != "exec_command" {
				t.Fatal("a returned record shares its review with the store")
			}
			if err = storage.UpdateReview(ctx, "missing", want); err == nil {
				t.Fatal("a review for an unknown record is an error")
			}
		})
	}
}

func TestPostgresReviewColumnsRoundTrip(t *testing.T) {
	input := pendingReviewRecord("postgres-review")
	review := completedReview(input)
	input.Review = &review
	insert, err := newPostgresInsertRecord(input)
	if err != nil {
		t.Fatal(err)
	}
	args, row := insert.args(), postgresRecordRow{signalsJSON: []byte("{}")}
	for column, want := range map[string]interface{}{
		"request_kind": RequestKindCodexApprovalReview, "review_status": ReviewStatusCompleted,
		"review_outcome": ReviewOutcomeDeny, "review_risk": "high",
	} {
		if got := args[columnIndex(t, column)]; got != want {
			t.Fatalf("%s = %v, want %v", column, got, want)
		}
	}
	row.review.requestKind.String, row.review.requestKind.Valid = RequestKindCodexApprovalReview, true
	destination, ok := row.scanDestinations()[columnIndex(t, "review")].(*[]byte)
	if !ok || destination != &row.review.reviewJSON {
		t.Fatal("review column scan misaligned")
	}
	*destination = args[columnIndex(t, "review")].([]byte)
	output, err := row.decode()
	if err != nil || output.RequestKind != RequestKindCodexApprovalReview || !reflect.DeepEqual(output.Review, input.Review) {
		t.Fatalf("decoded review = %+v (%v)", output.Review, err)
	}
}

// Outcome and risk exist only once a review completed, and a request that is
// no review leaves every review column NULL.
func TestPostgresReviewColumnsStayEmptyWithoutAVerdict(t *testing.T) {
	plain, err := newPostgresInsertRecord(Record{ID: "plain"})
	if err != nil {
		t.Fatal(err)
	}
	for _, column := range []string{"request_kind", "review_status", "review_outcome", "review_risk"} {
		if plain.args()[columnIndex(t, column)] != nil {
			t.Fatalf("a request that is no review stores %s", column)
		}
	}
	pending, err := newPostgresInsertRecord(pendingReviewRecord("pending"))
	if err != nil || pending.args()[columnIndex(t, "review_status")] != ReviewStatusPending || pending.args()[columnIndex(t, "review_outcome")] != nil {
		t.Fatalf("pending review columns = %v (%v)", pending.args()[columnIndex(t, "review_status"):], err)
	}
}

func TestPostgresReviewFiltersMigrationAndSummary(t *testing.T) {
	where, args := postgresReplayWhere(QueryFilters{SessionID: "s", RequestKind: RequestKindCodexApprovalReview, ReviewStatus: "completed", ReviewOutcome: "deny", ReviewRisk: "high"})
	if !reflect.DeepEqual(args, []interface{}{"s", RequestKindCodexApprovalReview, "completed", "deny", "high"}) {
		t.Fatalf("args = %#v", args)
	}
	for _, clause := range []string{"session_id = $1", "request_kind = $2", "review_status = $3", "review_outcome = $4", "review_risk = $5"} {
		if !strings.Contains(where, clause) {
			t.Fatalf("missing %s in %s", clause, where)
		}
	}
	schema := postgresCreateTableQuery(DefaultPostgresTableName)
	for _, statement := range []string{
		"ADD COLUMN IF NOT EXISTS request_kind VARCHAR(64);", "ADD COLUMN IF NOT EXISTS review JSONB;",
		"ADD COLUMN IF NOT EXISTS review_status", "ADD COLUMN IF NOT EXISTS review_outcome", "ADD COLUMN IF NOT EXISTS review_risk",
		"idx_router_replay_records_request_kind_timestamp", "idx_router_replay_records_review_status_outcome",
	} {
		if !strings.Contains(schema, statement) {
			t.Fatalf("fresh and existing tables must get %q", statement)
		}
	}
	if summary := postgresSummarySelectColumns(); !strings.Contains(summary, postgresReviewSummaryColumn) {
		t.Fatalf("list pages must drop review arguments: %s", summary)
	}
}

func openPostgresReviewStore(t *testing.T) *PostgresStore {
	t.Helper()
	storagetest.Require(t, "postgres")
	raw := os.Getenv("ROUTER_REPLAY_TEST_POSTGRES_CONFIG")
	if raw == "" {
		storagetest.Unavailable(t, "postgres", "set ROUTER_REPLAY_TEST_POSTGRES_CONFIG to run against PostgreSQL")
	}
	var cfg PostgresConfig
	if err := json.Unmarshal([]byte(raw), &cfg); err != nil {
		t.Fatal("invalid PostgreSQL test config")
	}
	cfg.TableName = fmt.Sprintf("replay_review_%d", time.Now().UnixNano())
	postgres, err := NewPostgresStore(&cfg, 0, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		//nolint:gosec // generated table name is validated by NewPostgresStore
		if _, dropErr := postgres.db.Exec("DROP TABLE IF EXISTS " + cfg.TableName); dropErr != nil {
			t.Error(dropErr)
		}
		_ = postgres.Close()
	})
	return postgres
}

// seedPostgresReviews writes a denied and an allowed review and a request of
// the same session that is no review, and returns the denied one.
func seedPostgresReviews(ctx context.Context, t *testing.T, postgres *PostgresStore) (Record, Review) {
	t.Helper()
	denied := pendingReviewRecord("review-denied")
	allowed := pendingReviewRecord("review-allowed")
	allowed.Timestamp = denied.Timestamp.Add(time.Second)
	plain := Record{ID: "plain-request", RequestID: "plain-request", Timestamp: denied.Timestamp, SessionID: "agent-session"}
	for _, record := range []Record{denied, allowed, plain} {
		if _, err := postgres.Add(ctx, record); err != nil {
			t.Fatal(err)
		}
	}
	allow := completedReview(allowed)
	allow.Assessment = &ReviewAssessment{Outcome: ReviewOutcomeAllow}
	want := completedReview(denied)
	for id, review := range map[string]Review{denied.ID: want, allowed.ID: allow} {
		if err := postgres.UpdateReview(ctx, id, review); err != nil {
			t.Fatal(err)
		}
	}
	return denied, want
}

// StorageIntegration: postgres
func TestPostgresReviewQueriesIntegration(t *testing.T) {
	postgres := openPostgresReviewStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	denied, want := seedPostgresReviews(ctx, t, postgres)

	page, err := postgres.QueryPage(ctx, QueryFilters{RequestKind: RequestKindCodexApprovalReview, ReviewOutcome: ReviewOutcomeDeny, ReviewRisk: "high"}, 10, 0, false)
	if err != nil || page.Total != 1 || page.Records[0].ID != denied.ID {
		t.Fatalf("deny page = %+v (%v)", page, err)
	}
	if summary := page.Records[0].Review; summary.Action.Arguments != "" || summary.Action.Summary != "rm -rf build" || summary.Assessment.Rationale == "" {
		t.Fatalf("summary review = %+v", summary)
	}
	detail, err := postgres.QueryPage(ctx, QueryFilters{RequestKind: RequestKindCodexApprovalReview}, 10, 0, true)
	if err != nil || detail.Total != 2 || detail.Records[1].ID != denied.ID || !reflect.DeepEqual(*detail.Records[1].Review, want) {
		t.Fatalf("detail page = %+v (%v)", detail, err)
	}
}

// The aggregate scan reads review enums and latency, never the action or the
// rationale, a row without a review stays one, and a session lists both.
//
// StorageIntegration: postgres
func TestPostgresReviewMetadataScanIntegration(t *testing.T) {
	postgres := openPostgresReviewStore(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	denied, _ := seedPostgresReviews(ctx, t, postgres)
	// The metadata scan carries no record ID; these fixtures share it as the request ID.
	scanned := map[string]Record{}
	if err := postgres.ScanMetadata(ctx, QueryFilters{}, func(record Record) error {
		scanned[record.RequestID] = record
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	scan := scanned[denied.ID]
	if scan.RequestKind != RequestKindCodexApprovalReview || scan.Review.Outcome() != ReviewOutcomeDeny || scan.Review.LatencyMS != 420 ||
		scan.Review.Action != nil || scan.Review.Assessment.Rationale != "" || !scan.Timestamp.Equal(denied.Timestamp) {
		t.Fatalf("metadata scan review = %+v at %v", scan.Review, scan.Timestamp)
	}
	if plain := scanned["plain-request"]; plain.RequestKind != "" || plain.Review != nil {
		t.Fatalf("plain request scanned as a review: %+v", plain.Review)
	}
	session, err := postgres.QueryPage(ctx, QueryFilters{SessionID: "agent-session"}, 10, 0, false)
	if err != nil || session.Total != 3 {
		t.Fatalf("a session holds its reviews and requests: %+v (%v)", session, err)
	}
}
