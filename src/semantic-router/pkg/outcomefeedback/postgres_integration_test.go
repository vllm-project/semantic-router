package outcomefeedback

import (
	"context"
	"database/sql"
	"errors"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
)

const (
	outcomeTestNamespace = "10000000-0000-4000-8000-000000000001"
	outcomeTestUser      = "10000000-0000-4000-8000-000000000002"
	outcomeTestTeam      = "10000000-0000-4000-8000-000000000003"
	outcomeTestKey       = "10000000-0000-4000-8000-000000000004"
	outcomeOtherUser     = "10000000-0000-4000-8000-000000000005"
	outcomeOtherKey      = "10000000-0000-4000-8000-000000000006"
)

func TestPostgresOutcomeOwnershipIdempotencyAndProjectionRebuild(t *testing.T) {
	database := outcomeIntegrationDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	seedOutcomeIdentity(t, ctx, database)
	seedOutcomeReplay(t, ctx, database, outcomeTestKey, outcomeTestUser, "replay-owned", "20000000-0000-4000-8000-000000000001")
	seedOutcomeReplay(t, ctx, database, outcomeOtherKey, outcomeOtherUser, "replay-other", "20000000-0000-4000-8000-000000000002")

	replicaA, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := NewPostgresRepository(database)
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil {
		t.Fatal(testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	replicaB, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := NewPostgresRepository(database)
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil {
		t.Fatal(testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	caller := Caller{
		NamespaceID: outcomeTestNamespace, APIKeyID: outcomeTestKey,
		UserID: outcomeTestUser, TeamID: outcomeTestTeam, Source: SourceAPIKey,
	}
	revision := int64(9)
	request := Request{
		ReplayID: "replay-owned", Target: TargetModel, TargetRef: "model/served",
		TargetRevision: &revision, Verdict: VerdictGoodFit,
	}

	const replicas = 12
	receipts := make([]Receipt, replicas)
	errorsSeen := make([]error, replicas)
	start := make(chan struct{})
	var wait sync.WaitGroup
	for index := 0; index < replicas; index++ {
		wait.Add(1)
		go func(index int) {
			defer wait.Done()
			<-start
			repository := replicaA
			if index%2 == 1 {
				repository = replicaB
			}
			receipts[index], errorsSeen[index] = repository.Record(ctx, caller, "concurrent-outcome", request)
		}(index)
	}
	close(start)
	wait.Wait()
	created := 0
	for index, recordErr := range errorsSeen {
		if recordErr != nil {
			t.Fatalf("replica %d Record() error = %v", index, recordErr)
		}
		if receipts[index].ID != receipts[0].ID || receipts[index].ProjectionRevision != 1 {
			t.Fatalf("replica %d receipt = %+v, first = %+v", index, receipts[index], receipts[0])
		}
		if !receipts[index].Duplicate {
			created++
		}
	}
	if created != 1 {
		t.Fatalf("new receipts = %d, want exactly one", created)
	}
	assertOutcomeCount(t, ctx, database, "inference_outcomes", 1)
	assertOutcomeCount(t, ctx, database, "inference_outcome_idempotency", 1)

	changed := request
	changed.Verdict = VerdictFailed
	if _, err := replicaB.Record(ctx, caller, "concurrent-outcome", changed); !errors.Is(err, ErrIdempotencyConflict) {
		t.Fatalf("changed idempotent request error = %v, want ErrIdempotencyConflict", err)
	}

	otherCaller := caller
	otherCaller.APIKeyID = outcomeOtherKey
	otherCaller.UserID = outcomeOtherUser
	for name, test := range map[string]struct {
		caller  Caller
		request Request
	}{
		"unknown replay": {caller: caller, request: func() Request { value := request; value.ReplayID = "replay-missing"; return value }()},
		"other key":      {caller: otherCaller, request: request},
		"other model": {caller: caller, request: func() Request {
			value := request
			value.TargetRef = "model/not-served"
			return value
		}()},
		"other revision": {caller: caller, request: func() Request {
			value := request
			wrong := int64(10)
			value.TargetRevision = &wrong
			return value
		}()},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := replicaA.Record(ctx, test.caller, "not-found-"+strings.ReplaceAll(name, " ", "-"), test.request); !errors.Is(err, ErrNotFound) {
				t.Fatalf("Record() error = %v, want nondisclosing ErrNotFound", err)
			}
		})
	}

	// A process can disappear after claiming the idempotency row but before the
	// transaction commits. A failed transaction must leave no poisoned claim.
	if _, err := database.ExecContext(ctx, `CREATE FUNCTION reject_outcome_once() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'injected outcome failure'; END; $$ LANGUAGE plpgsql;
CREATE TRIGGER reject_outcome_once BEFORE INSERT ON inference_outcomes
FOR EACH ROW EXECUTE FUNCTION reject_outcome_once()`); err != nil {
		t.Fatal(err)
	}
	if _, err := replicaA.Record(ctx, caller, "retry-after-rollback", changed); !errors.Is(err, ErrUnavailable) {
		t.Fatalf("injected failure error = %v, want ErrUnavailable", err)
	}
	if _, err := database.ExecContext(ctx, `DROP TRIGGER reject_outcome_once ON inference_outcomes;
DROP FUNCTION reject_outcome_once()`); err != nil {
		t.Fatal(err)
	}
	retried, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := replicaB.Record(ctx, caller, "retry-after-rollback", changed)
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil || retried.Duplicate || retried.ProjectionRevision != 2 {
		t.Fatalf("retry after rollback = (%+v, %v), want new revision 2", retried, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	assertOutcomeCount(t, ctx, database, "inference_outcomes", 2)
	assertOutcomeCount(t, ctx, database, "inference_outcome_idempotency", 2)

	firstPublisher := &projectionPublisherCapture{}
	firstProjector, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := NewProjector(ProjectorOptions{Repository: replicaA, Publisher: firstPublisher})
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil {
		t.Fatal(testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	if processed, err := firstProjector.ProcessOnce(ctx); err != nil || processed != 1 {
		t.Fatalf("ProcessOnce() = (%d, %v), want one namespace", processed, err)
	}
	if len(firstPublisher.projections) != 1 || firstPublisher.projections[0].Revision != 2 ||
		len(firstPublisher.projections[0].Entries) != 1 ||
		firstPublisher.projections[0].Entries[0].GoodFitCount != 1 ||
		firstPublisher.projections[0].Entries[0].FailedCount != 1 {
		t.Fatalf("first projection = %+v", firstPublisher.projections)
	}

	// A fresh repository/projector pair reconstructs the same revision entirely
	// from immutable PostgreSQL outcomes after a simulated process restart.
	restartedRepository, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := NewPostgresRepository(database)
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil {
		t.Fatal(testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	restartedPublisher := &projectionPublisherCapture{}
	restartedProjector, testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr := NewProjector(ProjectorOptions{Repository: restartedRepository, Publisher: restartedPublisher})
	if testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr != nil {
		t.Fatal(testPostgresOutcomeOwnershipIdempotencyAndProjectionRebuildErr)
	}
	if err := restartedProjector.Rebuild(ctx, outcomeTestNamespace); err != nil {
		t.Fatal(err)
	}
	if len(restartedPublisher.payloads) != 1 || string(restartedPublisher.payloads[0]) != string(firstPublisher.payloads[0]) ||
		restartedPublisher.digests[0] != firstPublisher.digests[0] {
		t.Fatalf("restart rebuild differs: first=%q restarted=%q", firstPublisher.payloads, restartedPublisher.payloads)
	}
}

type projectionPublisherCapture struct {
	projections []Projection
	payloads    [][]byte
	digests     [][32]byte
}

func (capture *projectionPublisherCapture) Publish(_ context.Context, projection Projection, payload []byte, digest [32]byte) error {
	capture.projections = append(capture.projections, projection)
	capture.payloads = append(capture.payloads, append([]byte(nil), payload...))
	capture.digests = append(capture.digests, digest)
	return nil
}

func outcomeIntegrationDatabase(t *testing.T) *sql.DB {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_OUTCOME_TEST_DATABASE_URL")
	if databaseURL == "" {
		databaseURL = os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	}
	if databaseURL == "" {
		t.Skip("outcome PostgreSQL integration database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	admin, outcomeIntegrationDatabaseErr := sql.Open("postgres", databaseURL)
	if outcomeIntegrationDatabaseErr != nil {
		t.Fatal(outcomeIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_outcome_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	scopedURL, outcomeIntegrationDatabaseErr := outcomeDatabaseURLWithSearchPath(databaseURL, schema)
	if outcomeIntegrationDatabaseErr != nil {
		t.Fatal(outcomeIntegrationDatabaseErr)
	}
	database, outcomeIntegrationDatabaseErr := sql.Open("postgres", scopedURL)
	if outcomeIntegrationDatabaseErr != nil {
		t.Fatal(outcomeIntegrationDatabaseErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	if err := (controlpostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return database
}

func outcomeDatabaseURLWithSearchPath(databaseURL, schema string) (string, error) {
	if !strings.Contains(databaseURL, "://") {
		return databaseURL + " search_path=" + schema, nil
	}
	parsed, err := url.Parse(databaseURL)
	if err != nil {
		return "", err
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	return parsed.String(), nil
}

func seedOutcomeIdentity(t *testing.T, ctx context.Context, database *sql.DB) {
	t.Helper()
	statements := []string{
		`INSERT INTO access_namespaces(id,name,quota_partition_id,billing_currency,status)
VALUES ('` + outcomeTestNamespace + `','outcome-test','outcome-test-partition','USD','active')`,
		`INSERT INTO access_subjects(namespace_id,id,kind) VALUES
('` + outcomeTestNamespace + `','` + outcomeTestUser + `','user'),
('` + outcomeTestNamespace + `','` + outcomeOtherUser + `','user'),
('` + outcomeTestNamespace + `','` + outcomeTestTeam + `','team'),
('` + outcomeTestNamespace + `','` + outcomeTestKey + `','api_key'),
('` + outcomeTestNamespace + `','` + outcomeOtherKey + `','api_key')`,
		`INSERT INTO access_users(id,namespace_id,email,display_name,status) VALUES
('` + outcomeTestUser + `','` + outcomeTestNamespace + `','owner@example.invalid','Owner','active'),
('` + outcomeOtherUser + `','` + outcomeTestNamespace + `','other@example.invalid','Other','active')`,
		`INSERT INTO access_teams(id,namespace_id,name,status)
VALUES ('` + outcomeTestTeam + `','` + outcomeTestNamespace + `','Outcome Team','active')`,
		`INSERT INTO access_team_memberships(namespace_id,team_id,user_id,role,status) VALUES
('` + outcomeTestNamespace + `','` + outcomeTestTeam + `','` + outcomeTestUser + `','member','active'),
('` + outcomeTestNamespace + `','` + outcomeTestTeam + `','` + outcomeOtherUser + `','member','active')`,
		`INSERT INTO access_api_keys(id,namespace_id,name,owner_user_id,context_team_id,status) VALUES
('` + outcomeTestKey + `','` + outcomeTestNamespace + `','Owner key','` + outcomeTestUser + `','` + outcomeTestTeam + `','active'),
('` + outcomeOtherKey + `','` + outcomeTestNamespace + `','Other key','` + outcomeOtherUser + `','` + outcomeTestTeam + `','active')`,
	}
	for _, statement := range statements {
		if _, err := database.ExecContext(ctx, statement); err != nil {
			t.Fatalf("seed outcome identity: %v", err)
		}
	}
}

func seedOutcomeReplay(t *testing.T, ctx context.Context, database *sql.DB, keyID, userID, replayID, eventID string) {
	t.Helper()
	occurredAt := time.Date(2026, 8, 23, 12, 0, 0, 0, time.UTC)
	admissionID := "admission-" + replayID
	if _, err := database.ExecContext(ctx, `SELECT ensure_usage_month_partition($1::date)`, occurredAt); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO usage_settlements(
namespace_id,admission_id,state,canonical_usage_digest,settled_at,event_partition_date)
VALUES ($1,$2,'settled',$3,$4,$5)`, outcomeTestNamespace, admissionID, make([]byte, 32), occurredAt, occurredAt); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO usage_events(
namespace_id,admission_id,event_date,event_id,event_kind,protocol,path,api_key_id,user_id,team_id,
status_code,input_tokens,output_tokens,total_tokens,served_input_tokens,served_output_tokens,
served_total_tokens,latency_ms,usage_state,costs,request_metadata,occurred_at,ingested_at)
VALUES ($1,$2,$3,$4,'actual','openai','/v1/chat/completions',$5,$6,$7,
200,1,1,2,1,1,2,10,'known_actual','[]','{}',$8,$8)`,
		outcomeTestNamespace, admissionID, occurredAt, eventID, keyID, userID, outcomeTestTeam, occurredAt); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO inference_replays(
namespace_id,replay_id,api_key_id,user_id,team_id,event_date,event_id,routing_context,served_models,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`, outcomeTestNamespace, replayID, keyID, userID,
		outcomeTestTeam, occurredAt, eventID,
		`{"recipe_id":"recipe-balanced","recipe_name":"Balanced","recipe_revision":4,"decision_id":"complex","decision_name":"Complex","decision_tier":3}`,
		`[{"id":"model/served","name":"served-model","revision":9}]`, occurredAt); err != nil {
		t.Fatal(err)
	}
}

func assertOutcomeCount(t *testing.T, ctx context.Context, database *sql.DB, table string, want int) {
	t.Helper()
	var count int
	if err := database.QueryRowContext(ctx, "SELECT count(*) FROM "+pq.QuoteIdentifier(table)).Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != want {
		t.Fatalf("%s count = %d, want %d", table, count, want)
	}
}
