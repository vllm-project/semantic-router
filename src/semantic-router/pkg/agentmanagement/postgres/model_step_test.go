package postgres

import (
	"context"
	"crypto/sha256"
	"errors"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestBeginModelStepRejectsCancelledOrLostLeaseBeforeInferenceFence(t *testing.T) {
	t.Parallel()
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	request := testModelStep(4)

	mock.ExpectBegin()
	mock.ExpectQuery(`SELECT EXISTS\(SELECT 1 FROM agent_turns`).
		WithArgs(
			request.NamespaceID, request.SessionID, request.TurnID,
			request.WorkerID, request.Fence,
		).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
	mock.ExpectRollback()

	_, _, err = (&Store{db: database}).BeginModelStep(context.Background(), request)
	if !errors.Is(err, agentmanagement.ErrLeaseLost) {
		t.Fatalf("cancelled lease error = %v, want ErrLeaseLost", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestBeginModelStepReplaysCompletedOutcomeWithoutNewInference(t *testing.T) {
	t.Parallel()
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	request := testModelStep(7)
	now := time.Now().UTC()

	mock.ExpectBegin()
	expectActiveModelStepLease(mock, request, true)
	mock.ExpectQuery(`FROM agent_model_steps`).
		WithArgs(request.NamespaceID, request.SessionID, request.TurnID, request.Ordinal).
		WillReturnRows(modelStepRows().AddRow(
			request.ID, request.NamespaceID, request.SessionID, request.TurnID,
			request.Ordinal, request.Fence, request.RegistryRevision, request.RequestDigest,
			"completed", "end_turn", make([]byte, sha256.Size), now, now,
		))
	mock.ExpectCommit()

	step, replayed, err := (&Store{db: database}).BeginModelStep(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if !replayed || step.Status != "completed" || step.ID != request.ID {
		t.Fatalf("completed replay = (%+v, %t)", step, replayed)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestBeginModelStepFencesUnknownOutcomeInsteadOfRepeatingRequest(t *testing.T) {
	t.Parallel()
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	request := testModelStep(11)
	now := time.Now().UTC()

	mock.ExpectBegin()
	expectActiveModelStepLease(mock, request, true)
	mock.ExpectQuery(`FROM agent_model_steps`).
		WithArgs(request.NamespaceID, request.SessionID, request.TurnID, request.Ordinal).
		WillReturnRows(modelStepRows().AddRow(
			request.ID, request.NamespaceID, request.SessionID, request.TurnID,
			request.Ordinal, int64(10), request.RegistryRevision, request.RequestDigest,
			"started", "", []byte{}, now, nil,
		))
	mock.ExpectExec(`UPDATE agent_model_steps`).
		WithArgs(
			request.NamespaceID, request.SessionID, request.TurnID,
			request.Ordinal, request.Fence,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()

	step, replayed, err := (&Store{db: database}).BeginModelStep(context.Background(), request)
	if err != nil {
		t.Fatal(err)
	}
	if !replayed || step.Status != "unknown" || step.Fence != request.Fence {
		t.Fatalf("fenced replay = (%+v, %t)", step, replayed)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func testModelStep(fence int64) agentmanagement.ModelStep {
	digest := sha256.Sum256([]byte("request"))
	return agentmanagement.ModelStep{
		ID: uuid.NewString(), NamespaceID: uuid.NewString(), SessionID: uuid.NewString(),
		TurnID: uuid.NewString(), Ordinal: 1, WorkerID: "worker-1", Fence: fence,
		RegistryRevision: "sha256:registry", RequestDigest: digest[:],
	}
}

func expectActiveModelStepLease(
	mock sqlmock.Sqlmock, request agentmanagement.ModelStep, active bool,
) {
	mock.ExpectQuery(`SELECT EXISTS\(SELECT 1 FROM agent_turns`).
		WithArgs(
			request.NamespaceID, request.SessionID, request.TurnID,
			request.WorkerID, request.Fence,
		).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(active))
}

func modelStepRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "namespace_id", "session_id", "turn_id", "ordinal", "fence",
		"registry_revision", "request_digest", "status", "stop_reason",
		"output_digest", "started_at", "completed_at",
	})
}
