package postgres

import (
	"context"
	"errors"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func TestClaimNextTurnUsesAtomicStatementAndClassifiesContention(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })

	expiresAt := time.Date(2026, time.August, 26, 5, 0, 0, 0, time.UTC)
	mock.ExpectQuery("^"+regexp.QuoteMeta("WITH candidate AS (")).
		WithArgs("worker/1", expiresAt).
		WillReturnError(&pq.Error{Code: "40001"})

	store := &Store{db: database}
	if _, err := store.ClaimNextTurn(context.Background(), "worker/1", expiresAt); !errors.Is(err, agentmanagement.ErrConflict) {
		t.Fatalf("ClaimNextTurn() error = %v, want conflict", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}

func TestClaimNextTurnReturnsFencedLease(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })

	expiresAt := time.Date(2026, time.August, 26, 5, 0, 0, 0, time.UTC)
	mock.ExpectQuery("^"+regexp.QuoteMeta("WITH candidate AS (")).
		WithArgs("worker/1", expiresAt).
		WillReturnRows(sqlmock.NewRows([]string{
			"namespace_id", "session_id", "turn_id", "worker_id", "fence", "registry_revision", "lease_expires_at",
		}).AddRow(
			"75f7055e-ae5f-4e87-b39d-22578694e120",
			"2a8bebd8-344f-4d91-b50c-9396b37bb202",
			"1dc3e68e-178f-44bc-a7e6-9274c60de2d5",
			"worker/1", int64(7), "sha256:registry", expiresAt,
		))

	store := &Store{db: database}
	lease, err := store.ClaimNextTurn(context.Background(), "worker/1", expiresAt)
	if err != nil {
		t.Fatal(err)
	}
	if lease.WorkerID != "worker/1" || lease.Fence != 7 || lease.RegistryRevision != "sha256:registry" ||
		!lease.ExpiresAt.Equal(expiresAt) {
		t.Fatalf("ClaimNextTurn() lease = %#v", lease)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
