package postgres

import (
	"context"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

func TestAppendAuditPersistsNilActorChainAsEmptyArray(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })

	now := time.Date(2026, 8, 25, 1, 2, 3, 0, time.UTC)
	const (
		namespaceID = "11111111-1111-4111-8111-111111111111"
		actorID     = "22222222-2222-4222-8222-222222222222"
		resourceID  = "33333333-3333-4333-8333-333333333333"
	)
	mock.ExpectBegin()
	mock.ExpectExec(regexp.QuoteMeta("INSERT INTO access_audit_heads (namespace_id)")).
		WithArgs(namespaceID).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(regexp.QuoteMeta("SELECT event_count,last_hash FROM access_audit_heads")).
		WithArgs(namespaceID).
		WillReturnRows(sqlmock.NewRows([]string{"event_count", "last_hash"}).AddRow(int64(0), nil))
	mock.ExpectQuery(regexp.QuoteMeta("SELECT clock_timestamp()")).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
	mock.ExpectExec(regexp.QuoteMeta("INSERT INTO access_audit_events")).
		WithArgs(
			sqlmock.AnyArg(), namespaceID, nil, int64(1), actorID, []byte("[]"),
			"invitation.created", "invitation", resourceID, "request-1", nil,
			"Create invitation.", nil, int64(1), sqlmock.AnyArg(), nil,
			sqlmock.AnyArg(), now,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectExec(regexp.QuoteMeta("UPDATE access_audit_heads")).
		WithArgs(namespaceID, sqlmock.AnyArg(), sqlmock.AnyArg(), now, int64(0)).
		WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectCommit()

	tx, err := database.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatalf("begin transaction: %v", err)
	}
	err = appendAudit(
		context.Background(), tx, namespaceID, nil,
		"invitation.created", "invitation", resourceID, nil, 1,
		invitationmanagement.Actor{
			PrincipalID: actorID,
			RequestID:   "request-1",
			Reason:      "Create invitation.",
		},
	)
	if err != nil {
		t.Fatalf("append invitation audit: %v", err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatalf("commit transaction: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}
