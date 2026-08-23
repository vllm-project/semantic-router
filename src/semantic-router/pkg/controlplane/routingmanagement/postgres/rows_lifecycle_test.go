package postgres

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

func TestListSnapshotsJoinsRowsCloseErrorOnScanFailure(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}

	closeErr := errors.New("rows close failed")
	mock.ExpectQuery(regexp.QuoteMeta("SELECT s.namespace_id, s.routing_revision")).
		WillReturnRows(sqlmock.NewRows([]string{
			"namespace_id", "routing_revision", "content_digest", "status",
			"failure_reason", "member_count", "created_at", "activated_at",
		}).AddRow(nil, nil, nil, nil, nil, nil, nil, nil).CloseError(closeErr)).
		RowsWillBeClosed()

	_, err = (&Store{db: database}).ListSnapshots(
		context.Background(), "namespace-id", routingmanagement.SnapshotListQuery{Limit: 10},
	)
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("ListSnapshots() error = %v, want joined rows close error", err)
	}
	if !strings.Contains(err.Error(), "scan routing snapshot metadata") {
		t.Fatalf("ListSnapshots() error = %v, want original scan error", err)
	}

	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}

func TestLoadModelsTxJoinsRowsCloseErrorOnScanFailure(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}
	mock.ExpectBegin()
	transaction, err := database.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatalf("begin transaction: %v", err)
	}

	closeErr := errors.New("rows close failed")
	mock.ExpectQuery(regexp.QuoteMeta("SELECT m.namespace_id,m.id,r.name")).
		WillReturnRows(sqlmock.NewRows([]string{"namespace_id"}).AddRow(nil).CloseError(closeErr)).
		RowsWillBeClosed()
	mock.ExpectRollback()

	_, err = loadModelsTx(context.Background(), transaction, "namespace-id", []string{"model-id"})
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("loadModelsTx() error = %v, want joined rows close error", err)
	}
	if !strings.Contains(err.Error(), "scan routing Model page") {
		t.Fatalf("loadModelsTx() error = %v, want original scan error", err)
	}

	if err := transaction.Rollback(); err != nil {
		t.Fatalf("roll back transaction: %v", err)
	}
	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}
