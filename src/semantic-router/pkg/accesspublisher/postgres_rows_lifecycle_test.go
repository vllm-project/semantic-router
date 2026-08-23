package accesspublisher

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

func TestLoadAccessPoliciesJoinsRowsCloseErrorOnEarlyReturn(t *testing.T) {
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
	now := time.Now().UTC()
	rows := sqlmock.NewRows([]string{"id", "name", "status", "revision", "created_at", "updated_at"}).
		AddRow("policy-id", "Policy", "active", int64(0), now, now).
		CloseError(closeErr)
	mock.ExpectQuery("SELECT id, name, status, revision, created_at, updated_at").
		WithArgs(accesscontrol.NamespaceID("namespace-id")).
		WillReturnRows(rows).
		RowsWillBeClosed()
	mock.ExpectRollback()

	_, err = loadAccessPolicies(context.Background(), transaction, accesscontrol.NamespaceID("namespace-id"))
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("loadAccessPolicies() error = %v, want joined rows close error", err)
	}
	if !strings.Contains(err.Error(), "invalid revision") {
		t.Fatalf("loadAccessPolicies() error = %v, want original revision validation error", err)
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
