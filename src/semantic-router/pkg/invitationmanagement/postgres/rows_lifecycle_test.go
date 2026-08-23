package postgres

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

func TestListJoinsRowsCloseErrorOnScanFailure(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}

	closeErr := errors.New("rows close failed")
	mock.ExpectQuery(regexp.QuoteMeta("SELECT id::text, namespace_id::text")).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "namespace_id", "created_by_principal_id", "expected_issuer",
			"expected_subject", "expected_email", "display_name", "grants",
			"expires_at", "status", "accepted_principal_id", "accepted_user_id",
			"accepted_management_session_id", "accepted_at", "revision", "created_at", "updated_at",
		}).AddRow(
			nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil, nil,
		).CloseError(closeErr)).
		RowsWillBeClosed()

	_, err = (&Store{database: database}).List(context.Background(), invitationmanagement.InvitationQuery{
		NamespaceID: "namespace-id",
		Now:         time.Now().UTC(),
		Limit:       10,
	})
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("List() error = %v, want joined rows close error", err)
	}
	if !strings.Contains(err.Error(), "scan invitation page") {
		t.Fatalf("List() error = %v, want original scan error", err)
	}

	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}
