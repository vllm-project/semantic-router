package postgres

import (
	"context"
	"errors"
	"regexp"
	"strings"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func TestGetPublicationModelIDsJoinsRowsCloseErrorOnScanFailure(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}

	closeErr := errors.New("rows close failed")
	mock.ExpectQuery(regexp.QuoteMeta("SELECT DISTINCT assignment.model_id")).
		WithArgs("namespace-id", "plan-id").
		WillReturnRows(sqlmock.NewRows([]string{"model_id"}).
			AddRow(nil).
			CloseError(closeErr)).
		RowsWillBeClosed()

	_, err = (&Store{db: database}).GetPublicationModelIDs(
		context.Background(), "namespace-id", "plan-id",
	)
	if err == nil || !errors.Is(err, closeErr) {
		t.Fatalf("GetPublicationModelIDs() error = %v, want joined rows close error", err)
	}
	if !strings.Contains(err.Error(), "converting NULL") {
		t.Fatalf("GetPublicationModelIDs() error = %v, want original scan error", err)
	}

	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}
