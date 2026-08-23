package runtimediagnostics

import (
	"context"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
)

func TestReadUsageStorageReportsBoundedLifecycleState(t *testing.T) {
	database, mock, err := sqlmock.New()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	oldest := time.Date(2025, time.January, 1, 0, 0, 0, 0, time.UTC)
	createdThrough := time.Date(2026, time.October, 1, 0, 0, 0, 0, time.UTC)
	mock.ExpectQuery(regexp.QuoteMeta(`SELECT
  count(*) FILTER (WHERE state='active'),
  count(*) FILTER (WHERE state='retired'),
  min(month_start) FILTER (WHERE state='active'),
  max(month_start) FILTER (WHERE state='active'),`)).
		WithArgs(usageStorageDiagnosticBucketLimit + 1).
		WillReturnRows(sqlmock.NewRows([]string{
			"active", "retired", "oldest", "created_through", "minutes", "hours", "days",
		}).AddRow(4, 2, oldest, createdThrough, usageStorageDiagnosticBucketLimit+1, 7, 0))

	status, err := (&Service{database: database}).readUsageStorage(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if status.Status != "ready" || status.ActiveMonths != 4 || status.RetiredMonths != 2 ||
		status.DirtyMinuteBuckets != usageStorageDiagnosticBucketLimit ||
		status.DirtyHourBuckets != 7 || status.DirtyDayBuckets != 0 || !status.DirtyCountsCapped {
		t.Fatalf("usage storage diagnostics = %+v", status)
	}
	if status.OldestActiveMonth == nil || !status.OldestActiveMonth.Equal(oldest) ||
		status.CreatedThrough == nil || !status.CreatedThrough.Equal(createdThrough) {
		t.Fatalf("usage storage diagnostic range = %+v", status)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
