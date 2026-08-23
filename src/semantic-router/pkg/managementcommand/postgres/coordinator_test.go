package postgres

import (
	"context"
	"database/sql"
	"errors"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID = "11111111-1111-4111-8111-111111111111"
	testPrincipalID = "22222222-2222-4222-8222-222222222222"
	testResourceID  = "33333333-3333-4333-8333-333333333333"
)

func TestLockAndCompleteResourceOwnNoTransactionBoundary(t *testing.T) {
	db, mock := newCommandMock(t)
	command, now := testCommand(t, []byte(`{"name":"one"}`), "opaque-key-0123456789")
	active := command.ActiveDigest()
	mock.ExpectBegin()
	tx, err := db.BeginTx(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	mock.ExpectQuery(regexp.QuoteMeta(lockCommandQuery)).
		WithArgs(advisoryKey(command)).
		WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
	mock.ExpectQuery(regexp.QuoteMeta(getLockedCommandQuery)).
		WithArgs(string(managementcommand.ScopeNamespace), testNamespaceID, command.PrincipalID, command.Endpoint, active.HMACVersion, active.KeyDigest[:]).
		WillReturnError(sql.ErrNoRows)
	stored, replayed, err := Lock(context.Background(), tx, command)
	if err != nil || replayed || stored.Resource != nil || stored.Operation != nil {
		t.Fatalf("Lock() = %#v, %t, %v", stored, replayed, err)
	}
	mock.ExpectExec(regexp.QuoteMeta(insertResourceCommandQuery)).
		WithArgs(
			string(managementcommand.ScopeNamespace), testNamespaceID, command.PrincipalID, command.Endpoint,
			active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:], "provider_credential",
			testResourceID, uint64(1), 201, command.ExpiresAt,
		).
		WillReturnResult(sqlmock.NewResult(0, 1))
	if err := CompleteResource(context.Background(), tx, command, managementcommand.ResourceResult{
		ResourceType: "provider_credential", ResourceID: testResourceID,
		ResourceRevision: 1, ResponseStatus: 201,
	}); err != nil {
		t.Fatal(err)
	}
	mock.ExpectCommit()
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	assertCommandExpectations(t, mock)
}

func TestLockReturnsReplayAndConflict(t *testing.T) {
	for name, differentRequest := range map[string]bool{"replay": false, "conflict": true} {
		t.Run(name, func(t *testing.T) {
			db, mock := newCommandMock(t)
			command, now := testCommand(t, []byte(`{"name":"one"}`), "opaque-key-0123456789")
			active := command.ActiveDigest()
			storedDigest := append([]byte(nil), active.RequestDigest[:]...)
			if differentRequest {
				other, _ := testCommand(t, []byte(`{"name":"two"}`), "opaque-key-0123456789")
				otherActive := other.ActiveDigest()
				storedDigest = append([]byte(nil), otherActive.RequestDigest[:]...)
			}
			mock.ExpectBegin()
			tx, err := db.BeginTx(context.Background(), nil)
			if err != nil {
				t.Fatal(err)
			}
			mock.ExpectQuery(regexp.QuoteMeta(lockCommandQuery)).
				WithArgs(advisoryKey(command)).
				WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
			mock.ExpectQuery(regexp.QuoteMeta(getLockedCommandQuery)).
				WithArgs(string(managementcommand.ScopeNamespace), testNamespaceID, command.PrincipalID, command.Endpoint, active.HMACVersion, active.KeyDigest[:]).
				WillReturnRows(commandRows().AddRow(
					storedDigest, nil, "provider_credential", testResourceID,
					int64(7), nil, 201, nil, nil, nil, nil, command.ExpiresAt,
				))
			stored, replayed, err := Lock(context.Background(), tx, command)
			if differentRequest {
				if !errors.Is(err, managementcommand.ErrConflict) || replayed {
					t.Fatalf("conflicting Lock() = %#v, %t, %v", stored, replayed, err)
				}
			} else if err != nil || !replayed || stored.Resource == nil ||
				stored.Resource.ResourceRevision != 7 || stored.Resource.ResponseStatus != 201 {
				t.Fatalf("replayed Lock() = %#v, %t, %v", stored, replayed, err)
			}
			mock.ExpectRollback()
			if err := tx.Rollback(); err != nil {
				t.Fatal(err)
			}
			assertCommandExpectations(t, mock)
		})
	}
}

func TestFailedCommandCanRollBackWithoutConsumingKey(t *testing.T) {
	db, mock := newCommandMock(t)
	command, now := testCommand(t, []byte(`{"name":"one"}`), "opaque-key-0123456789")
	active := command.ActiveDigest()
	for attempt := 0; attempt < 2; attempt++ {
		mock.ExpectBegin()
		tx, err := db.BeginTx(context.Background(), nil)
		if err != nil {
			t.Fatal(err)
		}
		mock.ExpectQuery(regexp.QuoteMeta(lockCommandQuery)).
			WithArgs(advisoryKey(command)).
			WillReturnRows(sqlmock.NewRows([]string{"clock_timestamp"}).AddRow(now))
		mock.ExpectQuery(regexp.QuoteMeta(getLockedCommandQuery)).
			WithArgs(string(managementcommand.ScopeNamespace), testNamespaceID, command.PrincipalID, command.Endpoint, active.HMACVersion, active.KeyDigest[:]).
			WillReturnError(sql.ErrNoRows)
		if _, replayed, err := Lock(context.Background(), tx, command); err != nil || replayed {
			t.Fatalf("attempt %d Lock() replayed=%t err=%v", attempt, replayed, err)
		}
		mock.ExpectRollback()
		if err := tx.Rollback(); err != nil {
			t.Fatal(err)
		}
	}
	assertCommandExpectations(t, mock)
}

func TestValidateReferencedHMACVersionsFailsClosedAndIgnoresExpiredRows(t *testing.T) {
	db, mock := newCommandMock(t)
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v2",
		Keys: map[string][]byte{
			"command-v1": []byte(strings.Repeat("o", 32)),
			"command-v2": []byte(strings.Repeat("n", 32)),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	mock.ExpectQuery(regexp.QuoteMeta(getReferencedHMACVersionsQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"hmac_version"}).AddRow("command-v1").AddRow("retired-v0"))
	if err := ValidateReferencedHMACVersions(context.Background(), db, codec); !errors.Is(err, managementcommand.ErrHMACVersionUnavailable) {
		t.Fatalf("missing live version readiness error = %v", err)
	}
	// The query excludes expired rows. Once command retention/GC has elapsed,
	// removing the old version is safe and readiness succeeds.
	mock.ExpectQuery(regexp.QuoteMeta(getReferencedHMACVersionsQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"hmac_version"}).AddRow("command-v2"))
	if err := ValidateReferencedHMACVersions(context.Background(), db, codec); err != nil {
		t.Fatalf("post-GC readiness = %v", err)
	}
	assertCommandExpectations(t, mock)
}

func TestValidateReferencedHMACVersionsJoinsRowsCloseError(t *testing.T) {
	database, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatalf("create SQL mock: %v", err)
	}
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys:          map[string][]byte{"command-v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatalf("create command codec: %v", err)
	}

	closeErr := errors.New("rows close failed")
	mock.ExpectQuery(regexp.QuoteMeta(getReferencedHMACVersionsQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"hmac_version"}).
			AddRow("retired-v0").
			CloseError(closeErr)).
		RowsWillBeClosed()
	err = ValidateReferencedHMACVersions(context.Background(), database, codec)
	if !errors.Is(err, managementcommand.ErrHMACVersionUnavailable) || !errors.Is(err, closeErr) {
		t.Fatalf("ValidateReferencedHMACVersions() error = %v, want version and rows-close errors", err)
	}

	mock.ExpectClose()
	if err := database.Close(); err != nil {
		t.Fatalf("close SQL mock: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("SQL expectations: %v", err)
	}
}

func newCommandMock(t *testing.T) (*sql.DB, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db, mock
}

func testCommand(t *testing.T, request []byte, key string) (managementcommand.Command, time.Time) {
	t.Helper()
	now := time.Now().UTC().Truncate(time.Microsecond)
	codec, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1", Keys: map[string][]byte{"command-v1": []byte(strings.Repeat("k", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	command, err := codec.Bind(
		managementcommand.NamespaceCommandScope(testNamespaceID), testPrincipalID, "/management/v1/provider-credentials",
		key, request, now, now.Add(time.Hour),
	)
	if err != nil {
		t.Fatal(err)
	}
	return command, now
}

func commandRows() *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"request_digest", "operation_id", "resource_type", "resource_id",
		"resource_revision", "desired_revision", "response_status",
		"secret_response_ciphertext", "secret_response_nonce", "response_kek_version",
		"secret_response_expires_at", "expires_at",
	})
}

func assertCommandExpectations(t *testing.T, mock sqlmock.Sqlmock) {
	t.Helper()
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatal(err)
	}
}
