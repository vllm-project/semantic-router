package postgres

import (
	"context"
	"errors"
	"regexp"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	sessionID   = "11111111-1111-4111-8111-111111111111"
	principalID = "22222222-2222-4222-8222-222222222222"
	sourceID    = "33333333-3333-4333-8333-333333333333"
)

var testNow = time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)

func newMockStore(t *testing.T) (*Store, sqlmock.Sqlmock) {
	t.Helper()
	database, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatalf("sqlmock.New() error = %v", err)
	}
	t.Cleanup(func() { _ = database.Close() })
	store, err := New(database)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	return store, mock
}

func queryPattern(query string) string { return regexp.QuoteMeta(query) }

func humanSessionRows(tokenID string, status string, revokedAt any) *sqlmock.Rows {
	return sqlmock.NewRows([]string{
		"id", "principal_id", "issuer_session_id", "token_id", "audience",
		"auth_source_kind", "auth_source_id", "evidence_kind", "assurance",
		"authenticated_at", "source_assured_at", "expires_at", "status",
		"revoked_at", "created_at", "principal_status", "auth_source_status",
		"source_not_before", "source_expires_at", "source_assured_at_current",
	}).AddRow(
		sessionID, principalID, "issuer-session", tokenID, "vllm-sr-management",
		"issuer", sourceID, "human", []byte(`{"aal":"aal2","amr":["pwd","otp"]}`),
		testNow.Add(-5*time.Minute), nil, testNow.Add(8*time.Hour), status,
		revokedAt, testNow.Add(-time.Minute), "active", "active", nil, nil, nil,
	)
}

func workloadSessionRows(tokenID string) *sqlmock.Rows {
	assuredAt := testNow.Add(-time.Hour).Truncate(time.Second)
	return sqlmock.NewRows([]string{
		"id", "principal_id", "issuer_session_id", "token_id", "audience",
		"auth_source_kind", "auth_source_id", "evidence_kind", "assurance",
		"authenticated_at", "source_assured_at", "expires_at", "status",
		"revoked_at", "created_at", "principal_status", "auth_source_status",
		"source_not_before", "source_expires_at", "source_assured_at_current",
	}).AddRow(
		sessionID, principalID, nil, tokenID, "vllm-sr-management",
		"service_credential", sourceID, "workload", []byte(`{"class":"workload_strong"}`),
		testNow.Add(-time.Minute), assuredAt, testNow.Add(8*time.Hour), "active",
		nil, testNow.Add(-time.Minute), "active", "active", testNow.Add(-2*time.Hour),
		testNow.Add(24*time.Hour), assuredAt,
	)
}

func assertExpectations(t *testing.T, mock sqlmock.Sqlmock) {
	t.Helper()
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet SQL expectations: %v", err)
	}
}

func TestGetHumanSession(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).
		WillReturnRows(humanSessionRows("token-current", "active", nil))

	live, err := store.Get(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if live.Human == nil || live.Workload != nil || live.Human.AAL != "aal2" ||
		live.PrincipalStatus != managementauth.ResourceActive || live.AuthSourceStatus != managementauth.ResourceActive {
		t.Fatalf("Get() = %+v", live)
	}
	if err := live.ValidateAt(testNow); err != nil {
		t.Fatalf("ValidateAt() error = %v", err)
	}
	assertExpectations(t, mock)
}

func TestGetWorkloadSession(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).
		WillReturnRows(workloadSessionRows("token-current"))

	live, err := store.Get(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if live.Workload == nil || live.Human != nil || live.Workload.Class != "workload_strong" {
		t.Fatalf("Get() = %+v", live)
	}
	if err := live.ValidateAt(testNow); err != nil {
		t.Fatalf("ValidateAt() error = %v", err)
	}
	assertExpectations(t, mock)
}

func TestGetFailsClosedForMissingSourceProjection(t *testing.T) {
	store, mock := newMockStore(t)
	// Rebuild the row with a missing source status; SQL NULL must never mean allow.
	rows := sqlmock.NewRows([]string{
		"id", "principal_id", "issuer_session_id", "token_id", "audience",
		"auth_source_kind", "auth_source_id", "evidence_kind", "assurance",
		"authenticated_at", "source_assured_at", "expires_at", "status",
		"revoked_at", "created_at", "principal_status", "auth_source_status",
		"source_not_before", "source_expires_at", "source_assured_at_current",
	}).AddRow(
		sessionID, principalID, "issuer-session", "token-current", "vllm-sr-management",
		"issuer", sourceID, "human", []byte(`{"aal":"aal2","amr":["pwd"]}`),
		testNow.Add(-5*time.Minute), nil, testNow.Add(time.Hour), "active", nil,
		testNow.Add(-time.Minute), "active", nil, nil, nil, nil,
	)
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).WillReturnRows(rows)

	live, err := store.Get(context.Background(), sessionID)
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}
	if !errors.Is(live.ValidateAt(testNow), managementauth.ErrSessionInactive) {
		t.Fatalf("ValidateAt() error = %v", live.ValidateAt(testNow))
	}
	assertExpectations(t, mock)
}

func TestCreateUsesDatabaseTimePolicyLimitAndSourceState(t *testing.T) {
	store, mock := newMockStore(t)
	draft := managementauth.SessionDraft{
		ID: sessionID, PrincipalID: principalID, TokenID: "token-current",
		IssuerSessionID: pointer("issuer-session"), Audience: "vllm-sr-management",
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: sourceID,
		EvidenceKind: managementauth.EvidenceHuman,
		Human: &managementauth.HumanEvidence{
			AuthenticationTime: testNow.Add(-5 * time.Minute).Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"},
		},
		AuthenticatedAt: testNow.Add(-5 * time.Minute), EvidenceExpiresAt: testNow.Add(8 * time.Hour),
	}
	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(lockPrincipalQuery)).WithArgs(principalID).
		WillReturnRows(sqlmock.NewRows([]string{"status"}).AddRow("active"))
	mock.ExpectQuery(queryPattern(loadSessionPolicyQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"now", "ttl", "max"}).AddRow(testNow, int64(3600), 5))
	mock.ExpectExec(queryPattern(expireSessionsQuery)).WithArgs(principalID, testNow).
		WillReturnResult(sqlmock.NewResult(0, 2))
	mock.ExpectQuery(queryPattern(countActiveSessionsQuery)).WithArgs(principalID, testNow).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(2))
	mock.ExpectExec(queryPattern(insertSessionQuery)).WithArgs(
		sessionID, principalID, sqlmock.AnyArg(), "token-current", "vllm-sr-management",
		managementauth.AuthSourceIssuer, sourceID, managementauth.EvidenceHuman, sqlmock.AnyArg(),
		draft.AuthenticatedAt, nil, testNow.Add(time.Hour), testNow,
	).WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).
		WillReturnRows(humanSessionRows("token-current", "active", nil))
	mock.ExpectCommit()

	live, err := store.Create(context.Background(), draft)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if live.ID != sessionID {
		t.Fatalf("Create() = %+v", live)
	}
	assertExpectations(t, mock)
}

func TestCreateRejectsActiveSessionLimit(t *testing.T) {
	store, mock := newMockStore(t)
	draft := managementauth.SessionDraft{
		ID: sessionID, PrincipalID: principalID, TokenID: "token-current", Audience: "vllm-sr-management",
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: sourceID,
		EvidenceKind:    managementauth.EvidenceHuman,
		Human:           &managementauth.HumanEvidence{AuthenticationTime: testNow.Add(-time.Minute).Unix(), AAL: "aal2", AMR: []string{"pwd"}},
		AuthenticatedAt: testNow.Add(-time.Minute), EvidenceExpiresAt: testNow.Add(time.Hour),
	}
	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(lockPrincipalQuery)).WithArgs(principalID).
		WillReturnRows(sqlmock.NewRows([]string{"status"}).AddRow("active"))
	mock.ExpectQuery(queryPattern(loadSessionPolicyQuery)).
		WillReturnRows(sqlmock.NewRows([]string{"now", "ttl", "max"}).AddRow(testNow, int64(3600), 5))
	mock.ExpectExec(queryPattern(expireSessionsQuery)).WithArgs(principalID, testNow).
		WillReturnResult(sqlmock.NewResult(0, 0))
	mock.ExpectQuery(queryPattern(countActiveSessionsQuery)).WithArgs(principalID, testNow).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(5))
	mock.ExpectRollback()

	if _, err := store.Create(context.Background(), draft); !errors.Is(err, managementauth.ErrSessionLimitExceeded) {
		t.Fatalf("Create() error = %v", err)
	}
	assertExpectations(t, mock)
}

func TestRotateTokenIDIsCompareAndSwap(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(lockSessionQuery)).WithArgs(sessionID).
		WillReturnRows(sqlmock.NewRows([]string{"token_id", "status", "expires_at", "revoked_at", "now"}).
			AddRow("token-current", "active", testNow.Add(time.Hour), nil, testNow))
	mock.ExpectExec(queryPattern(rotateSessionTokenIDQuery)).
		WithArgs(sessionID, "token-current", "token-next").WillReturnResult(sqlmock.NewResult(0, 1))
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).
		WillReturnRows(humanSessionRows("token-next", "active", nil))
	mock.ExpectCommit()

	live, err := store.RotateTokenID(context.Background(), sessionID, "token-current", "token-next")
	if err != nil {
		t.Fatalf("RotateTokenID() error = %v", err)
	}
	if live.TokenID != "token-next" {
		t.Fatalf("RotateTokenID() = %+v", live)
	}
	assertExpectations(t, mock)
}

func TestRotateTokenIDRejectsStaleTokenID(t *testing.T) {
	store, mock := newMockStore(t)
	mock.ExpectBegin()
	mock.ExpectQuery(queryPattern(lockSessionQuery)).WithArgs(sessionID).
		WillReturnRows(sqlmock.NewRows([]string{"token_id", "status", "expires_at", "revoked_at", "now"}).
			AddRow("token-newer", "active", testNow.Add(time.Hour), nil, testNow))
	mock.ExpectRollback()

	if _, err := store.RotateTokenID(context.Background(), sessionID, "token-current", "token-next"); !errors.Is(err, managementauth.ErrSessionConflict) {
		t.Fatalf("RotateTokenID() error = %v", err)
	}
	assertExpectations(t, mock)
}

func TestRevokeIsCASAndIdempotent(t *testing.T) {
	t.Run("first revoke", func(t *testing.T) {
		store, mock := newMockStore(t)
		mock.ExpectBegin()
		mock.ExpectQuery(queryPattern(lockSessionQuery)).WithArgs(sessionID).
			WillReturnRows(sqlmock.NewRows([]string{"token_id", "status", "expires_at", "revoked_at", "now"}).
				AddRow("token-current", "active", testNow.Add(time.Hour), nil, testNow))
		mock.ExpectQuery(queryPattern(revokeSessionQuery)).WithArgs(sessionID, "token-current").
			WillReturnRows(sqlmock.NewRows([]string{"revoked_at"}).AddRow(testNow))
		mock.ExpectCommit()

		mutation, err := store.Revoke(context.Background(), sessionID, "token-current")
		if err != nil || !mutation.Changed || !mutation.ChangedAt.Equal(testNow) {
			t.Fatalf("Revoke() = %+v, %v", mutation, err)
		}
		assertExpectations(t, mock)
	})

	t.Run("idempotent retry", func(t *testing.T) {
		store, mock := newMockStore(t)
		mock.ExpectBegin()
		mock.ExpectQuery(queryPattern(lockSessionQuery)).WithArgs(sessionID).
			WillReturnRows(sqlmock.NewRows([]string{"token_id", "status", "expires_at", "revoked_at", "now"}).
				AddRow("token-current", "revoked", testNow.Add(time.Hour), testNow, testNow.Add(time.Minute)))
		mock.ExpectCommit()

		mutation, err := store.Revoke(context.Background(), sessionID, "token-current")
		if err != nil || mutation.Changed {
			t.Fatalf("Revoke() = %+v, %v", mutation, err)
		}
		assertExpectations(t, mock)
	})
}

func TestGetRejectsUnknownAssuranceFields(t *testing.T) {
	store, mock := newMockStore(t)
	rows := sqlmock.NewRows([]string{
		"id", "principal_id", "issuer_session_id", "token_id", "audience",
		"auth_source_kind", "auth_source_id", "evidence_kind", "assurance",
		"authenticated_at", "source_assured_at", "expires_at", "status",
		"revoked_at", "created_at", "principal_status", "auth_source_status",
		"source_not_before", "source_expires_at", "source_assured_at_current",
	}).AddRow(
		sessionID, principalID, nil, "token-current", "vllm-sr-management", "issuer", sourceID,
		"human", []byte(`{"aal":"aal2","amr":["pwd"],"roles":["admin"]}`),
		testNow.Add(-time.Minute), nil, testNow.Add(time.Hour), "active", nil,
		testNow.Add(-time.Minute), "active", "active", nil, nil, nil,
	)
	mock.ExpectQuery(queryPattern(liveSessionQuery)).WithArgs(sessionID).WillReturnRows(rows)

	if _, err := store.Get(context.Background(), sessionID); err == nil {
		t.Fatal("Get() accepted unknown assurance fields")
	}
	assertExpectations(t, mock)
}

func pointer(value string) *string { return &value }
