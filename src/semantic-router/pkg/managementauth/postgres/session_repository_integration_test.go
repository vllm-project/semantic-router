package postgres

import (
	"context"
	"database/sql"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"

	controlplanepostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

func TestStoreIntegrationLifecycle(t *testing.T) {
	fixture := newSessionLifecycleFixture(t)
	newTokenID := createAndRotateLifecycleSession(t, fixture)
	reissueDraft := reissueLifecycleSession(t, fixture, newTokenID)
	assertConcurrentLifecycleReissue(t, fixture, reissueDraft, newTokenID)
	assertLifecycleSessionDurability(t, fixture, newTokenID)
	assertLifecycleSessionRevocation(t, fixture, newTokenID)
}

type sessionLifecycleFixture struct {
	database    *sql.DB
	ctx         context.Context
	store       *Store
	principalID string
	sessionID   string
	draft       managementauth.SessionDraft
}

type replicaReissueOutcome struct {
	live   managementauth.LiveSession
	issued managementauth.IssuedToken
	found  bool
	err    error
}

func newSessionLifecycleFixture(t *testing.T) sessionLifecycleFixture {
	t.Helper()
	dsn := os.Getenv("MANAGEMENTAUTH_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("MANAGEMENTAUTH_TEST_POSTGRES_DSN is not configured")
	}
	database, testStoreIntegrationLifecycleErr := sql.Open("postgres", dsn)
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatalf("open PostgreSQL: %v", testStoreIntegrationLifecycleErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	if err := (controlplanepostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatalf("apply migrations: %v", err)
	}

	principalID := uuid.NewString()
	issuerID := uuid.NewString()
	sessionID := uuid.NewString()
	now := time.Now().UTC().Truncate(time.Second)
	if _, err := database.ExecContext(ctx, `INSERT INTO management_session_policy (
  singleton, access_token_ttl_seconds, session_ttl_seconds, max_active_sessions,
  action_requirements, seed_version, revision
) VALUES (TRUE, 900, 28800, 5, '{}'::jsonb, 1, 1)
ON CONFLICT (singleton) DO UPDATE SET
  access_token_ttl_seconds = EXCLUDED.access_token_ttl_seconds,
  session_ttl_seconds = EXCLUDED.session_ttl_seconds,
  max_active_sessions = EXCLUDED.max_active_sessions`); err != nil {
		t.Fatalf("seed Management session policy: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO trusted_identity_issuers (
  id, issuer, kind, discovery_url, audiences, claim_mapping, assurance_mapping, status
) VALUES ($1, $2, 'oidc', $3, '["vllm-sr-management"]'::jsonb, '{}'::jsonb, '{}'::jsonb, 'active')`,
		issuerID, "https://issuer.example/"+issuerID, "https://issuer.example/"+issuerID+"/.well-known/openid-configuration"); err != nil {
		t.Fatalf("seed trusted issuer: %v", err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals (
  id, issuer, subject, display_name, status
) VALUES ($1, $2, $3, 'Integration User', 'active')`,
		principalID, "https://issuer.example/"+issuerID, "subject-"+principalID); err != nil {
		t.Fatalf("seed Management principal: %v", err)
	}
	store, testStoreIntegrationLifecycleErr := New(database)
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatal(testStoreIntegrationLifecycleErr)
	}
	draft := managementauth.SessionDraft{
		ID: sessionID, PrincipalID: principalID, IssuerSessionID: stringPointer("issuer-session-" + sessionID),
		TokenID: "token-initial-" + sessionID, Audience: "vllm-sr-management",
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: issuerID,
		EvidenceKind: managementauth.EvidenceHuman,
		Human: &managementauth.HumanEvidence{
			AuthenticationTime: now.Add(-time.Minute).Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"},
		},
		AuthenticatedAt: now.Add(-time.Minute), EvidenceExpiresAt: now.Add(2 * time.Hour),
	}
	return sessionLifecycleFixture{
		database: database, ctx: ctx, store: store, principalID: principalID, sessionID: sessionID, draft: draft,
	}
}

func createAndRotateLifecycleSession(t *testing.T, fixture sessionLifecycleFixture) string {
	t.Helper()
	live, err := fixture.store.Create(fixture.ctx, fixture.draft)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if err := live.ValidateAt(time.Now().UTC()); err != nil {
		t.Fatalf("created session ValidateAt() error = %v", err)
	}
	newTokenID := "token-refreshed-" + fixture.sessionID
	refreshed, err := fixture.store.RotateTokenID(
		fixture.ctx, fixture.sessionID, fixture.draft.TokenID, newTokenID,
	)
	if err != nil {
		t.Fatalf("RotateTokenID() error = %v", err)
	}
	if refreshed.TokenID != newTokenID {
		t.Fatalf("RotateTokenID() token id = %q", refreshed.TokenID)
	}
	return newTokenID
}

func reissueLifecycleSession(
	t *testing.T,
	fixture sessionLifecycleFixture,
	newTokenID string,
) managementauth.SessionDraft {
	t.Helper()
	reissueDraft := fixture.draft
	reissueDraft.ID = uuid.NewString()
	reissueDraft.TokenID = "token-reissued-" + fixture.sessionID
	tx, err := fixture.database.BeginTx(fixture.ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		t.Fatal(err)
	}
	reissued, envelope, found, err := fixture.store.ReissueIssuerSessionInTransaction(
		fixture.ctx,
		tx,
		reissueDraft,
		func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
			return managementauth.IssuedToken{
				AccessToken: "access-reissued", TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
				ManagementSessionID: session.ID,
			}, nil
		},
	)
	if err != nil || !found || reissued.ID != fixture.sessionID ||
		reissued.TokenID != newTokenID || envelope.ManagementSessionID != fixture.sessionID {
		_ = tx.Rollback()
		t.Fatalf("ReissueIssuerSessionInTransaction() = %+v, %+v, %t, %v",
			reissued, envelope, found, err)
	}
	if err = tx.Commit(); err != nil {
		t.Fatal(err)
	}
	return reissueDraft
}

func assertConcurrentLifecycleReissue(
	t *testing.T,
	fixture sessionLifecycleFixture,
	reissueDraft managementauth.SessionDraft,
	newTokenID string,
) {
	t.Helper()
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	outcomes := make(chan replicaReissueOutcome, 2)
	for replica := 0; replica < 2; replica++ {
		go func() {
			ready <- struct{}{}
			<-start
			replicaTx, err := fixture.database.BeginTx(
				fixture.ctx, &sql.TxOptions{Isolation: sql.LevelSerializable},
			)
			if err != nil {
				outcomes <- replicaReissueOutcome{err: err}
				return
			}
			defer func() { _ = replicaTx.Rollback() }()
			replicaDraft := reissueDraft
			replicaDraft.ID = uuid.NewString()
			replicaDraft.TokenID = "replica-token-" + uuid.NewString()
			live, issued, found, err := fixture.store.ReissueIssuerSessionInTransaction(
				fixture.ctx,
				replicaTx,
				replicaDraft,
				func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
					return managementauth.IssuedToken{
						AccessToken: "replica-access", TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
						ManagementSessionID: session.ID,
					}, nil
				},
			)
			if err == nil {
				err = replicaTx.Commit()
			}
			outcomes <- replicaReissueOutcome{live: live, issued: issued, found: found, err: err}
		}()
	}
	<-ready
	<-ready
	close(start)
	for replica := 0; replica < 2; replica++ {
		outcome := <-outcomes
		if outcome.err != nil || !outcome.found || outcome.live.ID != fixture.sessionID ||
			outcome.live.TokenID != newTokenID || outcome.issued.ManagementSessionID != fixture.sessionID {
			t.Fatalf("replica ReissueIssuerSessionInTransaction() = %+v", outcome)
		}
	}
}

func assertLifecycleSessionDurability(t *testing.T, fixture sessionLifecycleFixture, newTokenID string) {
	t.Helper()
	var sessionCount int
	var persistedTokenID string
	err := fixture.database.QueryRowContext(
		fixture.ctx,
		`SELECT count(*), max(token_id) FROM management_sessions WHERE principal_id=$1`,
		fixture.principalID,
	).Scan(&sessionCount, &persistedTokenID)
	if err != nil || sessionCount != 1 || persistedTokenID != newTokenID {
		t.Fatalf("durable session state = count %d, token %q, %v", sessionCount, persistedTokenID, err)
	}
	if _, err := fixture.store.RotateTokenID(
		fixture.ctx, fixture.sessionID, fixture.draft.TokenID, "stale-token-"+fixture.sessionID,
	); !errors.Is(err, managementauth.ErrSessionConflict) {
		t.Fatalf("stale RotateTokenID() error = %v", err)
	}
}

func assertLifecycleSessionRevocation(t *testing.T, fixture sessionLifecycleFixture, newTokenID string) {
	t.Helper()
	mutation, err := fixture.store.Revoke(fixture.ctx, fixture.sessionID, newTokenID)
	if err != nil || !mutation.Changed {
		t.Fatalf("Revoke() = %+v, %v", mutation, err)
	}
	replayed, err := fixture.store.Revoke(fixture.ctx, fixture.sessionID, newTokenID)
	if err != nil || replayed.Changed {
		t.Fatalf("idempotent Revoke() = %+v, %v", replayed, err)
	}
	revoked, err := fixture.store.Get(fixture.ctx, fixture.sessionID)
	if err != nil {
		t.Fatalf("Get() revoked error = %v", err)
	}
	if !errors.Is(revoked.ValidateAt(time.Now().UTC()), managementauth.ErrSessionInactive) {
		t.Fatalf("revoked ValidateAt() error = %v", revoked.ValidateAt(time.Now().UTC()))
	}
}

func TestConcurrentFirstIssuerExchangeReusesOneDurableSession(t *testing.T) {
	fixture := newConcurrentIssuerExchangeFixture(t)
	first, second := raceConcurrentIssuerExchanges(fixture)
	assertConcurrentIssuerExchange(t, fixture, first, second)
	changedIdentity, changedDraft := assertChangedIssuerEvidenceCreatesSession(t, fixture, first)
	assertDistinctIssuerSessionLimit(t, fixture, changedIdentity, changedDraft)
}

type concurrentIssuerExchangeFixture struct {
	database        *sql.DB
	ctx             context.Context
	coordinator     *IdentityExchangeCoordinator
	issuerID        string
	principalID     string
	issuerSessionID string
	identity        managementauth.VerifiedExternalIdentity
	baseDraft       managementauth.SessionDraft
}

type coordinatedExchangeOutcome struct {
	result managementauth.IdentityExchangeResult
	err    error
}

func newConcurrentIssuerExchangeFixture(t *testing.T) concurrentIssuerExchangeFixture {
	t.Helper()
	database, ctx := authCoordinationDatabase(t)
	issuerID := seedCoordinationIssuer(t, ctx, database)
	principalID := uuid.NewString()
	issuer := "https://issuer.example/" + issuerID
	subject := "concurrent-subject-" + principalID
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,$2,$3,'Concurrent Exchange User','active')`, principalID, issuer, subject); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `UPDATE management_session_policy
SET max_active_sessions=2 WHERE singleton=TRUE`); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_, _ = database.ExecContext(context.Background(), `UPDATE management_session_policy
SET max_active_sessions=5 WHERE singleton=TRUE`)
	})
	sessions, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	coordinator, err := NewIdentityExchangeCoordinator(database, sessions)
	if err != nil {
		t.Fatal(err)
	}
	issuerSessionID := "concurrent-dashboard-session-" + uuid.NewString()
	authenticatedAt := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	identity := managementauth.VerifiedExternalIdentity{
		IssuerID: issuerID, Issuer: issuer, Subject: subject,
		IssuerSessionID: &issuerSessionID, AAL: "aal2", AMR: []string{"pwd", "otp"},
		AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
	}
	baseDraft := managementauth.SessionDraft{
		IssuerSessionID: &issuerSessionID, Audience: "vllm-sr-management",
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: issuerID,
		EvidenceKind: managementauth.EvidenceHuman,
		Human: &managementauth.HumanEvidence{
			AuthenticationTime: authenticatedAt.Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"},
		},
		AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
	}
	return concurrentIssuerExchangeFixture{
		database: database, ctx: ctx, coordinator: coordinator, issuerID: issuerID,
		principalID: principalID, issuerSessionID: issuerSessionID,
		identity: identity, baseDraft: baseDraft,
	}
}

func raceConcurrentIssuerExchanges(
	fixture concurrentIssuerExchangeFixture,
) (coordinatedExchangeOutcome, coordinatedExchangeOutcome) {
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	outcomes := make(chan coordinatedExchangeOutcome, 2)
	for replica := 0; replica < 2; replica++ {
		go func() {
			ready <- struct{}{}
			<-start
			draft := fixture.baseDraft
			draft.ID = uuid.NewString()
			draft.TokenID = "replica-token-" + uuid.NewString()
			result, err := fixture.coordinator.ExchangeIdentity(fixture.ctx, managementauth.IdentityExchangeRequest{
				Identity: fixture.identity,
				Session:  draft,
			}, issueIdentityExchangeSession)
			outcomes <- coordinatedExchangeOutcome{result: result, err: err}
		}()
	}
	<-ready
	<-ready
	close(start)
	return <-outcomes, <-outcomes
}

func assertConcurrentIssuerExchange(
	t *testing.T,
	fixture concurrentIssuerExchangeFixture,
	first, second coordinatedExchangeOutcome,
) {
	t.Helper()
	if first.err != nil || second.err != nil || first.result.Issued.ManagementSessionID == "" ||
		first.result.Issued.ManagementSessionID != second.result.Issued.ManagementSessionID ||
		first.result.Issued.AccessToken == "" || first.result.Issued.AccessToken != second.result.Issued.AccessToken {
		t.Fatalf("concurrent issuer exchanges = %+v / %+v", first, second)
	}
	var count int
	var tokenID string
	if err := fixture.database.QueryRowContext(fixture.ctx, `SELECT count(*), max(token_id)
FROM management_sessions
WHERE principal_id=$1 AND auth_source_id=$2 AND issuer_session_id=$3`,
		fixture.principalID, fixture.issuerID, fixture.issuerSessionID,
	).Scan(&count, &tokenID); err != nil {
		t.Fatal(err)
	}
	if count != 1 || tokenID != first.result.Issued.AccessToken {
		t.Fatalf("concurrent durable issuer session = count %d, token %q", count, tokenID)
	}
}

func assertChangedIssuerEvidenceCreatesSession(
	t *testing.T,
	fixture concurrentIssuerExchangeFixture,
	first coordinatedExchangeOutcome,
) (managementauth.VerifiedExternalIdentity, managementauth.SessionDraft) {
	t.Helper()
	changedAuthenticatedAt := fixture.identity.AuthenticatedAt.Add(30 * time.Second)
	changedIdentity := fixture.identity
	changedIdentity.AuthenticatedAt = changedAuthenticatedAt
	changedIdentity.AAL = "aal3"
	changedDraft := fixture.baseDraft
	changedDraft.ID = uuid.NewString()
	changedDraft.TokenID = "changed-evidence-token-" + uuid.NewString()
	changedDraft.AuthenticatedAt = changedAuthenticatedAt
	changedDraft.Human = &managementauth.HumanEvidence{
		AuthenticationTime: changedAuthenticatedAt.Unix(), AAL: "aal3", AMR: []string{"pwd", "otp"},
	}
	changed, err := fixture.coordinator.ExchangeIdentity(fixture.ctx, managementauth.IdentityExchangeRequest{
		Identity: changedIdentity,
		Session:  changedDraft,
	}, issueIdentityExchangeSession)
	if err != nil || changed.Issued.ManagementSessionID == first.result.Issued.ManagementSessionID ||
		changed.Issued.AccessToken == first.result.Issued.AccessToken {
		t.Fatalf("changed-evidence issuer exchange = %+v, %v", changed, err)
	}
	var count int
	if err := fixture.database.QueryRowContext(fixture.ctx, `SELECT count(*) FROM management_sessions
WHERE principal_id=$1 AND status='active'`, fixture.principalID).Scan(&count); err != nil || count != 2 {
		t.Fatalf("active sessions after changed evidence = %d, %v", count, err)
	}
	return changedIdentity, changedDraft
}

func assertDistinctIssuerSessionLimit(
	t *testing.T,
	fixture concurrentIssuerExchangeFixture,
	changedIdentity managementauth.VerifiedExternalIdentity,
	changedDraft managementauth.SessionDraft,
) {
	t.Helper()
	distinctIssuerSessionID := "distinct-dashboard-session-" + uuid.NewString()
	distinctIdentity := changedIdentity
	distinctIdentity.IssuerSessionID = &distinctIssuerSessionID
	distinctDraft := changedDraft
	distinctDraft.ID = uuid.NewString()
	distinctDraft.TokenID = "distinct-session-token-" + uuid.NewString()
	distinctDraft.IssuerSessionID = &distinctIssuerSessionID
	if _, err := fixture.coordinator.ExchangeIdentity(fixture.ctx, managementauth.IdentityExchangeRequest{
		Identity: distinctIdentity,
		Session:  distinctDraft,
	}, issueIdentityExchangeSession); !errors.Is(err, managementauth.ErrSessionLimitExceeded) {
		t.Fatalf("distinct issuer session above active limit error = %v", err)
	}
}

func issueIdentityExchangeSession(
	_ context.Context,
	session managementauth.LiveSession,
	_ time.Time,
) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{
		AccessToken: session.TokenID, TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
		ManagementSessionID: session.ID,
	}, nil
}

func stringPointer(value string) *string { return &value }
