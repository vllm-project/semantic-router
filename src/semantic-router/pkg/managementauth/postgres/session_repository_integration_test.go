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
	defer cancel()
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
	live, testStoreIntegrationLifecycleErr := store.Create(ctx, draft)
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatalf("Create() error = %v", testStoreIntegrationLifecycleErr)
	}
	if err := live.ValidateAt(time.Now().UTC()); err != nil {
		t.Fatalf("created session ValidateAt() error = %v", err)
	}

	newTokenID := "token-refreshed-" + sessionID
	refreshed, testStoreIntegrationLifecycleErr := store.RotateTokenID(ctx, sessionID, draft.TokenID, newTokenID)
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatalf("RotateTokenID() error = %v", testStoreIntegrationLifecycleErr)
	}
	if refreshed.TokenID != newTokenID {
		t.Fatalf("RotateTokenID() token id = %q", refreshed.TokenID)
	}
	reissueDraft := draft
	reissueDraft.ID = uuid.NewString()
	reissueDraft.TokenID = "token-reissued-" + sessionID
	tx, testStoreIntegrationLifecycleErr := database.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatal(testStoreIntegrationLifecycleErr)
	}
	reissued, envelope, found, testStoreIntegrationLifecycleErr := store.ReissueIssuerSessionInTransaction(
		ctx,
		tx,
		reissueDraft,
		func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
			return managementauth.IssuedToken{
				AccessToken: "access-reissued", TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
				ManagementSessionID: session.ID,
			}, nil
		},
	)
	if testStoreIntegrationLifecycleErr != nil || !found || reissued.ID != sessionID ||
		reissued.TokenID != newTokenID || envelope.ManagementSessionID != sessionID {
		_ = tx.Rollback()
		t.Fatalf("ReissueIssuerSessionInTransaction() = %+v, %+v, %t, %v",
			reissued, envelope, found, testStoreIntegrationLifecycleErr)
	}
	if testStoreIntegrationLifecycleErr = tx.Commit(); testStoreIntegrationLifecycleErr != nil {
		t.Fatal(testStoreIntegrationLifecycleErr)
	}
	type replicaReissueOutcome struct {
		live   managementauth.LiveSession
		issued managementauth.IssuedToken
		found  bool
		err    error
	}
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	outcomes := make(chan replicaReissueOutcome, 2)
	for replica := 0; replica < 2; replica++ {
		go func() {
			ready <- struct{}{}
			<-start
			replicaTx, err := database.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
			if err != nil {
				outcomes <- replicaReissueOutcome{err: err}
				return
			}
			defer func() { _ = replicaTx.Rollback() }()
			replicaDraft := reissueDraft
			replicaDraft.ID = uuid.NewString()
			replicaDraft.TokenID = "replica-token-" + uuid.NewString()
			live, issued, found, err := store.ReissueIssuerSessionInTransaction(
				ctx,
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
		if outcome.err != nil || !outcome.found || outcome.live.ID != sessionID ||
			outcome.live.TokenID != newTokenID || outcome.issued.ManagementSessionID != sessionID {
			t.Fatalf("replica ReissueIssuerSessionInTransaction() = %+v", outcome)
		}
	}
	var sessionCount int
	var persistedTokenID string
	if testStoreIntegrationLifecycleErr = database.QueryRowContext(
		ctx,
		`SELECT count(*), max(token_id) FROM management_sessions WHERE principal_id=$1`,
		principalID,
	).Scan(&sessionCount, &persistedTokenID); testStoreIntegrationLifecycleErr != nil || sessionCount != 1 || persistedTokenID != newTokenID {
		t.Fatalf("durable session state = count %d, token %q, %v", sessionCount, persistedTokenID, testStoreIntegrationLifecycleErr)
	}
	if _, err := store.RotateTokenID(ctx, sessionID, draft.TokenID, "stale-token-"+sessionID); !errors.Is(err, managementauth.ErrSessionConflict) {
		t.Fatalf("stale RotateTokenID() error = %v", err)
	}

	mutation, testStoreIntegrationLifecycleErr := store.Revoke(ctx, sessionID, newTokenID)
	if testStoreIntegrationLifecycleErr != nil || !mutation.Changed {
		t.Fatalf("Revoke() = %+v, %v", mutation, testStoreIntegrationLifecycleErr)
	}
	replayed, testStoreIntegrationLifecycleErr := store.Revoke(ctx, sessionID, newTokenID)
	if testStoreIntegrationLifecycleErr != nil || replayed.Changed {
		t.Fatalf("idempotent Revoke() = %+v, %v", replayed, testStoreIntegrationLifecycleErr)
	}
	revoked, testStoreIntegrationLifecycleErr := store.Get(ctx, sessionID)
	if testStoreIntegrationLifecycleErr != nil {
		t.Fatalf("Get() revoked error = %v", testStoreIntegrationLifecycleErr)
	}
	if !errors.Is(revoked.ValidateAt(time.Now().UTC()), managementauth.ErrSessionInactive) {
		t.Fatalf("revoked ValidateAt() error = %v", revoked.ValidateAt(time.Now().UTC()))
	}
}

func TestConcurrentFirstIssuerExchangeReusesOneDurableSession(t *testing.T) {
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
	type exchangeOutcome struct {
		result managementauth.IdentityExchangeResult
		err    error
	}
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	outcomes := make(chan exchangeOutcome, 2)
	for replica := 0; replica < 2; replica++ {
		go func() {
			ready <- struct{}{}
			<-start
			draft := baseDraft
			draft.ID = uuid.NewString()
			draft.TokenID = "replica-token-" + uuid.NewString()
			result, err := coordinator.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
				Identity: identity,
				Session:  draft,
			}, func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
				return managementauth.IssuedToken{
					AccessToken: session.TokenID, TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
					ManagementSessionID: session.ID,
				}, nil
			})
			outcomes <- exchangeOutcome{result: result, err: err}
		}()
	}
	<-ready
	<-ready
	close(start)
	first, second := <-outcomes, <-outcomes
	if first.err != nil || second.err != nil || first.result.Issued.ManagementSessionID == "" ||
		first.result.Issued.ManagementSessionID != second.result.Issued.ManagementSessionID ||
		first.result.Issued.AccessToken == "" || first.result.Issued.AccessToken != second.result.Issued.AccessToken {
		t.Fatalf("concurrent issuer exchanges = %+v / %+v", first, second)
	}
	var count int
	var tokenID string
	if err := database.QueryRowContext(ctx, `SELECT count(*), max(token_id)
FROM management_sessions
WHERE principal_id=$1 AND auth_source_id=$2 AND issuer_session_id=$3`,
		principalID, issuerID, issuerSessionID,
	).Scan(&count, &tokenID); err != nil {
		t.Fatal(err)
	}
	if count != 1 || tokenID != first.result.Issued.AccessToken {
		t.Fatalf("concurrent durable issuer session = count %d, token %q", count, tokenID)
	}

	changedAuthenticatedAt := authenticatedAt.Add(30 * time.Second)
	changedIdentity := identity
	changedIdentity.AuthenticatedAt = changedAuthenticatedAt
	changedIdentity.AAL = "aal3"
	changedDraft := baseDraft
	changedDraft.ID = uuid.NewString()
	changedDraft.TokenID = "changed-evidence-token-" + uuid.NewString()
	changedDraft.AuthenticatedAt = changedAuthenticatedAt
	changedDraft.Human = &managementauth.HumanEvidence{
		AuthenticationTime: changedAuthenticatedAt.Unix(), AAL: "aal3", AMR: []string{"pwd", "otp"},
	}
	changed, err := coordinator.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
		Identity: changedIdentity,
		Session:  changedDraft,
	}, func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
		return managementauth.IssuedToken{
			AccessToken: session.TokenID, TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
			ManagementSessionID: session.ID,
		}, nil
	})
	if err != nil || changed.Issued.ManagementSessionID == first.result.Issued.ManagementSessionID ||
		changed.Issued.AccessToken == first.result.Issued.AccessToken {
		t.Fatalf("changed-evidence issuer exchange = %+v, %v", changed, err)
	}
	if err := database.QueryRowContext(ctx, `SELECT count(*) FROM management_sessions
WHERE principal_id=$1 AND status='active'`, principalID).Scan(&count); err != nil || count != 2 {
		t.Fatalf("active sessions after changed evidence = %d, %v", count, err)
	}

	distinctIssuerSessionID := "distinct-dashboard-session-" + uuid.NewString()
	distinctIdentity := changedIdentity
	distinctIdentity.IssuerSessionID = &distinctIssuerSessionID
	distinctDraft := changedDraft
	distinctDraft.ID = uuid.NewString()
	distinctDraft.TokenID = "distinct-session-token-" + uuid.NewString()
	distinctDraft.IssuerSessionID = &distinctIssuerSessionID
	if _, err := coordinator.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
		Identity: distinctIdentity,
		Session:  distinctDraft,
	}, func(_ context.Context, session managementauth.LiveSession, _ time.Time) (managementauth.IssuedToken, error) {
		return managementauth.IssuedToken{
			AccessToken: session.TokenID, TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
			ManagementSessionID: session.ID,
		}, nil
	}); !errors.Is(err, managementauth.ErrSessionLimitExceeded) {
		t.Fatalf("distinct issuer session above active limit error = %v", err)
	}
}

func stringPointer(value string) *string { return &value }
