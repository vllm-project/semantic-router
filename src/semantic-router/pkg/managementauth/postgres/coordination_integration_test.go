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

func TestPostgresChallengeIsOneTimeAndGloballyRateLimited(t *testing.T) {
	database, ctx := authCoordinationDatabase(t)
	issuerID := seedCoordinationIssuer(t, ctx, database)
	ids := []string{uuid.NewString(), uuid.NewString()}
	nextID := 0
	store, err := NewChallengeStore(ChallengeOptions{
		Database: database, TTL: 30 * time.Second, RateLimit: 1,
		NewID: func() (string, error) {
			id := ids[nextID]
			nextID++
			return id, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Truncate(time.Second)
	challenge, err := store.Create(ctx, issuerID, "test-rate-identity", now)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Create(ctx, issuerID, "test-rate-identity", now); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("rate-limited Create() = %v", err)
	}
	if err := store.Consume(ctx, challenge.ID, issuerID, challenge.Nonce, now.Add(time.Second)); err != nil {
		t.Fatal(err)
	}
	if err := store.Consume(ctx, challenge.ID, issuerID, challenge.Nonce, now.Add(2*time.Second)); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("replayed Consume() = %v", err)
	}
}

func TestPostgresBarrierChecksDurableLifecycleAndExplicitFences(t *testing.T) {
	database, ctx := authCoordinationDatabase(t)
	issuerID := seedCoordinationIssuer(t, ctx, database)
	principalID, sessionID := uuid.NewString(), uuid.NewString()
	issuer := "https://issuer.example/" + issuerID
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,$2,$3,'Coordination User','active')`, principalID, issuer, "subject-"+principalID); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_session_policy
  (singleton,access_token_ttl_seconds,session_ttl_seconds,max_active_sessions,
   action_requirements,seed_version,revision)
VALUES (TRUE,900,28800,5,'{}'::jsonb,1,1)
ON CONFLICT (singleton) DO NOTHING`); err != nil {
		t.Fatal(err)
	}
	sessions, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	now := time.Now().UTC().Truncate(time.Second)
	_, err = sessions.Create(ctx, managementauth.SessionDraft{
		ID: sessionID, PrincipalID: principalID, TokenID: "token-" + sessionID,
		Audience: "vllm-sr-management", AuthSourceKind: managementauth.AuthSourceIssuer,
		AuthSourceID: issuerID, EvidenceKind: managementauth.EvidenceHuman,
		Human: &managementauth.HumanEvidence{
			AuthenticationTime: now.Add(-time.Minute).Unix(), AAL: "aal2", AMR: []string{"pwd"},
		},
		AuthenticatedAt: now.Add(-time.Minute), EvidenceExpiresAt: now.Add(time.Hour),
	})
	if err != nil {
		t.Fatal(err)
	}
	barriers, err := NewBarrierStore(database)
	if err != nil {
		t.Fatal(err)
	}
	check := managementauth.BarrierCheck{
		SessionID: sessionID, PrincipalID: principalID,
		AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: issuerID,
	}
	state, err := barriers.Check(ctx, check)
	if err != nil || !state.Allows() {
		t.Fatalf("initial Check() = %+v, %v", state, err)
	}
	if installErr := barriers.InstallDeny(ctx, managementauth.BarrierManagementPrincipal, principalID); installErr != nil {
		t.Fatal(installErr)
	}
	state, err = barriers.Check(ctx, check)
	if err != nil || !state.PrincipalDenied || state.Allows() {
		t.Fatalf("explicit principal barrier Check() = %+v, %v", state, err)
	}
	if removeErr := barriers.RemoveDeny(ctx, managementauth.BarrierManagementPrincipal, principalID); removeErr != nil {
		t.Fatal(removeErr)
	}
	if _, revokeErr := sessions.Revoke(ctx, sessionID, "token-"+sessionID); revokeErr != nil {
		t.Fatal(revokeErr)
	}
	state, err = barriers.Check(ctx, check)
	if err != nil || !state.SessionDenied || state.Allows() {
		t.Fatalf("durably revoked session Check() = %+v, %v", state, err)
	}
}

func authCoordinationDatabase(t *testing.T) (*sql.DB, context.Context) {
	t.Helper()
	dsn := os.Getenv("MANAGEMENTAUTH_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("MANAGEMENTAUTH_TEST_POSTGRES_DSN is not configured")
	}
	database, err := sql.Open("postgres", dsn)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	if err := (controlplanepostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatal(err)
	}
	return database, ctx
}

func seedCoordinationIssuer(t *testing.T, ctx context.Context, database *sql.DB) string {
	t.Helper()
	issuerID := uuid.NewString()
	issuer := "https://issuer.example/" + issuerID
	if _, err := database.ExecContext(ctx, `INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,audiences,claim_mapping,assurance_mapping,status)
VALUES ($1,$2,'oidc',$3,'["vllm-sr-management"]'::jsonb,'{}'::jsonb,'{}'::jsonb,'active')`,
		issuerID, issuer, issuer+"/.well-known/openid-configuration"); err != nil {
		t.Fatal(err)
	}
	return issuerID
}
