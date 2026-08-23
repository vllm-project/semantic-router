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

func stringPointer(value string) *string { return &value }
