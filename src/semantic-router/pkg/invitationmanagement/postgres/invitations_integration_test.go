package postgres_test

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"net/netip"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	controlpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	invitationpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	authpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	testNamespaceID  = "11111111-1111-4111-8111-111111111111"
	testActorID      = "22222222-2222-4222-8222-222222222222"
	testTeamID       = "33333333-3333-4333-8333-333333333333"
	testAccessID     = "44444444-4444-4444-8444-444444444444"
	testRateID       = "55555555-5555-4555-8555-555555555555"
	testActorBinding = "66666666-6666-4666-8666-666666666666"
	testIssuerID     = "77777777-7777-4777-8777-777777777777"
	testConsumerRole = "10000000-0000-5000-8000-000000000008"
	testPlatformRole = "10000000-0000-5000-8000-000000000002"
)

func TestInvitationPostgresAtomicOnboardingAndBoundedReplay(t *testing.T) {
	dsn := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("PostgreSQL invitation test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	db := isolatedInvitationDatabase(t, ctx, dsn)
	if err := (controlpostgres.Migrator{DB: db}).Apply(ctx); err != nil {
		t.Fatal(err)
	}
	seedInvitationAuthority(t, ctx, db)
	service, exchanges, responseKEK := newInvitationService(t, db,
		invitationPepper("invite-v1", "i"), responseKeyring("response-v1", "r"))
	actor := invitationmanagement.Actor{
		PrincipalID: testActorID, ActorChain: []string{testActorID},
		RequestID: "create-invitation-1", SourceIP: netip.MustParseAddr("192.0.2.20"), Reason: "Create invitation.",
	}

	create := func(email, subject, idempotencyKey string) invitationmanagement.SecretResult {
		t.Helper()
		result, err := service.Create(ctx, invitationmanagement.CreateRequest{
			NamespaceID: testNamespaceID,
			Expected:    invitationmanagement.ExpectedIdentity{Issuer: "https://issuer.example", Subject: subject, Email: email},
			DisplayName: "Invited User",
			RoleGrants:  []invitationmanagement.RequestedRoleGrant{{RoleID: testConsumerRole, ScopeKind: "user"}},
			Team:        &invitationmanagement.TeamAssignment{TeamID: testTeamID, Role: accesscontrol.TeamRoleMember},
			ExpiresAt:   time.Now().UTC().Add(time.Hour), IdempotencyKey: idempotencyKey, Actor: actor,
		})
		if err != nil {
			t.Fatalf("Create(%q) error = %v", email, err)
		}
		return result
	}

	issued := create("invited@example.com", "subject-1", "invite-create-00000001")
	replayedCreate, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr := service.Create(ctx, invitationmanagement.CreateRequest{
		NamespaceID: testNamespaceID,
		Expected:    invitationmanagement.ExpectedIdentity{Issuer: "https://issuer.example", Subject: "subject-1", Email: "invited@example.com"},
		DisplayName: "Invited User",
		RoleGrants:  []invitationmanagement.RequestedRoleGrant{{RoleID: testConsumerRole, ScopeKind: "user"}},
		Team:        &invitationmanagement.TeamAssignment{TeamID: testTeamID, Role: accesscontrol.TeamRoleMember},
		ExpiresAt:   issued.Invitation.ExpiresAt, IdempotencyKey: "invite-create-00000001", Actor: actor,
	})
	if testInvitationPostgresAtomicOnboardingAndBoundedReplayErr != nil || !replayedCreate.Replayed || replayedCreate.Token != issued.Token ||
		!bytes.Equal(replayedCreate.CanonicalJSON, issued.CanonicalJSON) {
		t.Fatalf("create replay = %#v, %v", replayedCreate, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}

	accept := func(token, subject, email, requestID string) (managementauth.IdentityExchangeResult, error) {
		authenticatedAt := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
		return exchanges.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
			Identity: managementauth.VerifiedExternalIdentity{
				IssuerID: testIssuerID, Issuer: "https://issuer.example", Subject: subject,
				VerifiedEmail: email, DisplayName: "Invited User", Nonce: requestID,
				AAL: "aal2", AMR: []string{"pwd"}, AuthenticatedAt: authenticatedAt,
				EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
			},
			InvitationToken: token, RequestID: requestID,
			Session: managementauth.SessionDraft{
				ID: uuid.NewString(), TokenID: uuid.NewString(), Audience: "vllm-sr-management",
				AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: testIssuerID,
				EvidenceKind:    managementauth.EvidenceHuman,
				Human:           &managementauth.HumanEvidence{AuthenticationTime: authenticatedAt.Unix(), AAL: "aal2", AMR: []string{"pwd"}},
				AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
			},
		}, testSessionIssuer)
	}
	accepted, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr := accept(issued.Token, "subject-1", "invited@example.com", "accept-invitation-1")
	if testInvitationPostgresAtomicOnboardingAndBoundedReplayErr != nil || accepted.Replayed || accepted.Onboarding == nil || accepted.Onboarding.APIKey == "" ||
		accepted.Onboarding.APIKeyID == "" || accepted.Onboarding.UserID == "" ||
		accepted.Onboarding.TeamID != testTeamID || accepted.Issued.ManagementSessionID == "" {
		t.Fatalf("Accept() = %#v, %v", accepted, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}
	replayed, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr := accept(issued.Token, "subject-1", "invited@example.com", "accept-invitation-2")
	if testInvitationPostgresAtomicOnboardingAndBoundedReplayErr != nil || !replayed.Replayed || replayed.Onboarding == nil ||
		replayed.Onboarding.APIKey != accepted.Onboarding.APIKey ||
		replayed.Onboarding.UserID != accepted.Onboarding.UserID ||
		replayed.Issued.ManagementSessionID != accepted.Issued.ManagementSessionID {
		t.Fatalf("accepted replay = %#v, %v", replayed, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}
	authenticatedAt := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	standard, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr := exchanges.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
		Identity: managementauth.VerifiedExternalIdentity{
			IssuerID: testIssuerID, Issuer: "https://issuer.example", Subject: "subject-1",
			AAL: "aal2", AMR: []string{"pwd"},
			AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
		},
		RequestID: "standard-exchange",
		Session: managementauth.SessionDraft{
			ID: uuid.NewString(), TokenID: uuid.NewString(), Audience: "vllm-sr-management",
			AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: testIssuerID,
			EvidenceKind:    managementauth.EvidenceHuman,
			Human:           &managementauth.HumanEvidence{AuthenticationTime: authenticatedAt.Unix(), AAL: "aal2", AMR: []string{"pwd"}},
			AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: authenticatedAt.Add(2 * time.Hour),
		},
	}, testSessionIssuer)
	if testInvitationPostgresAtomicOnboardingAndBoundedReplayErr != nil || standard.Onboarding != nil || standard.Replayed ||
		standard.Issued.ManagementSessionID == "" || standard.Issued.ManagementSessionID == accepted.Issued.ManagementSessionID {
		t.Fatalf("standard exchange = %#v, %v", standard, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}

	var principalLinks, memberships, accessBindings, rateBindings, keys, credentials int
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_principal_user_links WHERE namespace_id=$1 AND user_id=$2),
  (SELECT count(*) FROM access_team_memberships WHERE namespace_id=$1 AND team_id=$3 AND user_id=$2 AND status='active'),
  (SELECT count(*) FROM access_policy_bindings WHERE namespace_id=$1 AND subject_id=$2 AND policy_id=$4 AND status='active'),
  (SELECT count(*) FROM rate_limit_bindings WHERE namespace_id=$1 AND subject_id=$2 AND policy_id=$5 AND status='active'),
  (SELECT count(*) FROM access_api_keys WHERE namespace_id=$1 AND owner_user_id=$2 AND status='active'),
  (SELECT count(*) FROM access_api_key_credentials c JOIN access_api_keys k ON k.id=c.api_key_id WHERE k.namespace_id=$1 AND k.owner_user_id=$2 AND c.status='active')`,
		testNamespaceID, accepted.Onboarding.UserID, testTeamID, testAccessID, testRateID).Scan(
		&principalLinks, &memberships, &accessBindings, &rateBindings, &keys, &credentials); err != nil {
		t.Fatal(err)
	}
	if principalLinks != 1 || memberships != 1 || accessBindings != 1 || rateBindings != 1 || keys != 1 || credentials != 1 {
		t.Fatalf("materialized counts = link:%d membership:%d access:%d rate:%d key:%d credential:%d",
			principalLinks, memberships, accessBindings, rateBindings, keys, credentials)
	}
	assertSecretsAbsentFromAccounting(t, ctx, db, issued.Token, accepted.Onboarding.APIKey)

	concurrent := create("double@example.com", "subject-2", "invite-create-00000002")
	start := make(chan struct{})
	type acceptanceOutcome struct {
		value managementauth.IdentityExchangeResult
		err   error
	}
	outcomes := make(chan acceptanceOutcome, 2)
	var ready sync.WaitGroup
	ready.Add(2)
	for index := 0; index < 2; index++ {
		go func(index int) {
			ready.Done()
			<-start
			value, acceptErr := accept(concurrent.Token, "subject-2", "double@example.com",
				"concurrent-accept-"+strconv.Itoa(index))
			outcomes <- acceptanceOutcome{value: value, err: acceptErr}
		}(index)
	}
	ready.Wait()
	close(start)
	first, second := <-outcomes, <-outcomes
	if first.err != nil || second.err != nil || first.value.Onboarding == nil || second.value.Onboarding == nil ||
		first.value.Onboarding.UserID != second.value.Onboarding.UserID ||
		first.value.Onboarding.APIKey != second.value.Onboarding.APIKey ||
		first.value.Issued.ManagementSessionID != second.value.Issued.ManagementSessionID ||
		first.value.Replayed == second.value.Replayed {
		t.Fatalf("concurrent acceptance = %#v / %#v", first, second)
	}

	rollback := create("rollback@example.com", "subject-rollback", "invite-create-rollback")
	staleAuthentication := time.Now().UTC().Add(-3 * time.Hour).Truncate(time.Second)
	rollbackSessionID := uuid.NewString()
	_, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr = exchanges.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
		Identity: managementauth.VerifiedExternalIdentity{
			IssuerID: testIssuerID, Issuer: "https://issuer.example", Subject: "subject-rollback",
			VerifiedEmail: "rollback@example.com", DisplayName: "Rollback User", Nonce: "rollback-exchange",
			AAL: "aal2", AMR: []string{"pwd"}, AuthenticatedAt: staleAuthentication,
			EvidenceExpiresAt: staleAuthentication.Add(time.Hour),
		},
		InvitationToken: rollback.Token, RequestID: "rollback-exchange",
		Session: managementauth.SessionDraft{
			ID: rollbackSessionID, TokenID: uuid.NewString(), Audience: "vllm-sr-management",
			AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: testIssuerID,
			EvidenceKind:    managementauth.EvidenceHuman,
			Human:           &managementauth.HumanEvidence{AuthenticationTime: staleAuthentication.Unix(), AAL: "aal2", AMR: []string{"pwd"}},
			AuthenticatedAt: staleAuthentication, EvidenceExpiresAt: staleAuthentication.Add(time.Hour),
		},
	}, testSessionIssuer)
	if !errors.Is(testInvitationPostgresAtomicOnboardingAndBoundedReplayErr, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("rollback exchange error = %v", testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}
	var rollbackStatus string
	var rollbackWrites int
	if err := db.QueryRowContext(ctx, `SELECT status FROM management_invitations WHERE id=$1`, rollback.Invitation.ID).Scan(&rollbackStatus); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_principals WHERE issuer='https://issuer.example' AND subject='subject-rollback')+
  (SELECT count(*) FROM management_sessions WHERE id=$1)`, rollbackSessionID).Scan(&rollbackWrites); err != nil {
		t.Fatal(err)
	}
	if rollbackStatus != "pending" || rollbackWrites != 0 {
		t.Fatalf("failed exchange committed status=%q writes=%d", rollbackStatus, rollbackWrites)
	}

	issuerFailure := create("issuer-failure@example.com", "subject-issuer-failure", "invite-create-issuer-failure")
	issuerFailureAt := time.Now().UTC().Add(-time.Minute).Truncate(time.Second)
	issuerFailureSessionID := uuid.NewString()
	_, testInvitationPostgresAtomicOnboardingAndBoundedReplayErr = exchanges.ExchangeIdentity(ctx, managementauth.IdentityExchangeRequest{
		Identity: managementauth.VerifiedExternalIdentity{
			IssuerID: testIssuerID, Issuer: "https://issuer.example", Subject: "subject-issuer-failure",
			VerifiedEmail: "issuer-failure@example.com", DisplayName: "Issuer Failure",
			Nonce: "issuer-failure-exchange", AAL: "aal2", AMR: []string{"pwd"},
			AuthenticatedAt: issuerFailureAt, EvidenceExpiresAt: issuerFailureAt.Add(2 * time.Hour),
		},
		InvitationToken: issuerFailure.Token, RequestID: "issuer-failure-exchange",
		Session: managementauth.SessionDraft{
			ID: issuerFailureSessionID, TokenID: uuid.NewString(), Audience: "vllm-sr-management",
			AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: testIssuerID,
			EvidenceKind:    managementauth.EvidenceHuman,
			Human:           &managementauth.HumanEvidence{AuthenticationTime: issuerFailureAt.Unix(), AAL: "aal2", AMR: []string{"pwd"}},
			AuthenticatedAt: issuerFailureAt, EvidenceExpiresAt: issuerFailureAt.Add(2 * time.Hour),
		},
	}, func(context.Context, managementauth.LiveSession, time.Time) (managementauth.IssuedToken, error) {
		return managementauth.IssuedToken{}, managementauth.ErrAuthenticationUnavailable
	})
	if !errors.Is(testInvitationPostgresAtomicOnboardingAndBoundedReplayErr, managementauth.ErrAuthenticationUnavailable) {
		t.Fatalf("issuer failure exchange error = %v", testInvitationPostgresAtomicOnboardingAndBoundedReplayErr)
	}
	var issuerFailureStatus string
	var issuerFailureWrites int
	if err := db.QueryRowContext(ctx, `SELECT status FROM management_invitations WHERE id=$1`,
		issuerFailure.Invitation.ID).Scan(&issuerFailureStatus); err != nil {
		t.Fatal(err)
	}
	if err := db.QueryRowContext(ctx, `SELECT
  (SELECT count(*) FROM management_principals WHERE issuer='https://issuer.example' AND subject='subject-issuer-failure')+
  (SELECT count(*) FROM management_sessions WHERE id=$1)`, issuerFailureSessionID).Scan(&issuerFailureWrites); err != nil {
		t.Fatal(err)
	}
	if issuerFailureStatus != "pending" || issuerFailureWrites != 0 {
		t.Fatalf("issuer failure committed status=%q writes=%d", issuerFailureStatus, issuerFailureWrites)
	}

	expired := create("expired@example.com", "subject-3", "invite-create-00000003")
	if _, err := db.ExecContext(ctx, `UPDATE management_invitations SET expires_at=clock_timestamp()-interval '1 second'
WHERE id=$1`, expired.Invitation.ID); err != nil {
		t.Fatal(err)
	}
	if _, err := accept(expired.Token, "subject-3", "expired@example.com", "expired-accept"); !errors.Is(err, managementauth.ErrInvitationExpired) {
		t.Fatalf("expired Accept() error = %v", err)
	}

	if _, err := db.ExecContext(ctx, `UPDATE management_invitations
SET acceptance_result_expires_at=acceptance_result_delivered_at+interval '1 microsecond'
WHERE id=$1`, issued.Invitation.ID); err != nil {
		t.Fatal(err)
	}
	time.Sleep(2 * time.Millisecond)
	if _, err := accept(issued.Token, "subject-1", "invited@example.com", "expired-delivery"); !errors.Is(err, managementauth.ErrInvitationResultExpired) {
		t.Fatalf("expired delivery Accept() error = %v", err)
	}
	var ciphertext []byte
	var erasedAt sql.NullTime
	if err := db.QueryRowContext(ctx, `SELECT acceptance_response_ciphertext,acceptance_result_erased_at
FROM management_invitations WHERE id=$1`, issued.Invitation.ID).Scan(&ciphertext, &erasedAt); err != nil ||
		ciphertext != nil || !erasedAt.Valid {
		t.Fatalf("erased result = ciphertext %d erased %v error %v", len(ciphertext), erasedAt.Valid, err)
	}

	missingInvitationPepper, _, _ := newInvitationService(t, db,
		invitationPepper("invite-v2", "j"), responseKEK)
	if err := missingInvitationPepper.Ready(ctx); err == nil {
		t.Fatal("Ready() accepted a missing referenced invitation pepper")
	}
	missingResponseKEK, _, _ := newInvitationService(t, db,
		invitationPepper("invite-v1", "i"), responseKeyring("response-v2", "s"))
	if err := missingResponseKEK.Ready(ctx); err == nil {
		t.Fatal("Ready() accepted a missing referenced response KEK")
	}
	if err := service.Ready(ctx); err != nil {
		t.Fatalf("Ready() with retained key versions = %v", err)
	}
}

func seedInvitationAuthority(t *testing.T, ctx context.Context, db *sql.DB) {
	t.Helper()
	statements := []struct {
		query string
		args  []any
	}{
		{`INSERT INTO access_namespaces (id,name,quota_partition_id,billing_currency,status)
VALUES ($1,'invitation-test','invitation-partition','USD','active')`, []any{testNamespaceID}},
		{`INSERT INTO management_principals (id,issuer,subject,display_name,verified_email,status)
VALUES ($1,'https://issuer.example','actor','Actor','actor@example.com','active')`, []any{testActorID}},
		{`INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,audiences,claim_mapping,assurance_mapping,status)
VALUES ($1,'https://issuer.example','oidc','https://issuer.example/.well-known/openid-configuration',
  '["vllm-sr-management"]','{}','{}','active')`, []any{testIssuerID}},
		{
			`INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,delegation_ceiling,status,revision)
SELECT $1,$2,$3,'namespace',$4,permissions,'active',1 FROM management_roles WHERE id=$3`,
			[]any{testActorBinding, testActorID, testPlatformRole, testNamespaceID},
		},
		{`INSERT INTO access_subjects (namespace_id,id,kind) VALUES ($1,$2,'team')`, []any{testNamespaceID, testTeamID}},
		{`INSERT INTO access_teams (id,namespace_id,name,description,status)
VALUES ($1,$2,'Invitation Team','Test Team','active')`, []any{testTeamID, testNamespaceID}},
		{`INSERT INTO access_policies (id,namespace_id,name,status) VALUES ($1,$2,'default-access','active')`, []any{testAccessID, testNamespaceID}},
		{`INSERT INTO rate_limit_policies (id,namespace_id,name,status) VALUES ($1,$2,'default-rate','active')`, []any{testRateID, testNamespaceID}},
		{`INSERT INTO self_service_policies
  (namespace_id,automatic_first_key,default_access_policy_id,default_rate_limit_policy_id,seed_version)
VALUES ($1,TRUE,$2,$3,1)`, []any{testNamespaceID, testAccessID, testRateID}},
	}
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, statement := range statements {
		if _, err := tx.ExecContext(ctx, statement.query, statement.args...); err != nil {
			_ = tx.Rollback()
			t.Fatal(err)
		}
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
}

func newInvitationService(t *testing.T, db *sql.DB, invitationPeppers accesscredential.PepperKeyring,
	responseKEK accesscredential.KEKKeyring) (*invitationmanagement.Service,
	*invitationmanagement.IdentityExchangeCoordinator, accesscredential.KEKKeyring,
) {
	t.Helper()
	invitationStore, err := invitationpostgres.New(db)
	if err != nil {
		t.Fatal(err)
	}
	sessionStore, err := authpostgres.New(db)
	if err != nil {
		t.Fatal(err)
	}
	store, err := invitationpostgres.NewAtomicExchangeStore(invitationStore, sessionStore)
	if err != nil {
		t.Fatal(err)
	}
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "command-v1",
		Keys:          map[string][]byte{"command-v1": []byte(strings.Repeat("c", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	firstKeys, err := invitationmanagement.NewAPIKeyFirstKeyPreparer(
		accesscredential.PepperKeyring{ActiveVersion: "api-key-v1", Keys: map[string][]byte{
			"api-key-v1": []byte(strings.Repeat("k", 32)),
		}}, nil)
	if err != nil {
		t.Fatal(err)
	}
	service, err := invitationmanagement.NewService(invitationmanagement.Options{
		Repository: store, Commands: commands, InvitationPeppers: invitationPeppers,
		CursorKeyring: securitykeyring.Symmetric{
			ActiveVersion: "cursor-v1",
			Keys: map[string][]byte{
				"cursor-v1": []byte(strings.Repeat("u", 32)),
			},
		},
		ResponseKEK: responseKEK, FirstKeys: firstKeys,
		IdempotencyTTL: time.Hour, SecretDeliveryTTL: time.Hour,
	})
	if err != nil {
		t.Fatal(err)
	}
	exchanges, err := invitationmanagement.NewIdentityExchangeCoordinator(service)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(service.Close)
	return service, exchanges, responseKEK
}

func testSessionIssuer(_ context.Context, session managementauth.LiveSession,
	now time.Time,
) (managementauth.IssuedToken, error) {
	if err := session.ValidateAt(now); err != nil {
		return managementauth.IssuedToken{}, err
	}
	return managementauth.IssuedToken{
		AccessToken: "test-management-token-" + session.ID, TokenType: "Bearer",
		ExpiresIn: time.Minute, ManagementSessionID: session.ID,
	}, nil
}

func invitationPepper(version, fill string) accesscredential.PepperKeyring {
	return accesscredential.PepperKeyring{
		ActiveVersion: version,
		Keys:          map[string][]byte{version: []byte(strings.Repeat(fill, 32))},
	}
}

func responseKeyring(version, fill string) accesscredential.KEKKeyring {
	return accesscredential.KEKKeyring{
		ActiveVersion: version,
		Keys:          map[string][]byte{version: []byte(strings.Repeat(fill, 32))},
	}
}

func assertSecretsAbsentFromAccounting(t *testing.T, ctx context.Context, db *sql.DB, secrets ...string) {
	t.Helper()
	var accounting string
	if err := db.QueryRowContext(ctx, `SELECT
  COALESCE((SELECT string_agg(details::text,'') FROM access_audit_events),'') ||
  COALESCE((SELECT string_agg(payload::text,'') FROM policy_outbox),'')`).Scan(&accounting); err != nil {
		t.Fatal(err)
	}
	for _, secret := range secrets {
		if secret != "" && strings.Contains(accounting, secret) {
			t.Fatal("audit or outbox exposed invitation or first-key plaintext")
		}
	}
}

func isolatedInvitationDatabase(t *testing.T, ctx context.Context, dsn string) *sql.DB {
	t.Helper()
	admin, isolatedInvitationDatabaseErr := sql.Open("postgres", dsn)
	if isolatedInvitationDatabaseErr != nil {
		t.Fatal(isolatedInvitationDatabaseErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	if err := admin.PingContext(ctx); err != nil {
		t.Fatal(err)
	}
	schema := "vsr_invitation_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, isolatedInvitationDatabaseErr := url.Parse(dsn)
	if isolatedInvitationDatabaseErr != nil {
		t.Fatal(isolatedInvitationDatabaseErr)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	db, isolatedInvitationDatabaseErr := sql.Open("postgres", parsed.String())
	if isolatedInvitationDatabaseErr != nil {
		t.Fatal(isolatedInvitationDatabaseErr)
	}
	t.Cleanup(func() { _ = db.Close() })
	return db
}
