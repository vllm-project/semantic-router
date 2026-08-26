package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	authpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestBackchannelLogoutTombstoneRacesIssuerExchange(t *testing.T) {
	for _, testCase := range []struct {
		name string
		sid  bool
	}{
		{name: "issuer SID", sid: true},
		{name: "issuer subject", sid: false},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			runBackchannelLogoutRace(t, testCase.sid)
		})
	}
}

type logoutRaceFixture struct {
	database             *sql.DB
	ctx                  context.Context
	identityStore        *Store
	exchanges            *authpostgres.IdentityExchangeCoordinator
	issuerID             string
	issuer               string
	principalID          string
	subject              string
	issuerSessionPointer *string
	logoutIssuedAt       time.Time
	logoutRequest        managementidentity.BackchannelLogout
	staleExchange        managementauth.IdentityExchangeRequest
}

type logoutOutcome struct {
	result managementidentity.BackchannelLogoutResult
	err    error
}

type exchangeOutcome struct {
	result managementauth.IdentityExchangeResult
	err    error
}

func newLogoutRaceFixture(t *testing.T, sid bool) logoutRaceFixture {
	t.Helper()
	database := bootstrapTestDatabase(t)
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	t.Cleanup(cancel)
	issuerID := uuid.NewString()
	issuer := "https://issuer.example/" + issuerID
	principalID := uuid.NewString()
	subject := "logout-race-subject-" + principalID
	if _, err := database.ExecContext(ctx, `INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,audiences,claim_mapping,assurance_mapping,status)
VALUES ($1,$2,'oidc',$3,'["vllm-sr-management"]'::jsonb,'{}'::jsonb,'{}'::jsonb,'active')`,
		issuerID, issuer, issuer+"/.well-known/openid-configuration"); err != nil {
		t.Fatal(err)
	}
	if _, err := database.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,status)
VALUES ($1,$2,$3,'Logout race principal','active')`, principalID, issuer, subject); err != nil {
		t.Fatal(err)
	}
	commands, err := managementcommand.NewCodec(securitykeyring.Symmetric{
		ActiveVersion: "v1",
		Keys:          map[string][]byte{"v1": []byte(strings.Repeat("l", 32))},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = commands.Close() })
	identityStore, err := New(database, commands)
	if err != nil {
		t.Fatal(err)
	}
	sessions, err := authpostgres.New(database)
	if err != nil {
		t.Fatal(err)
	}
	exchanges, err := authpostgres.NewIdentityExchangeCoordinator(database, sessions)
	if err != nil {
		t.Fatal(err)
	}
	issuerSessionID := "logout-race-sid-" + uuid.NewString()
	logoutIssuedAt := time.Now().UTC().Add(-2 * time.Minute).Truncate(time.Second)
	logoutIdentity := managementauth.BackchannelLogoutIdentity{
		IssuerID: issuerID, TokenID: "logout-token-" + uuid.NewString(), Subject: subject,
		IssuedAt: logoutIssuedAt, ExpiresAt: time.Now().UTC().Add(5 * time.Minute),
		ClaimsDigest: sha256.Sum256([]byte("logout-claims-" + uuid.NewString())),
	}
	var issuerSessionPointer *string
	if sid {
		logoutIdentity.IssuerSessionID = issuerSessionID
		issuerSessionPointer = &issuerSessionID
	}
	logoutRequest := managementidentity.BackchannelLogout{
		Identity: logoutIdentity, RequestID: "logout-race-" + uuid.NewString(),
	}
	return logoutRaceFixture{
		database: database, ctx: ctx, identityStore: identityStore, exchanges: exchanges,
		issuerID: issuerID, issuer: issuer, principalID: principalID, subject: subject,
		issuerSessionPointer: issuerSessionPointer, logoutIssuedAt: logoutIssuedAt,
		logoutRequest: logoutRequest,
		staleExchange: issuerLogoutExchangeRequest(
			issuerID, issuer, subject, issuerSessionPointer, logoutIssuedAt.Add(-time.Minute),
		),
	}
}

func raceIssuerLogoutAndExchange(
	fixture logoutRaceFixture,
	exchange managementauth.IdentityExchangeRequest,
) (logoutOutcome, exchangeOutcome) {
	ready := make(chan struct{}, 2)
	start := make(chan struct{})
	logoutResults := make(chan logoutOutcome, 1)
	exchangeResults := make(chan exchangeOutcome, 1)
	go func() {
		ready <- struct{}{}
		<-start
		result, err := fixture.identityStore.ApplyBackchannelLogout(fixture.ctx, fixture.logoutRequest)
		logoutResults <- logoutOutcome{result: result, err: err}
	}()
	go func() {
		ready <- struct{}{}
		<-start
		result, err := fixture.exchanges.ExchangeIdentity(fixture.ctx, exchange, issuerLogoutTestIssuer)
		exchangeResults <- exchangeOutcome{result: result, err: err}
	}()
	<-ready
	<-ready
	close(start)
	return <-logoutResults, <-exchangeResults
}

func runBackchannelLogoutRace(t *testing.T, sid bool) {
	t.Helper()
	fixture := newLogoutRaceFixture(t, sid)
	logoutResult, exchangeResult := raceIssuerLogoutAndExchange(fixture, fixture.staleExchange)
	if logoutResult.err != nil {
		t.Fatalf("concurrent ApplyBackchannelLogout() error = %v", logoutResult.err)
	}
	if exchangeResult.err != nil && !errors.Is(exchangeResult.err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("concurrent ExchangeIdentity() error = %v", exchangeResult.err)
	}
	assertActiveLogoutRaceSessions(t, fixture, 0, "active sessions after concurrent logout")
	replayed, err := fixture.identityStore.ApplyBackchannelLogout(fixture.ctx, fixture.logoutRequest)
	if err != nil || !replayed.Replayed {
		t.Fatalf("ApplyBackchannelLogout() replay = %+v, %v", replayed, err)
	}
	staleExchange := fixture.staleExchange
	staleExchange.Session.ID = uuid.NewString()
	staleExchange.Session.TokenID = "stale-token-" + uuid.NewString()
	if _, err := fixture.exchanges.ExchangeIdentity(
		fixture.ctx, staleExchange, issuerLogoutTestIssuer,
	); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
		t.Fatalf("stale ExchangeIdentity() error = %v", err)
	}
	assertLogoutRaceTombstone(t, fixture, sid)
	freshExchange := issuerLogoutExchangeRequest(
		fixture.issuerID, fixture.issuer, fixture.subject, fixture.issuerSessionPointer,
		fixture.logoutIssuedAt.Add(time.Minute),
	)
	if sid {
		_, err = fixture.exchanges.ExchangeIdentity(fixture.ctx, freshExchange, issuerLogoutTestIssuer)
		if !errors.Is(err, managementauth.ErrAuthenticationDenied) {
			t.Fatalf("same-SID fresh ExchangeIdentity() error = %v", err)
		}
		return
	}
	freshLogout, freshResult := raceIssuerLogoutAndExchange(fixture, freshExchange)
	if freshLogout.err != nil || !freshLogout.result.Replayed {
		t.Fatalf("concurrent subject logout replay = %+v", freshLogout)
	}
	if freshResult.err != nil || freshResult.result.Issued.ManagementSessionID == "" {
		t.Fatalf("concurrent subject reauthentication = %+v", freshResult)
	}
	assertActiveLogoutRaceSessions(t, fixture, 1, "fresh subject sessions after concurrent logout")
}

func assertActiveLogoutRaceSessions(t *testing.T, fixture logoutRaceFixture, expected int, label string) {
	t.Helper()
	var activeSessions int
	err := fixture.database.QueryRowContext(fixture.ctx, `SELECT count(*) FROM management_sessions
WHERE principal_id=$1 AND status='active'`, fixture.principalID).Scan(&activeSessions)
	if err != nil || activeSessions != expected {
		t.Fatalf("%s = %d, %v", label, activeSessions, err)
	}
}

func assertLogoutRaceTombstone(t *testing.T, fixture logoutRaceFixture, sid bool) {
	t.Helper()
	selectorKind := "subject"
	if sid {
		selectorKind = "sid"
	}
	var selectorCount int
	err := fixture.database.QueryRowContext(fixture.ctx, `SELECT count(*)
FROM management_issuer_logout_tombstones WHERE issuer_id=$1 AND selector_kind=$2`,
		fixture.issuerID, selectorKind,
	).Scan(&selectorCount)
	if err != nil || selectorCount != 1 {
		t.Fatalf("logout tombstones = %d, %v", selectorCount, err)
	}
}

func issuerLogoutExchangeRequest(
	issuerID, issuer, subject string,
	issuerSessionID *string,
	authenticatedAt time.Time,
) managementauth.IdentityExchangeRequest {
	evidenceExpiresAt := time.Now().UTC().Add(time.Hour)
	return managementauth.IdentityExchangeRequest{
		Identity: managementauth.VerifiedExternalIdentity{
			IssuerID: issuerID, Issuer: issuer, Subject: subject, IssuerSessionID: issuerSessionID,
			AAL: "aal2", AMR: []string{"pwd", "otp"},
			AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: evidenceExpiresAt,
		},
		Session: managementauth.SessionDraft{
			ID: uuid.NewString(), TokenID: "logout-race-token-" + uuid.NewString(),
			IssuerSessionID: issuerSessionID, Audience: "vllm-sr-management",
			AuthSourceKind: managementauth.AuthSourceIssuer, AuthSourceID: issuerID,
			EvidenceKind: managementauth.EvidenceHuman,
			Human: &managementauth.HumanEvidence{
				AuthenticationTime: authenticatedAt.Unix(), AAL: "aal2", AMR: []string{"pwd", "otp"},
			},
			AuthenticatedAt: authenticatedAt, EvidenceExpiresAt: evidenceExpiresAt,
		},
	}
}

func issuerLogoutTestIssuer(
	_ context.Context,
	session managementauth.LiveSession,
	_ time.Time,
) (managementauth.IssuedToken, error) {
	return managementauth.IssuedToken{
		AccessToken: session.TokenID, TokenType: "Bearer", ExpiresIn: 15 * time.Minute,
		ManagementSessionID: session.ID,
	}, nil
}
