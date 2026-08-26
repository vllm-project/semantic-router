package postgres

import (
	"context"
	"crypto/sha256"
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
			database := bootstrapTestDatabase(t)
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
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
			staleAuthenticatedAt := logoutIssuedAt.Add(-time.Minute)
			logoutIdentity := managementauth.BackchannelLogoutIdentity{
				IssuerID: issuerID, TokenID: "logout-token-" + uuid.NewString(), Subject: subject,
				IssuedAt: logoutIssuedAt, ExpiresAt: time.Now().UTC().Add(5 * time.Minute),
				ClaimsDigest: sha256.Sum256([]byte("logout-claims-" + uuid.NewString())),
			}
			var issuerSessionPointer *string
			if testCase.sid {
				logoutIdentity.IssuerSessionID = issuerSessionID
				issuerSessionPointer = &issuerSessionID
			}
			logoutRequest := managementidentity.BackchannelLogout{
				Identity: logoutIdentity, RequestID: "logout-race-" + uuid.NewString(),
			}
			staleExchange := issuerLogoutExchangeRequest(
				issuerID, issuer, subject, issuerSessionPointer, staleAuthenticatedAt,
			)

			type logoutOutcome struct {
				result managementidentity.BackchannelLogoutResult
				err    error
			}
			type exchangeOutcome struct {
				result managementauth.IdentityExchangeResult
				err    error
			}
			ready := make(chan struct{}, 2)
			start := make(chan struct{})
			logoutResults := make(chan logoutOutcome, 1)
			exchangeResults := make(chan exchangeOutcome, 1)
			go func() {
				ready <- struct{}{}
				<-start
				result, err := identityStore.ApplyBackchannelLogout(ctx, logoutRequest)
				logoutResults <- logoutOutcome{result: result, err: err}
			}()
			go func() {
				ready <- struct{}{}
				<-start
				result, err := exchanges.ExchangeIdentity(ctx, staleExchange, issuerLogoutTestIssuer)
				exchangeResults <- exchangeOutcome{result: result, err: err}
			}()
			<-ready
			<-ready
			close(start)
			logoutResult := <-logoutResults
			exchangeResult := <-exchangeResults
			if logoutResult.err != nil {
				t.Fatalf("concurrent ApplyBackchannelLogout() error = %v", logoutResult.err)
			}
			if exchangeResult.err != nil && !errors.Is(exchangeResult.err, managementauth.ErrAuthenticationDenied) {
				t.Fatalf("concurrent ExchangeIdentity() error = %v", exchangeResult.err)
			}
			var activeSessions int
			if err := database.QueryRowContext(ctx, `SELECT count(*) FROM management_sessions
WHERE principal_id=$1 AND status='active'`, principalID).Scan(&activeSessions); err != nil || activeSessions != 0 {
				t.Fatalf("active sessions after concurrent logout = %d, %v", activeSessions, err)
			}

			replayed, err := identityStore.ApplyBackchannelLogout(ctx, logoutRequest)
			if err != nil || !replayed.Replayed {
				t.Fatalf("ApplyBackchannelLogout() replay = %+v, %v", replayed, err)
			}
			staleExchange.Session.ID = uuid.NewString()
			staleExchange.Session.TokenID = "stale-token-" + uuid.NewString()
			if _, err := exchanges.ExchangeIdentity(ctx, staleExchange, issuerLogoutTestIssuer); !errors.Is(err, managementauth.ErrAuthenticationDenied) {
				t.Fatalf("stale ExchangeIdentity() error = %v", err)
			}
			var selectorCount int
			selectorKind := "subject"
			if testCase.sid {
				selectorKind = "sid"
			}
			if err := database.QueryRowContext(ctx, `SELECT count(*)
FROM management_issuer_logout_tombstones WHERE issuer_id=$1 AND selector_kind=$2`,
				issuerID, selectorKind,
			).Scan(&selectorCount); err != nil || selectorCount != 1 {
				t.Fatalf("logout tombstones = %d, %v", selectorCount, err)
			}

			freshExchange := issuerLogoutExchangeRequest(
				issuerID, issuer, subject, issuerSessionPointer, logoutIssuedAt.Add(time.Minute),
			)
			if testCase.sid {
				_, err = exchanges.ExchangeIdentity(ctx, freshExchange, issuerLogoutTestIssuer)
				if !errors.Is(err, managementauth.ErrAuthenticationDenied) {
					t.Fatalf("same-SID fresh ExchangeIdentity() error = %v", err)
				}
			} else {
				freshReady := make(chan struct{}, 2)
				freshStart := make(chan struct{})
				freshLogoutResults := make(chan logoutOutcome, 1)
				freshExchangeResults := make(chan exchangeOutcome, 1)
				go func() {
					freshReady <- struct{}{}
					<-freshStart
					result, err := identityStore.ApplyBackchannelLogout(ctx, logoutRequest)
					freshLogoutResults <- logoutOutcome{result: result, err: err}
				}()
				go func() {
					freshReady <- struct{}{}
					<-freshStart
					result, err := exchanges.ExchangeIdentity(ctx, freshExchange, issuerLogoutTestIssuer)
					freshExchangeResults <- exchangeOutcome{result: result, err: err}
				}()
				<-freshReady
				<-freshReady
				close(freshStart)
				freshLogout := <-freshLogoutResults
				freshResult := <-freshExchangeResults
				if freshLogout.err != nil || !freshLogout.result.Replayed {
					t.Fatalf("concurrent subject logout replay = %+v", freshLogout)
				}
				if freshResult.err != nil || freshResult.result.Issued.ManagementSessionID == "" {
					t.Fatalf("concurrent subject reauthentication = %+v", freshResult)
				}
				if err := database.QueryRowContext(ctx, `SELECT count(*) FROM management_sessions
WHERE principal_id=$1 AND status='active'`, principalID).Scan(&activeSessions); err != nil || activeSessions != 1 {
					t.Fatalf("fresh subject sessions after concurrent logout = %d, %v", activeSessions, err)
				}
			}
		})
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
