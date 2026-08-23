package postgres

import (
	"context"
	"crypto/ed25519"
	"database/sql"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"

	controlplanepostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type clockAuthorityCredentials struct {
	verified managementauth.VerifiedServiceCredential
}

func (credentials clockAuthorityCredentials) VerifyServiceCredential(context.Context, string, time.Time) (managementauth.VerifiedServiceCredential, error) {
	return credentials.verified, nil
}

type clockAuthorityChallenges struct{}

func (clockAuthorityChallenges) Ready(context.Context) error { return nil }

func (clockAuthorityChallenges) Create(context.Context, string, string, time.Time) (managementauth.ExchangeChallenge, error) {
	return managementauth.ExchangeChallenge{}, errors.New("not used")
}
func (clockAuthorityChallenges) Consume(context.Context, string, string, string, time.Time) error {
	return errors.New("not used")
}

type clockAuthorityAssertions struct{}

func (clockAuthorityAssertions) ValidateIssuer(context.Context, string) error {
	return errors.New("not used")
}
func (clockAuthorityAssertions) Verify(context.Context, string, managementauth.SubjectTokenType, string, time.Time) (managementauth.VerifiedExternalIdentity, error) {
	return managementauth.VerifiedExternalIdentity{}, errors.New("not used")
}

type clockAuthorityExchanges struct{}

func (clockAuthorityExchanges) Ready(context.Context) error { return nil }
func (clockAuthorityExchanges) ExchangeIdentity(context.Context, managementauth.IdentityExchangeRequest, managementauth.PreparedSessionIssuer) (managementauth.IdentityExchangeResult, error) {
	return managementauth.IdentityExchangeResult{}, errors.New("not used")
}

type clockAuthorityMTLS struct{}

func (clockAuthorityMTLS) ResolveMTLSIdentity(context.Context, managementauth.VerifiedMTLSEvidence, time.Time) (managementauth.VerifiedMTLSIdentity, error) {
	return managementauth.VerifiedMTLSIdentity{}, errors.New("not used")
}

type clockAuthorityBarriers struct{}

func (clockAuthorityBarriers) Check(context.Context, managementauth.BarrierCheck) (managementauth.BarrierState, error) {
	return managementauth.BarrierState{Ready: true}, nil
}
func (clockAuthorityBarriers) InstallDeny(context.Context, managementauth.BarrierKind, string) error {
	return nil
}

func TestAuthServiceUsesPostgresSessionClockAuthority(t *testing.T) {
	dsn := os.Getenv("MANAGEMENTAUTH_TEST_POSTGRES_DSN")
	if dsn == "" {
		t.Skip("MANAGEMENTAUTH_TEST_POSTGRES_DSN is not configured")
	}
	database, openErr := sql.Open("postgres", dsn)
	if openErr != nil {
		t.Fatal(openErr)
	}
	t.Cleanup(func() { _ = database.Close() })
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if migrateErr := (controlplanepostgres.Migrator{DB: database}).Apply(ctx); migrateErr != nil {
		t.Fatal(migrateErr)
	}

	principalID, accountID, credentialID := uuid.NewString(), uuid.NewString(), uuid.NewString()
	clientNow := time.Now().UTC().Add(-time.Hour).Truncate(time.Second)
	assuredAt := clientNow.Add(-time.Minute)
	evidenceExpiresAt := clientNow.Add(4 * time.Hour)
	if _, insertErr := database.ExecContext(ctx, `INSERT INTO management_principals
	  (id,issuer,subject,display_name,status) VALUES ($1,'urn:vllm-sr:service-account',$2,'Clock authority','active')`, principalID, accountID); insertErr != nil {
		t.Fatal(insertErr)
	}
	if _, insertErr := database.ExecContext(ctx, `INSERT INTO management_service_accounts
	  (id,principal_id,owner_scope,status) VALUES ($1,$2,'cluster','active')`, accountID, principalID); insertErr != nil {
		t.Fatal(insertErr)
	}
	if _, insertErr := database.ExecContext(ctx, `INSERT INTO management_service_account_credentials
	  (id,service_account_id,public_id,secret_hmac,pepper_version,workload_class,source_assured_at,status,not_before,expires_at)
	VALUES ($1,$2,$3,$4,'test','workload_strong',$5,'active',$6,$7)`, credentialID, accountID, credentialID, make([]byte, 32), assuredAt, clientNow.Add(-time.Minute), evidenceExpiresAt); insertErr != nil {
		t.Fatal(insertErr)
	}
	store, err := New(database)
	if err != nil {
		t.Fatal(err)
	}
	publicKey, privateKey, err := ed25519.GenerateKey(nil)
	if err != nil {
		t.Fatal(err)
	}
	codec := managementauth.TokenCodec{
		Keyring: securitykeyring.Signing{ActiveVersion: "test", Private: map[string]ed25519.PrivateKey{"test": privateKey}, Public: map[string]ed25519.PublicKey{"test": publicKey}},
		Issuer:  "vllm-sr", Audience: "vllm-sr-management", MaxSkew: 5 * time.Second,
	}
	runtime := managementauth.SessionRuntime{Codec: codec, Sessions: store, Barriers: clockAuthorityBarriers{}, PolicyLoader: store}
	ids := []string{uuid.NewString(), uuid.NewString()}
	nextID := 0
	service, err := managementauth.NewAuthService(managementauth.AuthServiceOptions{
		Challenges: clockAuthorityChallenges{}, Assertions: clockAuthorityAssertions{}, Exchanges: clockAuthorityExchanges{},
		ServiceCredentials: clockAuthorityCredentials{verified: managementauth.VerifiedServiceCredential{
			PrincipalID: principalID, CredentialID: credentialID, WorkloadClass: "workload_strong",
			SourceAssuredAt: assuredAt, EvidenceExpiresAt: evidenceExpiresAt,
		}},
		MTLSIdentities: clockAuthorityMTLS{}, Sessions: store, Runtime: runtime,
		Now: func() time.Time { return clientNow }, NewID: func() (string, error) {
			value := ids[nextID]
			nextID++
			return value, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	issued, err := service.ServiceToken(ctx, "opaque-test-credential")
	if err != nil {
		t.Fatalf("ServiceToken with server-authoritative CreatedAt: %v", err)
	}
	var createdAt time.Time
	if err := database.QueryRowContext(ctx, `SELECT created_at FROM management_sessions WHERE id=$1`, issued.ManagementSessionID).Scan(&createdAt); err != nil {
		t.Fatal(err)
	}
	if !createdAt.After(clientNow) {
		t.Fatalf("fixture did not create server/client clock ordering: created=%s client=%s", createdAt, clientNow)
	}
	if _, err := runtime.Authenticate(ctx, issued.AccessToken, "", createdAt.Add(time.Second)); err != nil {
		t.Fatalf("authenticate issued Management token: %v", err)
	}
}
