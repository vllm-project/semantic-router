package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"net/url"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	controlplanepostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func TestBootstrapExactReplayConflictExpiryAndRedaction(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	service := newBootstrapTestService(t, database, func() time.Time { return now })
	request := bootstrapTestRequest("bootstrap-key-0001", "First administrator")

	first, err := service.Bootstrap(context.Background(), request, bootstrapTestToken)
	if err != nil {
		t.Fatalf("bootstrap: %v", err)
	}
	if first.Replayed || first.ResponseStatus != 201 || first.PrincipalID == "" ||
		first.RoleBindingID == "" || first.ServiceAccountID == "" ||
		first.ServiceCredentialID == "" || !strings.HasPrefix(first.ServiceCredential, "vsm_") ||
		!first.ServiceCredentialExpiresAt.Equal(now.Add(30*24*time.Hour)) {
		t.Fatalf("unexpected bootstrap result: %+v", first)
	}

	replayed, err := service.Bootstrap(context.Background(), request, bootstrapTestToken)
	if err != nil {
		t.Fatalf("replay bootstrap: %v", err)
	}
	if !replayed.Replayed || replayed.PrincipalID != first.PrincipalID ||
		replayed.ServiceCredential != first.ServiceCredential {
		t.Fatalf("replay did not return the exact original result: %+v", replayed)
	}

	mismatch := bootstrapTestRequest(request.IdempotencyKey, "Different administrator")
	if _, err := service.Bootstrap(context.Background(), mismatch, bootstrapTestToken); !errors.Is(err, managementidentity.ErrBootstrapConflict) {
		t.Fatalf("payload mismatch error = %v, want conflict", err)
	}

	var persisted string
	if err := database.QueryRow(`SELECT receipt::text || ' ' || COALESCE(string_agg(details::text, ' '), '')
FROM management_installation_state CROSS JOIN access_audit_events
GROUP BY receipt`).Scan(&persisted); err != nil {
		t.Fatalf("read safe bootstrap metadata: %v", err)
	}
	if strings.Contains(persisted, first.ServiceCredential) || strings.Contains(persisted, "vsm_") {
		t.Fatal("bootstrap secret was persisted in receipt or audit metadata")
	}
	var emptyActorChain bool
	if err := database.QueryRow(`SELECT actor_chain = '[]'::jsonb
FROM access_audit_events
WHERE action='management.bootstrap' AND resource_id=$1`, first.PrincipalID).Scan(&emptyActorChain); err != nil {
		t.Fatalf("read bootstrap audit actor chain: %v", err)
	}
	if !emptyActorChain {
		t.Fatal("bootstrap audit actor chain was not persisted as an empty array")
	}

	now = now.Add(bootstrapResultTTL + time.Second)
	if _, err := service.Bootstrap(context.Background(), request, bootstrapTestToken); !errors.Is(err, managementidentity.ErrBootstrapResultExpired) {
		t.Fatalf("expired replay error = %v, want gone", err)
	}
	var ciphertext, nonce []byte
	var keyVersion sql.NullString
	if err := database.QueryRow(`SELECT bootstrap_response_ciphertext,
 bootstrap_response_nonce, bootstrap_response_kek_version
FROM management_installation_state WHERE singleton=TRUE`).Scan(&ciphertext, &nonce, &keyVersion); err != nil {
		t.Fatalf("read cleared bootstrap envelope: %v", err)
	}
	if len(ciphertext) != 0 || len(nonce) != 0 || keyVersion.Valid {
		t.Fatal("expired bootstrap response envelope was not cleared")
	}
}

func TestBootstrapConcurrentReplicasReturnOneResult(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	firstService := newBootstrapTestService(t, database, func() time.Time { return now })
	secondService := newBootstrapTestService(t, database, func() time.Time { return now })
	request := bootstrapTestRequest("bootstrap-key-0002", "Concurrent administrator")

	start := make(chan struct{})
	results := make(chan managementidentity.BootstrapResult, 2)
	errorsFound := make(chan error, 2)
	var wait sync.WaitGroup
	for _, service := range []*BootstrapService{firstService, secondService} {
		wait.Add(1)
		go func(service *BootstrapService) {
			defer wait.Done()
			<-start
			result, err := service.Bootstrap(context.Background(), request, bootstrapTestToken)
			if err != nil {
				errorsFound <- err
				return
			}
			results <- result
		}(service)
	}
	close(start)
	wait.Wait()
	close(results)
	close(errorsFound)
	for err := range errorsFound {
		t.Fatalf("concurrent bootstrap: %v", err)
	}
	var values []managementidentity.BootstrapResult
	for result := range results {
		values = append(values, result)
	}
	if len(values) != 2 || values[0].PrincipalID != values[1].PrincipalID ||
		values[0].ServiceCredential != values[1].ServiceCredential ||
		values[0].Replayed == values[1].Replayed {
		t.Fatalf("concurrent bootstrap results are not winner/replay: %+v", values)
	}
	var principalCount, bindingCount int
	if err := database.QueryRow(`SELECT
 (SELECT count(*) FROM management_principals),
 (SELECT count(*) FROM management_role_bindings)`).Scan(&principalCount, &bindingCount); err != nil {
		t.Fatal(err)
	}
	if principalCount != 1 || bindingCount != 1 {
		t.Fatalf("bootstrap created %d principals and %d bindings", principalCount, bindingCount)
	}
}

func TestBootstrapReadinessFailsWhenRetainedKeysAreRemoved(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	service := newBootstrapTestService(t, database, func() time.Time { return now })
	if _, err := service.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-0003", "Administrator"), bootstrapTestToken); err != nil {
		t.Fatal(err)
	}
	if err := service.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a consumed bootstrap while its token remains configured")
	}
	finalizedOptions := bootstrapTestOptions(database, func() time.Time { return now })
	finalizedOptions.BootstrapToken = nil
	finalized, finalizedErr := NewBootstrapService(finalizedOptions)
	if finalizedErr != nil {
		t.Fatal(finalizedErr)
	}
	if err := finalized.Ready(context.Background()); err != nil {
		t.Fatalf("finalized bootstrap readiness: %v", err)
	}
	missing := bootstrapTestOptions(database, func() time.Time { return now })
	missing.BootstrapToken = nil
	missing.IdempotencyKeys = securitykeyring.Symmetric{
		ActiveVersion: "v2", Keys: map[string][]byte{"v2": bytes.Repeat([]byte{9}, 32)},
	}
	withoutRetained, retainedErr := NewBootstrapService(missing)
	if retainedErr != nil {
		t.Fatal(retainedErr)
	}
	if err := withoutRetained.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a live bootstrap result with a missing HMAC version")
	}
}

func TestBootstrapFileFinalizationErasesInMemoryAuthority(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 22, 1, 2, 3, 0, time.UTC)
	present := true
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.BootstrapTokenPresent = func() (bool, error) { return present, nil }
	service, err := NewBootstrapService(options)
	if err != nil {
		t.Fatal(err)
	}
	request := bootstrapTestRequest("bootstrap-key-finalize-0001", "Administrator")
	if _, err := service.Bootstrap(context.Background(), request, bootstrapTestToken); err != nil {
		t.Fatal(err)
	}
	if err := service.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a consumed bootstrap before source removal")
	}
	present = false
	if err := service.Ready(context.Background()); err != nil {
		t.Fatalf("readiness did not converge after source removal: %v", err)
	}
	present = true
	if _, err := service.Bootstrap(context.Background(), request, bootstrapTestToken); !errors.Is(err, managementidentity.ErrBootstrapUnavailable) {
		t.Fatalf("recreated source restored spent authority: %v", err)
	}
}

func TestBootstrapReadinessRejectsAuthorizationWithoutLoginAuthority(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	principalID := uuid.NewString()
	if _, err := database.Exec(`INSERT INTO management_principals
  (id,issuer,subject,display_name,status,revision) VALUES ($1,$2,$3,'Imported administrator','active',1)`,
		principalID, "https://missing-issuer.example.test", uuid.NewString()); err != nil {
		t.Fatal(err)
	}
	if _, err := database.Exec(`INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,status,revision)
VALUES ($1,$2,'10000000-0000-5000-8000-000000000001','cluster','active',1)`, uuid.NewString(), principalID); err != nil {
		t.Fatal(err)
	}
	service := newBootstrapTestService(t, database, func() time.Time { return now })
	if err := service.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted an unconsumed installation with an imported cluster administrator")
	}

	database = bootstrapTestDatabase(t)
	service = newBootstrapTestService(t, database, func() time.Time { return now })
	result, bootstrapErr := service.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-readiness-0001", "Bootstrap administrator"), bootstrapTestToken)
	if bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	if _, err := database.Exec(`UPDATE management_service_account_credentials
SET status='revoked',revoked_at=clock_timestamp() WHERE id=$1`, result.ServiceCredentialID); err != nil {
		t.Fatal(err)
	}
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.BootstrapToken = nil
	finalized, finalizedErr := NewBootstrapService(options)
	if finalizedErr != nil {
		t.Fatal(finalizedErr)
	}
	if err := finalized.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a consumed installation without a login-capable cluster administrator")
	}
}

func TestBootstrapReadinessAcceptsTrustedExternalAdministrator(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	bootstrap := newBootstrapTestService(t, database, func() time.Time { return now })
	bootstrapResult, bootstrapErr := bootstrap.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-readiness-0002", "Bootstrap administrator"), bootstrapTestToken)
	if bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	if _, err := database.Exec(`UPDATE management_principals
SET status='disabled',revision=revision+1 WHERE id=$1`, bootstrapResult.PrincipalID); err != nil {
		t.Fatal(err)
	}
	issuerID, principalID := uuid.NewString(), uuid.NewString()
	issuer := "https://dashboard.example.test"
	if _, err := database.Exec(`INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,audiences,claim_mapping,assurance_mapping,status,revision)
VALUES ($1,$2,'oidc',$3,'["vllm-sr-management"]'::jsonb,'{}'::jsonb,'{}'::jsonb,'active',1)`,
		issuerID, issuer, issuer+"/.well-known/openid-configuration"); err != nil {
		t.Fatal(err)
	}
	if _, err := database.Exec(`INSERT INTO management_principals
  (id,issuer,subject,display_name,status,revision) VALUES ($1,$2,$3,'External administrator','active',1)`,
		principalID, issuer, uuid.NewString()); err != nil {
		t.Fatal(err)
	}
	if _, err := database.Exec(`INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,status,revision)
VALUES ($1,$2,'10000000-0000-5000-8000-000000000001','cluster','active',1)`, uuid.NewString(), principalID); err != nil {
		t.Fatal(err)
	}
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.BootstrapToken = nil
	service, serviceErr := NewBootstrapService(options)
	if serviceErr != nil {
		t.Fatal(serviceErr)
	}
	if err := service.Ready(context.Background()); err != nil {
		t.Fatalf("trusted external administrator readiness: %v", err)
	}
}

func TestBootstrapReadinessAcceptsRetiringServiceCredential(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	bootstrap := newBootstrapTestService(t, database, func() time.Time { return now })
	result, bootstrapErr := bootstrap.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-readiness-0003", "Bootstrap administrator"), bootstrapTestToken)
	if bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	if _, err := database.Exec(`UPDATE management_service_account_credentials
SET status='retiring' WHERE id=$1`, result.ServiceCredentialID); err != nil {
		t.Fatal(err)
	}
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.BootstrapToken = nil
	service, serviceErr := NewBootstrapService(options)
	if serviceErr != nil {
		t.Fatal(serviceErr)
	}
	if err := service.Ready(context.Background()); err != nil {
		t.Fatalf("retiring service credential readiness: %v", err)
	}
}

func TestBootstrapReadinessRejectsFutureServiceCredentialAssurance(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	bootstrap := newBootstrapTestService(t, database, func() time.Time { return now })
	result, bootstrapErr := bootstrap.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-readiness-0004", "Bootstrap administrator"), bootstrapTestToken)
	if bootstrapErr != nil {
		t.Fatal(bootstrapErr)
	}
	if _, err := database.Exec(`UPDATE management_service_account_credentials
SET source_assured_at=$2 WHERE id=$1`, result.ServiceCredentialID, now.Add(time.Minute)); err != nil {
		t.Fatal(err)
	}
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.BootstrapToken = nil
	service, serviceErr := NewBootstrapService(options)
	if serviceErr != nil {
		t.Fatal(serviceErr)
	}
	if err := service.Ready(context.Background()); err == nil {
		t.Fatal("readiness accepted a service credential with future assurance")
	}
}

func TestBootstrapServiceCredentialAuthenticates(t *testing.T) {
	database := bootstrapTestDatabase(t)
	now := time.Date(2026, 8, 24, 0, 0, 0, 0, time.UTC)
	options := bootstrapTestOptions(database, func() time.Time { return now })
	options.RandomBytes = func(destination []byte) (int, error) {
		for index := range destination {
			destination[index] = 0xff
		}
		return len(destination), nil
	}
	service, err := NewBootstrapService(options)
	if err != nil {
		t.Fatal(err)
	}
	result, err := service.Bootstrap(context.Background(), bootstrapTestRequest("bootstrap-key-authentication-0001", "Bootstrap administrator"), bootstrapTestToken)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Count(result.ServiceCredential, "_") <= 2 {
		t.Fatal("bootstrap fixture did not exercise URL-safe base64 underscore")
	}
	verifier, err := NewServiceCredentialVerifier(database, options.ServiceCredentialPeppers)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(verifier.Close)
	verified, err := verifier.VerifyServiceCredential(context.Background(), result.ServiceCredential, now.Add(time.Minute))
	if err != nil {
		t.Fatalf("authenticate bootstrap service credential: %v", err)
	}
	if verified.PrincipalID != result.PrincipalID || verified.CredentialID != result.ServiceCredentialID ||
		!verified.EvidenceExpiresAt.Equal(result.ServiceCredentialExpiresAt) {
		t.Fatalf("authenticated bootstrap identity = %+v", verified)
	}
}

const bootstrapTestToken = "bootstrap-token-that-is-at-least-thirty-two-bytes"

func bootstrapTestRequest(key, displayName string) managementidentity.BootstrapRequest {
	canonical := []byte(`{"kind":"service_account","displayName":"` + displayName + `"}`)
	return managementidentity.BootstrapRequest{
		Kind: managementidentity.BootstrapServiceAccount, DisplayName: displayName,
		IdempotencyKey: key, CanonicalRequest: canonical,
	}
}

func newBootstrapTestService(t *testing.T, database *sql.DB, now func() time.Time) *BootstrapService {
	t.Helper()
	service, err := NewBootstrapService(bootstrapTestOptions(database, now))
	if err != nil {
		t.Fatal(err)
	}
	return service
}

func bootstrapTestOptions(database *sql.DB, now func() time.Time) BootstrapOptions {
	return BootstrapOptions{
		Database: database, BootstrapToken: []byte(bootstrapTestToken),
		IdempotencyKeys: securitykeyring.Symmetric{
			ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{1}, 32)},
		},
		ResponseKEKs: accesscredential.KEKKeyring{
			ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{2}, 32)},
		},
		ServiceCredentialPeppers: securitykeyring.Symmetric{
			ActiveVersion: "v1", Keys: map[string][]byte{"v1": bytes.Repeat([]byte{3}, 32)},
		},
		Now: now,
	}
}

func bootstrapTestDatabase(t *testing.T) *sql.DB {
	t.Helper()
	databaseURL := os.Getenv("VLLM_SR_CONTROL_PLANE_TEST_DATABASE_URL")
	if databaseURL == "" {
		t.Skip("PostgreSQL test database is not configured")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	t.Cleanup(cancel)
	admin, openErr := sql.Open("postgres", databaseURL)
	if openErr != nil {
		t.Fatal(openErr)
	}
	t.Cleanup(func() { _ = admin.Close() })
	schema := "vsr_bootstrap_test_" + strings.ReplaceAll(uuid.NewString(), "-", "")
	if _, err := admin.ExecContext(ctx, "CREATE SCHEMA "+pq.QuoteIdentifier(schema)); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		cleanup, stop := context.WithTimeout(context.Background(), 15*time.Second)
		defer stop()
		_, _ = admin.ExecContext(cleanup, "DROP SCHEMA "+pq.QuoteIdentifier(schema)+" CASCADE")
	})
	parsed, err := url.Parse(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	query := parsed.Query()
	query.Set("search_path", schema)
	parsed.RawQuery = query.Encode()
	database, err := sql.Open("postgres", parsed.String())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = database.Close() })
	if err := (controlplanepostgres.Migrator{DB: database}).Apply(ctx); err != nil {
		t.Fatalf("apply control-plane migrations: %v", err)
	}
	return database
}
