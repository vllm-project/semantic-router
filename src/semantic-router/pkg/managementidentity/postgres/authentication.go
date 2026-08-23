package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

func (store *Store) ResolvePrincipal(ctx context.Context, issuer, subject string) (string, error) {
	if store == nil || store.database == nil || issuer == "" || subject == "" ||
		strings.TrimSpace(issuer) != issuer || strings.TrimSpace(subject) != subject {
		return "", managementauth.ErrAuthenticationDenied
	}
	var id, status string
	if err := store.database.QueryRowContext(ctx, `SELECT id::text,status
FROM management_principals WHERE issuer=$1 AND subject=$2`, issuer, subject).Scan(&id, &status); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", managementauth.ErrAuthenticationDenied
		}
		return "", fmt.Errorf("resolve Management principal: %w", managementauth.ErrAuthenticationUnavailable)
	}
	if !canonicalUUID(id) || status != "active" {
		return "", managementauth.ErrAuthenticationDenied
	}
	return id, nil
}

type ServiceCredentialVerifier struct {
	database *sql.DB
	peppers  securitykeyring.Symmetric
}

func NewServiceCredentialVerifier(database *sql.DB, peppers securitykeyring.Symmetric) (*ServiceCredentialVerifier, error) {
	if database == nil || peppers.ActiveVersion == "" || len(peppers.Keys) == 0 {
		return nil, errors.New("management service-credential database and pepper keyring are required")
	}
	keys := make(map[string][]byte, len(peppers.Keys))
	for version, key := range peppers.Keys {
		if version == "" || len(key) != sha256.Size {
			return nil, errors.New("management service-credential pepper versions require 256-bit keys")
		}
		keys[version] = append([]byte(nil), key...)
	}
	if _, found := keys[peppers.ActiveVersion]; !found {
		return nil, errors.New("active Management service-credential pepper is unavailable")
	}
	return &ServiceCredentialVerifier{database: database, peppers: securitykeyring.Symmetric{ActiveVersion: peppers.ActiveVersion, Keys: keys}}, nil
}

func (verifier *ServiceCredentialVerifier) Ready(ctx context.Context) error {
	if verifier == nil || verifier.database == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	rows, err := verifier.database.QueryContext(ctx, `SELECT DISTINCT pepper_version
FROM management_service_account_credentials
WHERE status IN ('active','retiring') AND expires_at>clock_timestamp()`)
	if err != nil {
		return fmt.Errorf("read live Management service-credential pepper versions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			return err
		}
		if _, found := verifier.peppers.Keys[version]; !found {
			return fmt.Errorf("live Management service credential references unavailable pepper version %q", version)
		}
	}
	return rows.Err()
}

// Close erases the service-credential peppers cloned for this verifier.
func (verifier *ServiceCredentialVerifier) Close() {
	if verifier == nil {
		return
	}
	for _, key := range verifier.peppers.Keys {
		zeroBytes(key)
	}
	verifier.peppers = securitykeyring.Symmetric{}
}

func (verifier *ServiceCredentialVerifier) VerifyServiceCredential(ctx context.Context, encoded string, now time.Time) (managementauth.VerifiedServiceCredential, error) {
	publicID, secret, ok := parseServiceCredential(encoded)
	if verifier == nil || !ok || now.IsZero() {
		return managementauth.VerifiedServiceCredential{}, managementauth.ErrAuthenticationDenied
	}
	defer zeroBytes(secret)
	var (
		credentialID, principalID, pepperVersion, workloadClass string
		storedHMAC                                              []byte
		notBefore, expiresAt, sourceAssuredAt                   time.Time
		credentialStatus, accountStatus, principalStatus        string
	)
	err := verifier.database.QueryRowContext(ctx, `SELECT credential.id::text,account.principal_id::text,
       credential.secret_hmac,credential.pepper_version,credential.workload_class,
       credential.source_assured_at,credential.not_before,credential.expires_at,
       credential.status,account.status,principal.status
FROM management_service_account_credentials credential
JOIN management_service_accounts account ON account.id=credential.service_account_id
JOIN management_principals principal ON principal.id=account.principal_id
WHERE credential.public_id=$1`, publicID).Scan(
		&credentialID, &principalID, &storedHMAC, &pepperVersion, &workloadClass,
		&sourceAssuredAt, &notBefore, &expiresAt, &credentialStatus, &accountStatus, &principalStatus,
	)
	if err != nil {
		return managementauth.VerifiedServiceCredential{}, managementauth.ErrAuthenticationDenied
	}
	pepper, found := verifier.peppers.Keys[pepperVersion]
	if !found || len(storedHMAC) != sha256.Size || !canonicalUUID(credentialID) || !canonicalUUID(principalID) {
		return managementauth.VerifiedServiceCredential{}, managementauth.ErrAuthenticationDenied
	}
	computed := ComputeServiceCredentialHMAC(pepper, credentialID, secret)
	if subtle.ConstantTimeCompare(storedHMAC, computed[:]) != 1 ||
		(credentialStatus != "active" && credentialStatus != "retiring") || accountStatus != "active" || principalStatus != "active" ||
		now.Before(notBefore) || !now.Before(expiresAt) || sourceAssuredAt.After(now) {
		return managementauth.VerifiedServiceCredential{}, managementauth.ErrAuthenticationDenied
	}
	return managementauth.VerifiedServiceCredential{
		PrincipalID: principalID, CredentialID: credentialID, WorkloadClass: workloadClass,
		SourceAssuredAt: sourceAssuredAt.UTC(), EvidenceExpiresAt: expiresAt.UTC(),
	}, nil
}

func ComputeServiceCredentialHMAC(pepper []byte, credentialID string, secret []byte) [sha256.Size]byte {
	return managementidentity.ComputeServiceCredentialHMAC(pepper, credentialID, secret)
}

func parseServiceCredential(value string) (string, []byte, bool) {
	if len(value) > 256 || !strings.HasPrefix(value, "vsm_") || strings.TrimSpace(value) != value {
		return "", nil, false
	}
	publicID, encodedSecret, found := strings.Cut(strings.TrimPrefix(value, "vsm_"), "_")
	if !found || !canonicalUUID(publicID) || len(encodedSecret) != 43 {
		return "", nil, false
	}
	secret, err := base64.RawURLEncoding.DecodeString(encodedSecret)
	if err != nil {
		return "", nil, false
	}
	if len(secret) != 32 || base64.RawURLEncoding.EncodeToString(secret) != encodedSecret {
		zeroBytes(secret)
		return "", nil, false
	}
	return publicID, secret, true
}

var (
	_ managementauth.PrincipalResolver         = (*Store)(nil)
	_ managementauth.ServiceCredentialVerifier = (*ServiceCredentialVerifier)(nil)
)
