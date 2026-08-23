package postgres

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const (
	bootstrapResultTTL       = 15 * time.Minute
	bootstrapMaxRequestBytes = 128 << 10
)

type BootstrapOptions struct {
	Database                 *sql.DB
	BootstrapToken           []byte
	BootstrapTokenPresent    func() (bool, error)
	IdempotencyKeys          securitykeyring.Symmetric
	ResponseKEKs             accesscredential.KEKKeyring
	ServiceCredentialPeppers securitykeyring.Symmetric
	Now                      func() time.Time
	RandomBytes              func([]byte) (int, error)
}

type BootstrapService struct {
	database        *sql.DB
	tokenDigest     [sha256.Size]byte
	tokenConfigured bool
	idempotency     securitykeyring.Symmetric
	responseKEKs    accesscredential.KEKKeyring
	peppers         securitykeyring.Symmetric
	now             func() time.Time
	randomBytes     func([]byte) (int, error)
	tokenPresent    func() (bool, error)
	tokenMu         sync.RWMutex
}

type bootstrapDigest struct {
	version string
	key     [sha256.Size]byte
	request [sha256.Size]byte
}

type bootstrapState struct {
	consumedAt     sql.NullTime
	principalID    sql.NullString
	hmacVersion    sql.NullString
	keyDigest      []byte
	requestDigest  []byte
	ciphertext     []byte
	nonce          []byte
	kekVersion     sql.NullString
	responseStatus sql.NullInt64
	expiresAt      sql.NullTime
	deliveredAt    sql.NullTime
	revision       int64
}

type bootstrapTransactionResult struct {
	result   managementidentity.BootstrapResult
	terminal error
}

func NewBootstrapService(options BootstrapOptions) (*BootstrapService, error) {
	if options.Database == nil || (len(options.BootstrapToken) != 0 && len(options.BootstrapToken) < 32) ||
		validateSymmetric(options.IdempotencyKeys) != nil || validateSymmetric(options.ServiceCredentialPeppers) != nil ||
		options.ResponseKEKs.Validate() != nil {
		return nil, errors.New("management bootstrap dependencies are invalid")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	randomBytes := options.RandomBytes
	if randomBytes == nil {
		randomBytes = rand.Read
	}
	return &BootstrapService{
		database: options.Database, tokenDigest: sha256.Sum256(options.BootstrapToken),
		tokenConfigured: len(options.BootstrapToken) != 0,
		idempotency:     cloneSymmetric(options.IdempotencyKeys), responseKEKs: cloneKEKKeyring(options.ResponseKEKs),
		peppers: cloneSymmetric(options.ServiceCredentialPeppers), now: now, randomBytes: randomBytes,
		tokenPresent: options.BootstrapTokenPresent,
	}, nil
}

func (service *BootstrapService) Ready(ctx context.Context) error {
	if service == nil || service.database == nil {
		return managementidentity.ErrBootstrapUnavailable
	}
	var consumed sql.NullTime
	var version, kek sql.NullString
	var expires sql.NullTime
	if err := service.database.QueryRowContext(ctx, `SELECT bootstrap_consumed_at,bootstrap_idempotency_hmac_version,
       bootstrap_response_kek_version,bootstrap_result_expires_at
FROM management_installation_state WHERE singleton=TRUE`).Scan(&consumed, &version, &kek, &expires); err != nil {
		return fmt.Errorf("load Management bootstrap readiness: %w", err)
	}
	tokenConfigured, err := service.refreshTokenState()
	if err != nil {
		return fmt.Errorf("inspect Management bootstrap token source: %w", err)
	}
	if consumed.Valid == tokenConfigured {
		if consumed.Valid {
			return errors.New("management bootstrap token must be removed after consumption")
		}
		return errors.New("management bootstrap token is required before first administrator creation")
	}
	activeAdministrators, loginCapableAdministrators, err := service.installationAdministratorState(ctx)
	if err != nil {
		return err
	}
	if !consumed.Valid {
		if activeAdministrators != 0 {
			return errors.New("management bootstrap is unconsumed but an active cluster administrator already exists")
		}
	} else if loginCapableAdministrators == 0 {
		return errors.New("management bootstrap is consumed without a login-capable cluster administrator")
	}
	if version.Valid {
		if _, found := service.idempotency.Keys[version.String]; !found {
			return errors.New("management bootstrap references an unavailable idempotency HMAC version")
		}
	}
	if kek.Valid && expires.Valid && service.now().UTC().Before(expires.Time) {
		if _, found := service.responseKEKs.Keys[kek.String]; !found {
			return errors.New("management bootstrap response references an unavailable KEK version")
		}
	}
	return nil
}

// installationAdministratorState distinguishes authorization state from an
// authentication path. An imported role binding must never make bootstrap
// look complete unless at least one active cluster administrator can actually
// authenticate through a trusted issuer or a live service credential.
func (service *BootstrapService) installationAdministratorState(ctx context.Context) (int, int, error) {
	var active, loginCapable int
	now := service.now().UTC()
	if err := service.database.QueryRowContext(ctx, `WITH cluster_administrators AS (
  SELECT DISTINCT principal.id,principal.issuer
  FROM management_principals principal
  JOIN management_role_bindings binding ON binding.principal_id=principal.id
  JOIN management_roles role ON role.id=binding.role_id
  WHERE principal.status='active' AND binding.status='active'
    AND binding.scope_kind='cluster' AND role.name='cluster_admin' AND role.builtin=TRUE
), administrator_state AS (
  SELECT administrator.id,
    EXISTS (
      SELECT 1 FROM trusted_identity_issuers issuer
      WHERE issuer.status='active' AND issuer.issuer=administrator.issuer
    ) OR EXISTS (
      SELECT 1 FROM management_service_accounts account
      JOIN management_service_account_credentials credential
        ON credential.service_account_id=account.id
      WHERE account.principal_id=administrator.id AND account.status='active'
        AND credential.status IN ('active','retiring')
        AND credential.source_assured_at <= $1
        AND credential.not_before <= $1 AND credential.expires_at > $1
    ) AS login_capable
  FROM cluster_administrators administrator
)
SELECT count(*),count(*) FILTER (WHERE login_capable) FROM administrator_state`, now).Scan(&active, &loginCapable); err != nil {
		return 0, 0, fmt.Errorf("load Management installation administrator readiness: %w", err)
	}
	return active, loginCapable, nil
}

// Close erases the bootstrap-only key material cloned for this service.
func (service *BootstrapService) Close() {
	if service == nil {
		return
	}
	for _, key := range service.idempotency.Keys {
		zeroBytes(key)
	}
	for _, key := range service.responseKEKs.Keys {
		zeroBytes(key)
	}
	for _, key := range service.peppers.Keys {
		zeroBytes(key)
	}
	service.idempotency = securitykeyring.Symmetric{}
	service.responseKEKs = accesscredential.KEKKeyring{}
	service.peppers = securitykeyring.Symmetric{}
	service.tokenMu.Lock()
	service.tokenDigest = [sha256.Size]byte{}
	service.tokenConfigured = false
	service.tokenPresent = nil
	service.tokenMu.Unlock()
}

func (service *BootstrapService) Bootstrap(ctx context.Context, request managementidentity.BootstrapRequest, presentedToken string) (managementidentity.BootstrapResult, error) {
	if validateBootstrapRequest(request) != nil {
		return managementidentity.BootstrapResult{}, managementidentity.ErrInvalidBootstrapRequest
	}
	if service == nil {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
	}
	configured, sourceErr := service.refreshTokenState()
	if sourceErr != nil || !configured || !service.validToken(presentedToken) {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
	}
	now := service.now().UTC()
	digests := service.bootstrapDigests(request)
	var (
		outcome bootstrapTransactionResult
		err     error
	)
	for attempt := 0; attempt < 5; attempt++ {
		outcome, err = service.bootstrapTransaction(ctx, request, digests, now)
		if !isBootstrapRetryable(err) {
			break
		}
	}
	if err != nil {
		return managementidentity.BootstrapResult{}, err
	}
	if outcome.terminal != nil {
		return managementidentity.BootstrapResult{}, outcome.terminal
	}
	result := outcome.result
	if _, err := service.database.ExecContext(ctx, `UPDATE management_installation_state
SET bootstrap_result_delivered_at=COALESCE(bootstrap_result_delivered_at,clock_timestamp()),
    updated_at=clock_timestamp()
WHERE singleton=TRUE AND bootstrap_principal_id=$1`, result.PrincipalID); err != nil {
		return managementidentity.BootstrapResult{}, fmt.Errorf("mark Management bootstrap result delivered: %w", err)
	}
	return result, nil
}

func (service *BootstrapService) bootstrapTransaction(ctx context.Context, request managementidentity.BootstrapRequest, digests []bootstrapDigest, now time.Time) (bootstrapTransactionResult, error) {
	return inTransaction(ctx, &Store{database: service.database}, sql.LevelSerializable, func(tx *sql.Tx) (bootstrapTransactionResult, error) {
		// Bootstrap is a security boundary. Even deployments that normally relax
		// commit latency must wait for the configured synchronous PostgreSQL
		// durability policy before returning the first administrator credential.
		if _, err := tx.ExecContext(ctx, `SET LOCAL synchronous_commit = on`); err != nil {
			return bootstrapTransactionResult{}, fmt.Errorf("require synchronous Management bootstrap commit: %w", err)
		}
		state, err := lockBootstrapState(ctx, tx)
		if err != nil {
			return bootstrapTransactionResult{}, err
		}
		if state.consumedAt.Valid {
			result, replayErr := service.replayBootstrap(ctx, tx, request, state, digests, now)
			if errors.Is(replayErr, managementidentity.ErrBootstrapResultExpired) {
				// The secret envelope deletion must commit before the public 410 is
				// returned. Other failures do not mutate durable state and roll back.
				return bootstrapTransactionResult{terminal: replayErr}, nil
			}
			return bootstrapTransactionResult{result: result}, replayErr
		}
		var existingAdmin bool
		if err := tx.QueryRowContext(ctx, `SELECT EXISTS(
  SELECT 1 FROM management_role_bindings binding
  JOIN management_roles role ON role.id=binding.role_id
  WHERE role.name='cluster_admin' AND role.builtin=TRUE
		    AND binding.scope_kind='cluster' AND binding.status='active')`).Scan(&existingAdmin); err != nil {
			return bootstrapTransactionResult{}, err
		}
		if existingAdmin {
			return bootstrapTransactionResult{}, managementidentity.ErrBootstrapConsumed
		}
		if err := validateBootstrapSeed(ctx, tx); err != nil {
			return bootstrapTransactionResult{}, err
		}
		result, err := service.commitBootstrap(ctx, tx, request, digests, now)
		return bootstrapTransactionResult{result: result}, err
	})
}

func isBootstrapRetryable(err error) bool {
	var databaseError *pq.Error
	return errors.As(err, &databaseError) &&
		(databaseError.Code == "40001" || databaseError.Code == "40P01")
}

func (service *BootstrapService) commitBootstrap(ctx context.Context, tx *sql.Tx, request managementidentity.BootstrapRequest, digests []bootstrapDigest, now time.Time) (managementidentity.BootstrapResult, error) {
	principalID, bindingID := uuid.NewString(), uuid.NewString()
	issuer, subject := request.Issuer, request.Subject
	var serviceAccountID, credentialID, credential string
	var credentialExpiresAt time.Time
	if request.Kind == managementidentity.BootstrapExternalPrincipal {
		audiences, _ := json.Marshal([]string{request.Audience})
		if _, err := tx.ExecContext(ctx, `INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,audiences,claim_mapping,assurance_mapping,status,revision)
VALUES ($1,$2,'oidc',$3,$4,'{}'::jsonb,'{}'::jsonb,'active',1)`,
			request.IssuerID, request.Issuer, request.DiscoveryURL, audiences); err != nil {
			return managementidentity.BootstrapResult{}, mapWriteError("create bootstrap identity issuer", err)
		}
	} else {
		serviceAccountID, credentialID = uuid.NewString(), uuid.NewString()
		issuer, subject = "urn:vllm-sr:service-account", serviceAccountID
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,attributes,status,revision)
VALUES ($1,$2,$3,$4,'{}'::jsonb,'active',1)`, principalID, issuer, subject, request.DisplayName); err != nil {
		return managementidentity.BootstrapResult{}, mapWriteError("create bootstrap principal", err)
	}
	ceiling, _ := json.Marshal(accesscontrol.DelegablePermissions().Permissions())
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,delegation_ceiling,status,revision)
VALUES ($1,$2,$3,'cluster',$4,'active',1)`, bindingID,
		principalID, builtInRoleID(accesscontrol.BuiltInRoleClusterAdmin), ceiling); err != nil {
		return managementidentity.BootstrapResult{}, mapWriteError("create bootstrap cluster administrator binding", err)
	}
	if request.Kind == managementidentity.BootstrapServiceAccount {
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_service_accounts
  (id,principal_id,owner_scope,status,revision) VALUES ($1,$2,'cluster','active',1)`, serviceAccountID, principalID); err != nil {
			return managementidentity.BootstrapResult{}, mapWriteError("create bootstrap service account", err)
		}
		secret := make([]byte, 32)
		if count, err := service.randomBytes(secret); err != nil || count != len(secret) {
			return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
		}
		defer zeroBytes(secret)
		credential = "vsm_" + credentialID + "_" + base64.RawURLEncoding.EncodeToString(secret)
		pepperVersion := service.peppers.ActiveVersion
		digest := ComputeServiceCredentialHMAC(service.peppers.Keys[pepperVersion], credentialID, secret)
		credentialExpiresAt = now.Add(30 * 24 * time.Hour)
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_service_account_credentials
  (id,service_account_id,public_id,secret_hmac,pepper_version,workload_class,
   source_assured_at,status,not_before,expires_at)
		VALUES ($1,$2,$3,$4,$5,'workload_strong',$6,'active',$6,$7)`, credentialID,
			serviceAccountID, credentialID, digest[:], pepperVersion, now, credentialExpiresAt); err != nil {
			return managementidentity.BootstrapResult{}, mapWriteError("create bootstrap service credential", err)
		}
	}
	result := managementidentity.BootstrapResult{
		PrincipalID: principalID, RoleBindingID: bindingID, ServiceAccountID: serviceAccountID,
		ServiceCredentialID: credentialID, ServiceCredential: credential,
		ServiceCredentialExpiresAt: credentialExpiresAt,
		FinalizationRequired:       true, ResponseStatus: 201,
	}
	plaintext, err := json.Marshal(result)
	if err != nil {
		return managementidentity.BootstrapResult{}, err
	}
	defer zeroBytes(plaintext)
	envelope, err := service.responseKEKs.Seal(plaintext, bootstrapAAD(principalID, bindingID))
	if err != nil {
		return managementidentity.BootstrapResult{}, fmt.Errorf("seal Management bootstrap response: %w", err)
	}
	safeReceipt, _ := json.Marshal(map[string]any{
		"principalId": principalID, "roleBindingId": bindingID,
		"serviceAccountId": serviceAccountID, "serviceCredentialId": credentialID,
		"serviceCredentialExpiresAt": credentialExpiresAt,
		"finalizationRequired":       true,
	})
	active := activeBootstrapDigest(service.idempotency.ActiveVersion, digests)
	update, err := tx.ExecContext(ctx, `UPDATE management_installation_state SET
  bootstrap_consumed_at=$1,bootstrap_principal_id=$2,
  bootstrap_idempotency_hmac_version=$3,bootstrap_idempotency_key_digest=$4,
  bootstrap_request_digest=$5,bootstrap_response_ciphertext=$6,
  bootstrap_response_nonce=$7,bootstrap_response_kek_version=$8,
  bootstrap_response_status=201,bootstrap_result_expires_at=$9,
  receipt=$10,revision=revision+1,updated_at=$1
WHERE singleton=TRUE AND bootstrap_consumed_at IS NULL`, now, principalID, active.version,
		active.key[:], active.request[:], envelope.Ciphertext, envelope.Nonce,
		envelope.KeyVersion, now.Add(bootstrapResultTTL), safeReceipt)
	if err != nil {
		return managementidentity.BootstrapResult{}, fmt.Errorf("consume Management bootstrap: %w", err)
	}
	if changed, err := update.RowsAffected(); err != nil || changed != 1 {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapConsumed
	}
	if err := appendAudit(ctx, tx, auditMutation{Action: "management.bootstrap", ResourceType: "management_principal", ResourceID: principalID, AfterRevision: 1, Actor: managementidentity.MutationActor{PrincipalID: principalID, RequestID: "bootstrap:" + bindingID, Reason: "Installation bootstrap"}}); err != nil {
		return managementidentity.BootstrapResult{}, err
	}
	return result, nil
}

func (service *BootstrapService) replayBootstrap(ctx context.Context, tx *sql.Tx, request managementidentity.BootstrapRequest, state bootstrapState, digests []bootstrapDigest, now time.Time) (managementidentity.BootstrapResult, error) {
	matched := false
	for _, candidate := range digests {
		if candidate.version == state.hmacVersion.String && len(state.keyDigest) == sha256.Size &&
			subtle.ConstantTimeCompare(candidate.key[:], state.keyDigest) == 1 {
			if len(state.requestDigest) != sha256.Size || subtle.ConstantTimeCompare(candidate.request[:], state.requestDigest) != 1 {
				return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapConflict
			}
			matched = true
		}
	}
	if !matched {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapConflict
	}
	if !state.expiresAt.Valid || !now.Before(state.expiresAt.Time) {
		if _, err := tx.ExecContext(ctx, `UPDATE management_installation_state SET
 bootstrap_response_ciphertext=NULL,bootstrap_response_nonce=NULL,
			bootstrap_response_kek_version=NULL,updated_at=clock_timestamp() WHERE singleton=TRUE`); err != nil {
			return managementidentity.BootstrapResult{}, fmt.Errorf("clear expired Management bootstrap response: %w", err)
		}
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapResultExpired
	}
	if len(state.ciphertext) == 0 || len(state.nonce) == 0 || !state.kekVersion.Valid ||
		!state.responseStatus.Valid || state.responseStatus.Int64 != 201 {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapResultExpired
	}
	var receipt struct {
		RoleBindingID string `json:"roleBindingId"`
	}
	var receiptJSON []byte
	if err := tx.QueryRowContext(ctx, `SELECT receipt FROM management_installation_state WHERE singleton=TRUE`).Scan(&receiptJSON); err != nil || json.Unmarshal(receiptJSON, &receipt) != nil {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
	}
	plaintext, err := service.responseKEKs.Open(accesscredential.Envelope{Ciphertext: state.ciphertext, Nonce: state.nonce, KeyVersion: state.kekVersion.String}, bootstrapAAD(state.principalID.String, receipt.RoleBindingID))
	if err != nil {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
	}
	defer zeroBytes(plaintext)
	var result managementidentity.BootstrapResult
	if err := json.Unmarshal(plaintext, &result); err != nil {
		return managementidentity.BootstrapResult{}, managementidentity.ErrBootstrapUnavailable
	}
	result.Replayed = true
	return result, nil
}

func lockBootstrapState(ctx context.Context, tx *sql.Tx) (bootstrapState, error) {
	var state bootstrapState
	err := tx.QueryRowContext(ctx, `SELECT bootstrap_consumed_at,bootstrap_principal_id::text,
 bootstrap_idempotency_hmac_version,bootstrap_idempotency_key_digest,
 bootstrap_request_digest,bootstrap_response_ciphertext,bootstrap_response_nonce,
 bootstrap_response_kek_version,bootstrap_response_status,
 bootstrap_result_expires_at,bootstrap_result_delivered_at,revision
FROM management_installation_state WHERE singleton=TRUE FOR UPDATE`).Scan(&state.consumedAt, &state.principalID, &state.hmacVersion, &state.keyDigest, &state.requestDigest, &state.ciphertext, &state.nonce, &state.kekVersion, &state.responseStatus, &state.expiresAt, &state.deliveredAt, &state.revision)
	if errors.Is(err, sql.ErrNoRows) {
		return bootstrapState{}, managementidentity.ErrBootstrapUnavailable
	}
	return state, err
}

func validateBootstrapSeed(ctx context.Context, tx *sql.Tx) error {
	var count int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM management_session_policy WHERE singleton=TRUE AND seed_version=1`).Scan(&count); err != nil || count != 1 {
		return managementidentity.ErrBootstrapUnavailable
	}
	return nil
}

func validateBootstrapRequest(request managementidentity.BootstrapRequest) error {
	if !canonicalBootstrapText(request.DisplayName, 1, 200) ||
		!visibleBootstrapASCII(request.IdempotencyKey, 16, 200) ||
		len(request.CanonicalRequest) == 0 || len(request.CanonicalRequest) > bootstrapMaxRequestBytes {
		return errors.New("invalid bootstrap request")
	}
	switch request.Kind {
	case managementidentity.BootstrapExternalPrincipal:
		if !canonicalUUID(request.IssuerID) || !canonicalHTTPSURL(request.Issuer) ||
			!canonicalBootstrapText(request.Subject, 1, 512) || !canonicalHTTPSURL(request.DiscoveryURL) ||
			!canonicalBootstrapText(request.Audience, 1, 512) {
			return errors.New("invalid external bootstrap")
		}
	case managementidentity.BootstrapServiceAccount:
		if request.IssuerID != "" || request.Issuer != "" || request.Subject != "" || request.DiscoveryURL != "" || request.Audience != "" {
			return errors.New("invalid service bootstrap")
		}
	default:
		return errors.New("invalid bootstrap kind")
	}
	return nil
}

func canonicalHTTPSURL(value string) bool {
	if !canonicalBootstrapText(value, 1, 2048) {
		return false
	}
	parsed, err := url.Parse(value)
	return err == nil && parsed.Scheme == "https" && parsed.Host != "" && parsed.User == nil &&
		parsed.Fragment == "" && parsed.RawQuery == ""
}

func canonicalBootstrapText(value string, minimum, maximum int) bool {
	if len(value) < minimum || len(value) > maximum || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}

func visibleBootstrapASCII(value string, minimum, maximum int) bool {
	if len(value) < minimum || len(value) > maximum {
		return false
	}
	for index := range value {
		if value[index] < '!' || value[index] > '~' {
			return false
		}
	}
	return true
}

func (service *BootstrapService) validToken(token string) bool {
	digest := sha256.Sum256([]byte(token))
	service.tokenMu.RLock()
	defer service.tokenMu.RUnlock()
	return service.tokenConfigured && subtle.ConstantTimeCompare(digest[:], service.tokenDigest[:]) == 1
}

// refreshTokenState permits exactly one transition: configured to finalized.
// Removing the deployment-owned file erases the in-memory verifier and the
// process will never trust a recreated file without an explicit restart.
func (service *BootstrapService) refreshTokenState() (bool, error) {
	if service == nil {
		return false, managementidentity.ErrBootstrapUnavailable
	}
	service.tokenMu.RLock()
	configured, probe := service.tokenConfigured, service.tokenPresent
	service.tokenMu.RUnlock()
	if !configured || probe == nil {
		return configured, nil
	}
	present, err := probe()
	if err != nil {
		return false, err
	}
	if present {
		return true, nil
	}
	service.tokenMu.Lock()
	if service.tokenConfigured {
		service.tokenDigest = [sha256.Size]byte{}
		service.tokenConfigured = false
	}
	service.tokenMu.Unlock()
	return false, nil
}

func (service *BootstrapService) bootstrapDigests(request managementidentity.BootstrapRequest) []bootstrapDigest {
	values := make([]bootstrapDigest, 0, len(service.idempotency.Keys))
	for version, key := range service.idempotency.Keys {
		values = append(values, bootstrapDigest{version: version, key: bootstrapHMAC(key, "key", []byte(request.IdempotencyKey)), request: bootstrapHMAC(key, "request", request.CanonicalRequest)})
	}
	return values
}

func bootstrapHMAC(key []byte, label string, value []byte) [sha256.Size]byte {
	digest := hmac.New(sha256.New, key)
	_, _ = digest.Write([]byte("management-bootstrap/v1\x00" + label + "\x00"))
	_, _ = digest.Write(value)
	var result [sha256.Size]byte
	copy(result[:], digest.Sum(nil))
	return result
}

func activeBootstrapDigest(version string, values []bootstrapDigest) bootstrapDigest {
	for _, value := range values {
		if value.version == version {
			return value
		}
	}
	return bootstrapDigest{}
}

func bootstrapAAD(principalID, bindingID string) []byte {
	return []byte("management-bootstrap/v1\x00" + principalID + "\x00" + bindingID)
}

func validateSymmetric(value securitykeyring.Symmetric) error {
	if value.ActiveVersion == "" || len(value.Keys) == 0 || len(value.Keys) > 8 {
		return errors.New("invalid keyring")
	}
	for version, key := range value.Keys {
		if strings.TrimSpace(version) != version || version == "" || len(key) != 32 {
			return errors.New("invalid keyring")
		}
	}
	if _, ok := value.Keys[value.ActiveVersion]; !ok {
		return errors.New("invalid keyring")
	}
	return nil
}

func cloneSymmetric(value securitykeyring.Symmetric) securitykeyring.Symmetric {
	keys := make(map[string][]byte, len(value.Keys))
	for version, key := range value.Keys {
		keys[version] = append([]byte(nil), key...)
	}
	return securitykeyring.Symmetric{ActiveVersion: value.ActiveVersion, Keys: keys}
}

func cloneKEKKeyring(value accesscredential.KEKKeyring) accesscredential.KEKKeyring {
	keys := make(map[string][]byte, len(value.Keys))
	for version, key := range value.Keys {
		keys[version] = append([]byte(nil), key...)
	}
	return accesscredential.KEKKeyring{ActiveVersion: value.ActiveVersion, Keys: keys}
}

func zeroBytes(value []byte) {
	for index := range value {
		value[index] = 0
	}
}
