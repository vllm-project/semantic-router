package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const recoveryResultTTL = 15 * time.Minute

type RecoveryOptions struct {
	Database        *sql.DB
	RecoveryToken   []byte
	IdempotencyKeys securitykeyring.Symmetric
	Now             func() time.Time
}

// RecoveryService is a deliberately narrow break-glass authority. It can
// restore the built-in cluster-administrator binding for an existing durable
// principal, but cannot create principals or issue credentials.
type RecoveryService struct {
	database    *sql.DB
	tokenDigest [sha256.Size]byte
	idempotency securitykeyring.Symmetric
	now         func() time.Time
	closeOnce   sync.Once
}

type recoveryState struct {
	consumedAt      sql.NullTime
	tokenDigest     []byte
	hmacVersion     sql.NullString
	keyDigest       []byte
	requestDigest   []byte
	principalID     sql.NullString
	bindingID       sql.NullString
	receipt         []byte
	resultExpiresAt sql.NullTime
}

type recoveryReceipt struct {
	PrincipalID   string `json:"principalId"`
	RoleBindingID string `json:"roleBindingId"`
}

func NewRecoveryService(options RecoveryOptions) (*RecoveryService, error) {
	if options.Database == nil || len(options.RecoveryToken) < 32 || validateSymmetric(options.IdempotencyKeys) != nil {
		return nil, errors.New("management recovery dependencies are invalid")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &RecoveryService{
		database: options.Database, tokenDigest: sha256.Sum256(options.RecoveryToken),
		idempotency: cloneSymmetric(options.IdempotencyKeys), now: now,
	}, nil
}

func (service *RecoveryService) Ready(ctx context.Context) error {
	if service == nil || service.database == nil {
		return managementidentity.ErrRecoveryUnavailable
	}
	var tokenDigest []byte
	var hmacVersion sql.NullString
	var resultExpiresAt sql.NullTime
	if err := service.database.QueryRowContext(ctx, `SELECT recovery_token_digest,
       recovery_idempotency_hmac_version,recovery_result_expires_at
FROM management_installation_state WHERE singleton=TRUE`).Scan(&tokenDigest, &hmacVersion, &resultExpiresAt); err != nil {
		return fmt.Errorf("load Management recovery readiness: %w", err)
	}
	if hmacVersion.Valid && resultExpiresAt.Valid && service.now().UTC().Before(resultExpiresAt.Time) {
		if _, found := service.idempotency.Keys[hmacVersion.String]; !found {
			return errors.New("management recovery references an unavailable idempotency HMAC version")
		}
	}
	if len(tokenDigest) == sha256.Size && subtle.ConstantTimeCompare(tokenDigest, service.tokenDigest[:]) == 1 {
		return errors.New("management recovery credential was consumed; restart with recovery disabled")
	}
	return nil
}

func (service *RecoveryService) Close() {
	if service == nil {
		return
	}
	service.closeOnce.Do(func() {
		for _, key := range service.idempotency.Keys {
			zeroBytes(key)
		}
		service.idempotency = securitykeyring.Symmetric{}
		service.tokenDigest = [sha256.Size]byte{}
	})
}

func (service *RecoveryService) Recover(
	ctx context.Context,
	request managementidentity.RecoveryRequest,
	presentedToken string,
) (managementidentity.RecoveryResult, error) {
	if validateRecoveryRequest(request) != nil {
		return managementidentity.RecoveryResult{}, managementidentity.ErrInvalidRecoveryRequest
	}
	if service == nil || !service.validRecoveryToken(presentedToken) {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryUnavailable
	}
	now := service.now().UTC()
	digests := service.recoveryDigests(request)
	var (
		result managementidentity.RecoveryResult
		err    error
	)
	for attempt := 0; attempt < 5; attempt++ {
		result, err = service.recoveryTransaction(ctx, request, digests, now)
		if !isBootstrapRetryable(err) {
			break
		}
	}
	return result, err
}

func (service *RecoveryService) recoveryTransaction(
	ctx context.Context,
	request managementidentity.RecoveryRequest,
	digests []bootstrapDigest,
	now time.Time,
) (managementidentity.RecoveryResult, error) {
	return inTransaction(ctx, &Store{database: service.database}, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.RecoveryResult, error) {
		if _, err := tx.ExecContext(ctx, `SET LOCAL synchronous_commit = on`); err != nil {
			return managementidentity.RecoveryResult{}, fmt.Errorf("require synchronous Management recovery commit: %w", err)
		}
		state, err := lockRecoveryState(ctx, tx)
		if err != nil {
			return managementidentity.RecoveryResult{}, err
		}
		if len(state.tokenDigest) == sha256.Size && subtle.ConstantTimeCompare(state.tokenDigest, service.tokenDigest[:]) == 1 {
			return replayRecovery(request, state, digests, now)
		}
		if err := validateBootstrapSeed(ctx, tx); err != nil {
			return managementidentity.RecoveryResult{}, err
		}
		return service.commitRecovery(ctx, tx, request, digests, now)
	})
}

func (service *RecoveryService) commitRecovery(
	ctx context.Context,
	tx *sql.Tx,
	request managementidentity.RecoveryRequest,
	digests []bootstrapDigest,
	now time.Time,
) (managementidentity.RecoveryResult, error) {
	var principalStatus string
	var principalRevision int64
	if err := tx.QueryRowContext(ctx, `SELECT status,revision FROM management_principals
WHERE id=$1 FOR UPDATE`, request.PrincipalID).Scan(&principalStatus, &principalRevision); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.RecoveryResult{}, managementidentity.ErrNotFound
		}
		return managementidentity.RecoveryResult{}, err
	}
	if principalStatus != string(accesscontrol.PrincipalStatusActive) {
		if principalStatus != string(accesscontrol.PrincipalStatusDisabled) {
			return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryUnavailable
		}
		if _, err := tx.ExecContext(ctx, `UPDATE management_principals
SET status='active',revision=revision+1,updated_at=$2 WHERE id=$1`, request.PrincipalID, now); err != nil {
			return managementidentity.RecoveryResult{}, mapWriteError("restore recovery principal", err)
		}
		before := uint64(principalRevision)
		principalRevision++
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "management.principal.recover", ResourceType: "management_principal", ResourceID: request.PrincipalID,
			BeforeRevision: &before, AfterRevision: uint64(principalRevision), ExternalActor: true,
			Actor: managementidentity.MutationActor{RequestID: request.RequestID, Reason: request.Reason},
		}); err != nil {
			return managementidentity.RecoveryResult{}, err
		}
	}

	ceiling, _ := json.Marshal(accesscontrol.DelegablePermissions().Permissions())
	var bindingID string
	var bindingRevision int64
	var bindingCount int
	if err := tx.QueryRowContext(ctx, `SELECT count(*)
FROM management_role_bindings binding
JOIN management_roles role ON role.id=binding.role_id
WHERE binding.principal_id=$1 AND role.name='cluster_admin' AND role.builtin=TRUE
  AND binding.scope_kind='cluster'`, request.PrincipalID).Scan(&bindingCount); err != nil {
		return managementidentity.RecoveryResult{}, err
	}
	if bindingCount > 1 {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryUnavailable
	}
	commitRecoveryErr := tx.QueryRowContext(ctx, `SELECT binding.id::text,binding.revision
FROM management_role_bindings binding
JOIN management_roles role ON role.id=binding.role_id
WHERE binding.principal_id=$1 AND role.name='cluster_admin' AND role.builtin=TRUE
  AND binding.scope_kind='cluster'
ORDER BY binding.id LIMIT 1 FOR UPDATE OF binding`, request.PrincipalID).Scan(&bindingID, &bindingRevision)
	switch {
	case errors.Is(commitRecoveryErr, sql.ErrNoRows):
		bindingID, bindingRevision = uuid.NewString(), 1
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,delegation_ceiling,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,'cluster',$4,'active',1,$5,$5)`, bindingID, request.PrincipalID,
			builtInRoleID(accesscontrol.BuiltInRoleClusterAdmin), ceiling, now); err != nil {
			return managementidentity.RecoveryResult{}, mapWriteError("create recovery cluster administrator binding", err)
		}
	case commitRecoveryErr != nil:
		return managementidentity.RecoveryResult{}, commitRecoveryErr
	default:
		bindingRevision++
		if _, err := tx.ExecContext(ctx, `UPDATE management_role_bindings
SET status='active',delegation_ceiling=$2,revision=$3,updated_at=$4 WHERE id=$1`,
			bindingID, ceiling, bindingRevision, now); err != nil {
			return managementidentity.RecoveryResult{}, mapWriteError("restore recovery cluster administrator binding", err)
		}
	}

	receipt, _ := json.Marshal(recoveryReceipt{PrincipalID: request.PrincipalID, RoleBindingID: bindingID})
	active := activeBootstrapDigest(service.idempotency.ActiveVersion, digests)
	updated, commitRecoveryErr := tx.ExecContext(ctx, `UPDATE management_installation_state SET
  recovery_consumed_at=$1,recovery_token_digest=$2,
  recovery_idempotency_hmac_version=$3,recovery_nonce_hmac=$4,
  recovery_request_digest=$5,recovery_principal_id=$6,recovery_binding_id=$7,
  recovery_receipt=$8,recovery_result_expires_at=$9,
  revision=revision+1,updated_at=$1
WHERE singleton=TRUE`, now, service.tokenDigest[:], active.version, active.key[:], active.request[:],
		request.PrincipalID, bindingID, receipt, now.Add(recoveryResultTTL))
	if commitRecoveryErr != nil {
		return managementidentity.RecoveryResult{}, fmt.Errorf("consume Management recovery credential: %w", commitRecoveryErr)
	}
	if count, err := updated.RowsAffected(); err != nil || count != 1 {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryUnavailable
	}
	if err := appendAudit(ctx, tx, auditMutation{
		Action: "management.recovery", ResourceType: "management_role_binding", ResourceID: bindingID,
		AfterRevision: uint64(bindingRevision), ExternalActor: true,
		Actor: managementidentity.MutationActor{RequestID: request.RequestID, Reason: request.Reason},
	}); err != nil {
		return managementidentity.RecoveryResult{}, err
	}
	return managementidentity.RecoveryResult{
		PrincipalID: request.PrincipalID, RoleBindingID: bindingID, ResponseStatus: 201,
	}, nil
}

func replayRecovery(
	request managementidentity.RecoveryRequest,
	state recoveryState,
	digests []bootstrapDigest,
	now time.Time,
) (managementidentity.RecoveryResult, error) {
	matchedKey := false
	for _, candidate := range digests {
		if candidate.version != state.hmacVersion.String || len(state.keyDigest) != sha256.Size ||
			subtle.ConstantTimeCompare(candidate.key[:], state.keyDigest) != 1 {
			continue
		}
		matchedKey = true
		if len(state.requestDigest) != sha256.Size || subtle.ConstantTimeCompare(candidate.request[:], state.requestDigest) != 1 {
			return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryConflict
		}
	}
	if !matchedKey {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryConsumed
	}
	if !state.resultExpiresAt.Valid || !now.Before(state.resultExpiresAt.Time) {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryConsumed
	}
	var receipt recoveryReceipt
	if json.Unmarshal(state.receipt, &receipt) != nil || receipt.PrincipalID != request.PrincipalID ||
		receipt.PrincipalID != state.principalID.String || receipt.RoleBindingID != state.bindingID.String {
		return managementidentity.RecoveryResult{}, managementidentity.ErrRecoveryUnavailable
	}
	return managementidentity.RecoveryResult{
		PrincipalID: receipt.PrincipalID, RoleBindingID: receipt.RoleBindingID,
		Replayed: true, ResponseStatus: 201,
	}, nil
}

func lockRecoveryState(ctx context.Context, tx *sql.Tx) (recoveryState, error) {
	var state recoveryState
	err := tx.QueryRowContext(ctx, `SELECT recovery_consumed_at,recovery_token_digest,
 recovery_idempotency_hmac_version,recovery_nonce_hmac,recovery_request_digest,
 recovery_principal_id::text,recovery_binding_id::text,recovery_receipt,
 recovery_result_expires_at
FROM management_installation_state WHERE singleton=TRUE FOR UPDATE`).Scan(
		&state.consumedAt, &state.tokenDigest, &state.hmacVersion, &state.keyDigest,
		&state.requestDigest, &state.principalID, &state.bindingID, &state.receipt,
		&state.resultExpiresAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return recoveryState{}, managementidentity.ErrRecoveryUnavailable
	}
	return state, err
}

func validateRecoveryRequest(request managementidentity.RecoveryRequest) error {
	if !canonicalUUID(request.PrincipalID) || !canonicalBootstrapText(request.Reason, 1, 500) ||
		!canonicalBootstrapText(request.RequestID, 1, 200) || !visibleBootstrapASCII(request.IdempotencyKey, 16, 200) ||
		len(request.CanonicalRequest) == 0 || len(request.CanonicalRequest) > bootstrapMaxRequestBytes {
		return errors.New("invalid recovery request")
	}
	return nil
}

func (service *RecoveryService) validRecoveryToken(token string) bool {
	digest := sha256.Sum256([]byte(token))
	return service != nil && subtle.ConstantTimeCompare(digest[:], service.tokenDigest[:]) == 1
}

func (service *RecoveryService) recoveryDigests(request managementidentity.RecoveryRequest) []bootstrapDigest {
	values := make([]bootstrapDigest, 0, len(service.idempotency.Keys))
	for version, key := range service.idempotency.Keys {
		values = append(values, bootstrapDigest{
			version: version,
			key:     bootstrapHMAC(key, "recovery-key", []byte(request.IdempotencyKey)),
			request: bootstrapHMAC(key, "recovery-request", request.CanonicalRequest),
		})
	}
	return values
}
