// Package postgres provides transaction-scoped Management command
// idempotency. Callers retain ownership of the SQL transaction so the domain
// mutation and replay reference commit or roll back together.
package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

const (
	lockCommandQuery = `SELECT clock_timestamp()
FROM pg_advisory_xact_lock($1)`
	getCommandQuery = `SELECT request_digest, operation_id::text,
       resource_type, resource_id, resource_revision, desired_revision,
       response_status, secret_response_ciphertext, secret_response_nonce,
       response_kek_version, secret_response_expires_at, expires_at
FROM management_idempotency
WHERE scope_kind = $1 AND namespace_id IS NOT DISTINCT FROM $2
  AND principal_id = $3 AND endpoint = $4
  AND hmac_version = $5 AND idempotency_key_digest = $6`
	getLiveCommandQuery       = getCommandQuery + ` AND expires_at > clock_timestamp()`
	getLockedCommandQuery     = getCommandQuery + ` FOR UPDATE`
	deleteExpiredCommandQuery = `DELETE FROM management_idempotency
WHERE scope_kind = $1 AND namespace_id IS NOT DISTINCT FROM $2
  AND principal_id = $3 AND endpoint = $4
  AND hmac_version = $5 AND idempotency_key_digest = $6 AND expires_at <= $7`
	insertResourceCommandQuery = `INSERT INTO management_idempotency
  (scope_kind, namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest,
   request_digest, operation_id, resource_type, resource_id,
   resource_revision, desired_revision, response_status,
   expires_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, $8, $9, $10, NULL, $11, $12)`
	insertOperationCommandQuery = `INSERT INTO management_idempotency
  (scope_kind, namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest,
   request_digest, operation_id, resource_type, resource_id,
   resource_revision, desired_revision, response_status,
   expires_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NULL, NULL, NULL, $9, $10, $11)`
	insertSecretResourceCommandQuery = `INSERT INTO management_idempotency
  (scope_kind, namespace_id, principal_id, endpoint, hmac_version, idempotency_key_digest,
   request_digest, operation_id, resource_type, resource_id,
   resource_revision, desired_revision, response_status,
   secret_response_ciphertext, secret_response_nonce, response_kek_version,
   secret_response_expires_at, expires_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, NULL, $8, $9, $10, NULL, $11,
        $12, $13, $14, $15, $16)`
	getReferencedHMACVersionsQuery = `SELECT DISTINCT hmac_version
FROM management_idempotency
WHERE expires_at > clock_timestamp()
ORDER BY hmac_version`
)

// Querier is implemented by *sql.DB and *sql.Tx.
type Querier interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

// RowsQuerier is implemented by *sql.DB and *sql.Tx.
type RowsQuerier interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

// Lookup returns a committed, unexpired replay result. It is an optimization
// for callers that must replay before consulting mutable dependencies; Lock
// remains the authoritative cross-replica serialization point.
func Lookup(ctx context.Context, queryer Querier, command managementcommand.Command) (managementcommand.StoredResult, bool, error) {
	if queryer == nil {
		return managementcommand.StoredResult{}, false, errors.New("management command queryer is required")
	}
	if err := command.Validate(time.Now().UTC()); err != nil {
		return managementcommand.StoredResult{}, false, err
	}
	var found *managementcommand.StoredResult
	scopeKind, namespaceID := commandScopeArguments(command.Scope)
	for _, candidate := range command.CandidateDigests() {
		result, requestDigest, err := scanCommand(queryer.QueryRowContext(
			ctx, getLiveCommandQuery, scopeKind, namespaceID, command.PrincipalID,
			command.Endpoint, candidate.HMACVersion, candidate.KeyDigest[:],
		))
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return managementcommand.StoredResult{}, false, fmt.Errorf("lookup management command: %w", err)
		}
		if found != nil {
			return managementcommand.StoredResult{}, false, errors.New("management command exists under multiple HMAC versions")
		}
		if !command.SameRequest(candidate.HMACVersion, requestDigest) {
			return managementcommand.StoredResult{}, false, managementcommand.ErrConflict
		}
		found = &result
	}
	if found == nil {
		return managementcommand.StoredResult{}, false, nil
	}
	return *found, true, nil
}

// Lock serializes all replicas for this command identity. The transaction is
// the claim: a failed mutation releases the lock without persisting a key.
// A committed winner returns new=true; a retry receives its stored result.
func Lock(ctx context.Context, tx *sql.Tx, command managementcommand.Command) (result managementcommand.StoredResult, replayed bool, err error) {
	if tx == nil {
		return managementcommand.StoredResult{}, false, errors.New("management command transaction is required")
	}
	if err := command.Validate(time.Now().UTC()); err != nil {
		return managementcommand.StoredResult{}, false, err
	}
	var databaseNow time.Time
	if err := tx.QueryRowContext(ctx, lockCommandQuery, advisoryKey(command)).Scan(&databaseNow); err != nil {
		return managementcommand.StoredResult{}, false, fmt.Errorf("lock management command: %w", err)
	}
	if err := command.Validate(databaseNow.UTC()); err != nil {
		return managementcommand.StoredResult{}, false, err
	}
	var found *managementcommand.StoredResult
	scopeKind, namespaceID := commandScopeArguments(command.Scope)
	for _, candidate := range command.CandidateDigests() {
		result, requestDigest, err := scanCommand(tx.QueryRowContext(
			ctx, getLockedCommandQuery, scopeKind, namespaceID, command.PrincipalID,
			command.Endpoint, candidate.HMACVersion, candidate.KeyDigest[:],
		))
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return managementcommand.StoredResult{}, false, fmt.Errorf("read locked management command: %w", err)
		}
		if !databaseNow.Before(result.ExpiresAt) {
			if _, err := tx.ExecContext(ctx, deleteExpiredCommandQuery,
				scopeKind, namespaceID, command.PrincipalID, command.Endpoint,
				candidate.HMACVersion, candidate.KeyDigest[:], databaseNow); err != nil {
				return managementcommand.StoredResult{}, false, fmt.Errorf("delete expired management command: %w", err)
			}
			continue
		}
		if found != nil {
			return managementcommand.StoredResult{}, false, errors.New("management command exists under multiple HMAC versions")
		}
		if !command.SameRequest(candidate.HMACVersion, requestDigest) {
			return managementcommand.StoredResult{}, false, managementcommand.ErrConflict
		}
		found = &result
	}
	if found == nil {
		return managementcommand.StoredResult{}, false, nil
	}
	return *found, true, nil
}

func CompleteResource(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	result managementcommand.ResourceResult,
) error {
	if tx == nil {
		return errors.New("management command transaction is required")
	}
	if err := command.Validate(time.Now().UTC()); err != nil {
		return err
	}
	if err := result.Validate(); err != nil {
		return err
	}
	active := command.ActiveDigest()
	scopeKind, namespaceID := commandScopeArguments(command.Scope)
	if _, err := tx.ExecContext(ctx, insertResourceCommandQuery,
		scopeKind, namespaceID, command.PrincipalID, command.Endpoint,
		active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:], result.ResourceType,
		result.ResourceID, result.ResourceRevision, result.ResponseStatus, command.ExpiresAt,
	); err != nil {
		return fmt.Errorf("complete Management resource command: %w", err)
	}
	return nil
}

// CompleteSecretResource persists the mutation receipt and its encrypted
// canonical response in the same transaction. The plaintext never enters SQL,
// logs, audit fields, or the idempotency key digest.
func CompleteSecretResource(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	result managementcommand.ResourceResult,
	secret managementcommand.SecretResponse,
) error {
	if tx == nil {
		return errors.New("management command transaction is required")
	}
	if err := command.Validate(time.Now().UTC()); err != nil {
		return err
	}
	if err := result.Validate(); err != nil {
		return err
	}
	if err := secret.Validate(); err != nil || secret.ExpiresAt.After(command.ExpiresAt) {
		return errors.New("management command secret response is invalid")
	}
	active := command.ActiveDigest()
	scopeKind, namespaceID := commandScopeArguments(command.Scope)
	if _, err := tx.ExecContext(ctx, insertSecretResourceCommandQuery,
		scopeKind, namespaceID, command.PrincipalID, command.Endpoint,
		active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:], result.ResourceType,
		result.ResourceID, result.ResourceRevision, result.ResponseStatus,
		secret.Ciphertext, secret.Nonce, secret.KEKVersion, secret.ExpiresAt, command.ExpiresAt,
	); err != nil {
		return fmt.Errorf("complete Management secret resource command: %w", err)
	}
	return nil
}

func CompleteOperation(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	result managementcommand.OperationResult,
) error {
	if tx == nil {
		return errors.New("management command transaction is required")
	}
	if err := command.Validate(time.Now().UTC()); err != nil {
		return err
	}
	if err := result.Validate(); err != nil {
		return err
	}
	active := command.ActiveDigest()
	scopeKind, namespaceID := commandScopeArguments(command.Scope)
	if _, err := tx.ExecContext(ctx, insertOperationCommandQuery,
		scopeKind, namespaceID, command.PrincipalID, command.Endpoint,
		active.HMACVersion, active.KeyDigest[:], active.RequestDigest[:], result.OperationID,
		nullableRevision(result.DesiredRevision), result.ResponseStatus,
		command.ExpiresAt,
	); err != nil {
		return fmt.Errorf("complete Management operation command: %w", err)
	}
	return nil
}

func commandScopeArguments(scope managementcommand.CommandScope) (string, any) {
	if scope.Kind == managementcommand.ScopeNamespace {
		return string(scope.Kind), scope.NamespaceID
	}
	return string(scope.Kind), nil
}

// ValidateReferencedHMACVersions fails closed when any unexpired command row
// requires a key version that the current codec no longer retains. Production
// startup and readiness must call this before accepting Management traffic.
func ValidateReferencedHMACVersions(
	ctx context.Context,
	queryer RowsQuerier,
	codec *managementcommand.Codec,
) (returnErr error) {
	if queryer == nil || codec == nil {
		return errors.New("management command readiness dependencies are required")
	}
	rows, err := queryer.QueryContext(ctx, getReferencedHMACVersionsQuery)
	if err != nil {
		return fmt.Errorf("read management command HMAC versions: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			return fmt.Errorf("scan management command HMAC version: %w", err)
		}
		if !codec.RecognizesHMACVersion(version) {
			return fmt.Errorf("%w: %q is referenced by an unexpired command", managementcommand.ErrHMACVersionUnavailable, version)
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate management command HMAC versions: %w", err)
	}
	return nil
}

type rowScanner interface{ Scan(...any) error }

func scanCommand(scanner rowScanner) (managementcommand.StoredResult, []byte, error) {
	var (
		requestDigest                         []byte
		operationID, resourceType, resourceID sql.NullString
		resourceRevision, desiredRevision     sql.NullInt64
		secretCiphertext, secretNonce         []byte
		secretKEKVersion                      sql.NullString
		secretExpiresAt                       sql.NullTime
		responseStatus                        int
		expiresAt                             time.Time
	)
	if err := scanner.Scan(
		&requestDigest, &operationID, &resourceType, &resourceID,
		&resourceRevision, &desiredRevision, &responseStatus,
		&secretCiphertext, &secretNonce, &secretKEKVersion, &secretExpiresAt, &expiresAt,
	); err != nil {
		return managementcommand.StoredResult{}, nil, err
	}
	if len(requestDigest) != sha256.Size {
		return managementcommand.StoredResult{}, nil, errors.New("stored management command request digest is invalid")
	}
	result := managementcommand.StoredResult{ExpiresAt: expiresAt.UTC()}
	switch {
	case operationID.Valid && !resourceType.Valid && !resourceID.Valid && !resourceRevision.Valid:
		result.Operation = &managementcommand.OperationResult{
			OperationID: operationID.String, ResponseStatus: responseStatus,
		}
		if desiredRevision.Valid {
			if desiredRevision.Int64 <= 0 {
				return managementcommand.StoredResult{}, nil, errors.New("stored management command desired revision is invalid")
			}
			value := uint64(desiredRevision.Int64)
			result.Operation.DesiredRevision = &value
		}
	case !operationID.Valid && resourceType.Valid && resourceID.Valid && resourceRevision.Valid && !desiredRevision.Valid:
		if resourceRevision.Int64 <= 0 {
			return managementcommand.StoredResult{}, nil, errors.New("stored management command revisions are invalid")
		}
		result.Resource = &managementcommand.ResourceResult{
			ResourceType: resourceType.String, ResourceID: resourceID.String,
			ResourceRevision: uint64(resourceRevision.Int64),
			ResponseStatus:   responseStatus,
		}
	default:
		return managementcommand.StoredResult{}, nil, errors.New("stored management command result kind is invalid")
	}
	secretFields := 0
	if len(secretCiphertext) > 0 {
		secretFields++
	}
	if len(secretNonce) > 0 {
		secretFields++
	}
	if secretKEKVersion.Valid {
		secretFields++
	}
	if secretExpiresAt.Valid {
		secretFields++
	}
	switch secretFields {
	case 0:
	case 4:
		result.Secret = &managementcommand.SecretResponse{
			Ciphertext: append([]byte(nil), secretCiphertext...), Nonce: append([]byte(nil), secretNonce...),
			KEKVersion: secretKEKVersion.String, ExpiresAt: secretExpiresAt.Time.UTC(),
		}
	default:
		return managementcommand.StoredResult{}, nil, errors.New("stored management command secret response is incomplete")
	}
	if err := result.Validate(); err != nil {
		return managementcommand.StoredResult{}, nil, err
	}
	return result, append([]byte(nil), requestDigest...), nil
}

func nullableRevision(value *uint64) any {
	if value == nil {
		return nil
	}
	return *value
}

func advisoryKey(command managementcommand.Command) int64 {
	return command.AdvisoryLockKey()
}
