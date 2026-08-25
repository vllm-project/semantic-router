package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	// #nosec G101 -- this is a database column list and contains no credential value.
	credentialColumns = `id, namespace_id, api_key_id, kid, secret_hmac, pepper_version,
       secret_ciphertext, ciphertext_nonce, kek_version, status,
       not_before, expires_at, revoked_at, created_at`
	listCredentialsQuery = `SELECT ` + credentialColumns + `
FROM access_api_key_credentials
	WHERE namespace_id = $1 AND api_key_id = $2
	ORDER BY created_at DESC, id`
	// #nosec G101 -- this is a parameterized insert statement and contains no credential value.
	insertCredentialQuery = `INSERT INTO access_api_key_credentials
  (id, namespace_id, api_key_id, kid, secret_hmac, pepper_version,
   secret_ciphertext, ciphertext_nonce, kek_version, status,
   not_before, expires_at, revoked_at, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)`
	retireCredentialQuery = `UPDATE access_api_key_credentials
SET status = 'retiring', expires_at = $4
WHERE namespace_id = $1 AND api_key_id = $2 AND id = $3 AND status = 'active'`
	revokeCredentialQuery = `UPDATE access_api_key_credentials
SET status = 'revoked', revoked_at = clock_timestamp(),
    secret_ciphertext = NULL, ciphertext_nonce = NULL, kek_version = NULL
WHERE namespace_id = $1 AND api_key_id = $2 AND id = $3
  AND status IN ('active', 'retiring')`
)

func (s *Store) ListCredentialVersions(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
) ([]CredentialRecord, error) {
	if err := validateIdentityIDs(namespaceID, string(keyID)); err != nil {
		return nil, err
	}
	rows, err := s.db.QueryContext(ctx, listCredentialsQuery, namespaceID, keyID)
	if err != nil {
		return nil, fmt.Errorf("list credential versions: %w", err)
	}
	defer rows.Close()
	records := make([]CredentialRecord, 0)
	for rows.Next() {
		record, err := scanCredential(rows)
		if err != nil {
			return nil, fmt.Errorf("scan credential version: %w", err)
		}
		records = append(records, record)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate credential versions: %w", err)
	}
	return records, nil
}

func (s *Store) RotateCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
	expected accesscontrol.Revision,
	rotation CredentialRotation,
	meta MutationMeta,
) (MutationResult[accesscontrol.APIKey], error) {
	if err := validateCredentialRotation(namespaceID, keyID, rotation); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.APIKey], error) {
		updatedKey, rotateCredentialErr := advanceAPIKeyRevision(ctx, tx, namespaceID, keyID, expectedRevision)
		if rotateCredentialErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, rotateCredentialErr
		}
		if err := retirePreviousCredential(ctx, tx, namespaceID, keyID, rotation); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, err
		}
		if err := insertCredential(ctx, tx, namespaceID, rotation.Credential); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, err
		}
		references := map[string]string{"credentialId": string(rotation.Credential.ID)}
		if rotation.RetireCredentialID != nil {
			references["retiredCredentialId"] = string(*rotation.RetireCredentialID)
		}
		receipt, rotateCredentialErr := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(keyID),
			AggregateRevision: updatedKey.Revision, Operation: outboxCredentialRotated,
			References: references,
		}, meta)
		if rotateCredentialErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, rotateCredentialErr
		}
		return MutationResult[accesscontrol.APIKey]{Value: updatedKey, Receipt: receipt}, nil
	})
}

func (s *Store) RevokeCredential(
	ctx context.Context,
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
	credentialID accesscontrol.CredentialVersionID,
	expected accesscontrol.Revision,
	meta MutationMeta,
) (MutationResult[accesscontrol.APIKey], error) {
	if err := validateIdentityIDs(namespaceID, string(keyID)); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateUUID("credential id", string(credentialID)); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	expectedRevision, err := revisionAsInt64(expected)
	if err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationResult[accesscontrol.APIKey]{}, err
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (MutationResult[accesscontrol.APIKey], error) {
		updatedKey, revokeCredentialErr := advanceAPIKeyRevision(ctx, tx, namespaceID, keyID, expectedRevision)
		if revokeCredentialErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, revokeCredentialErr
		}
		result, revokeCredentialErr := tx.ExecContext(ctx, revokeCredentialQuery, namespaceID, keyID, credentialID)
		if revokeCredentialErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, fmt.Errorf("revoke credential version: %w", revokeCredentialErr)
		}
		if err := requireOneRow(result, ErrNotFound); err != nil {
			return MutationResult[accesscontrol.APIKey]{}, err
		}
		receipt, revokeCredentialErr := appendMutationRecords(ctx, tx, namespaceID, outboxMutation{
			AggregateType: "api_key", AggregateID: string(keyID),
			AggregateRevision: updatedKey.Revision, Operation: outboxCredentialRevoked,
			References: map[string]string{"credentialId": string(credentialID)},
		}, meta)
		if revokeCredentialErr != nil {
			return MutationResult[accesscontrol.APIKey]{}, revokeCredentialErr
		}
		return MutationResult[accesscontrol.APIKey]{Value: updatedKey, Receipt: receipt}, nil
	})
}

func insertCredential(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	credential accesscontrol.CredentialVersion,
) error {
	if err := validateCredentialForWrite(namespaceID, credential); err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, insertCredentialQuery,
		credential.ID, namespaceID, credential.APIKeyID, credential.KID,
		credential.SecretHMAC, credential.PepperVersion,
		nullableBytes(credential.SecretCiphertext), nullableBytes(credential.CiphertextNonce),
		nullableString(credential.KEKVersion), credential.Status,
		credential.NotBefore, credential.ExpiresAt, credential.RevokedAt, credential.CreatedAt); err != nil {
		return fmt.Errorf("insert credential version: %w", err)
	}
	return nil
}

func advanceAPIKeyRevision(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
	expectedRevision int64,
) (accesscontrol.APIKey, error) {
	key, err := scanAPIKey(tx.QueryRowContext(ctx, advanceAPIKeyRevisionQuery, namespaceID, keyID, expectedRevision))
	if errors.Is(err, sql.ErrNoRows) {
		return accesscontrol.APIKey{}, ErrRevisionConflict
	}
	if err != nil {
		return accesscontrol.APIKey{}, fmt.Errorf("advance API-key revision: %w", err)
	}
	return key, nil
}

func retirePreviousCredential(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
	rotation CredentialRotation,
) error {
	if rotation.RetireCredentialID == nil {
		return nil
	}
	result, err := tx.ExecContext(ctx, retireCredentialQuery,
		namespaceID, keyID, *rotation.RetireCredentialID, *rotation.RetireAt)
	if err != nil {
		return fmt.Errorf("retire previous credential: %w", err)
	}
	return requireOneRow(result, ErrNotFound)
}

func validateCredentialRotation(
	namespaceID accesscontrol.NamespaceID,
	keyID accesscontrol.APIKeyID,
	rotation CredentialRotation,
) error {
	if err := validateIdentityIDs(namespaceID, string(keyID)); err != nil {
		return err
	}
	if rotation.Credential.APIKeyID != keyID || rotation.Credential.Status != accesscontrol.CredentialStatusActive {
		return fmt.Errorf("rotated credential must be active and belong to the logical key")
	}
	if err := validateCredentialForWrite(namespaceID, rotation.Credential); err != nil {
		return err
	}
	if (rotation.RetireCredentialID == nil) != (rotation.RetireAt == nil) {
		return fmt.Errorf("retiring credential id and overlap expiry must be supplied together")
	}
	if rotation.RetireCredentialID != nil {
		if err := validateUUID("retiring credential id", string(*rotation.RetireCredentialID)); err != nil {
			return err
		}
		if !rotation.RetireAt.After(rotation.Credential.NotBefore) {
			return fmt.Errorf("credential overlap expiry must follow the new credential not-before time")
		}
	}
	return nil
}

func validateCredentialForWrite(namespaceID accesscontrol.NamespaceID, credential accesscontrol.CredentialVersion) error {
	if err := credential.Validate(); err != nil {
		return err
	}
	if credential.Status == accesscontrol.CredentialStatusExpired {
		return fmt.Errorf("expired status is derived from expires_at and cannot be persisted")
	}
	if err := validateIdentityIDs(namespaceID, string(credential.APIKeyID)); err != nil {
		return err
	}
	return validateUUID("credential id", string(credential.ID))
}

func scanCredential(scanner rowScanner) (CredentialRecord, error) {
	var record CredentialRecord
	var ciphertext, nonce []byte
	var kekVersion sql.NullString
	var expiresAt, revokedAt sql.NullTime
	if err := scanner.Scan(
		&record.Credential.ID, &record.NamespaceID, &record.Credential.APIKeyID,
		&record.Credential.KID, &record.Credential.SecretHMAC, &record.Credential.PepperVersion,
		&ciphertext, &nonce, &kekVersion, &record.Credential.Status,
		&record.Credential.NotBefore, &expiresAt, &revokedAt, &record.Credential.CreatedAt,
	); err != nil {
		return CredentialRecord{}, err
	}
	record.Credential.SecretCiphertext = ciphertext
	record.Credential.CiphertextNonce = nonce
	if kekVersion.Valid {
		record.Credential.KEKVersion = kekVersion.String
	}
	record.Credential.ExpiresAt = nullTimePointer(expiresAt)
	record.Credential.RevokedAt = nullTimePointer(revokedAt)
	return record, nil
}

func requireOneRow(result sql.Result, emptyError error) error {
	rows, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("read affected rows: %w", err)
	}
	if rows != 1 {
		return emptyError
	}
	return nil
}

func nullableString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func nullableBytes(value []byte) any {
	if len(value) == 0 {
		return nil
	}
	return value
}
