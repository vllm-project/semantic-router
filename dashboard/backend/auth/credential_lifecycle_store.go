package auth

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	// #nosec G101 -- Credential lifecycle event name, not secret material.
	CredentialLifecycleAdminPasswordReset = "credential.admin_password_reset"
	legacyUserPasswordAuditAction         = "user.password"

	credentialLifecycleRequestRetention = 24 * time.Hour
)

var ErrCredentialLifecycleConflict = errors.New("credential lifecycle idempotency conflict")

type CredentialLifecycleMutation struct {
	Operation          string
	AuditAction        string
	ActorUserID        string
	TargetUserID       string
	PasswordHash       string
	IdempotencyKey     string
	RequestFingerprint string
	Method             string
	Path               string
	IP                 string
	UserAgent          string
	StatusCode         int
	CreatedAt          int64
}

type CredentialLifecycleResult struct {
	Replayed        bool `json:"replayed"`
	AuditLogID      int64
	RevokedSessions int64
}

type credentialLifecycleRequestRecord struct {
	targetUserID       string
	requestFingerprint string
	auditLogID         int64
}

func (s *Service) CredentialLifecycleFingerprint(operation, targetUserID, secretValue string) string {
	mac := hmac.New(sha256.New, s.jwtSecret)
	writeCredentialFingerprintPart(mac, operation)
	writeCredentialFingerprintPart(mac, targetUserID)
	writeCredentialFingerprintPart(mac, secretValue)
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}

func writeCredentialFingerprintPart(mac hashWriter, value string) {
	_, _ = mac.Write([]byte(value))
	_, _ = mac.Write([]byte{0})
}

type hashWriter interface {
	Write([]byte) (int, error)
}

func (s *Store) ResetUserPasswordWithAudit(
	ctx context.Context,
	mutation CredentialLifecycleMutation,
) (CredentialLifecycleResult, error) {
	mutation = normalizeCredentialLifecycleMutation(mutation)
	if mutation.TargetUserID == "" || mutation.PasswordHash == "" {
		return CredentialLifecycleResult{}, errors.New("target user and password hash are required")
	}
	if mutation.IdempotencyKey != "" && mutation.RequestFingerprint == "" {
		return CredentialLifecycleResult{}, errors.New("request fingerprint is required when idempotency key is provided")
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return CredentialLifecycleResult{}, err
	}
	defer func() {
		_ = tx.Rollback()
	}()

	if mutation.IdempotencyKey != "" {
		replayResult, replayed, replayErr := credentialLifecycleReplayResult(ctx, tx, mutation)
		if replayErr != nil || replayed {
			return replayResult, replayErr
		}
	}

	res, err := tx.ExecContext(
		ctx,
		`UPDATE users SET password_hash = ?, updated_at = ? WHERE id = ?`,
		mutation.PasswordHash,
		mutation.CreatedAt,
		mutation.TargetUserID,
	)
	if err != nil {
		return CredentialLifecycleResult{}, err
	}
	affected, _ := res.RowsAffected()
	if affected == 0 {
		return CredentialLifecycleResult{}, sql.ErrNoRows
	}

	revokeResult, err := tx.ExecContext(
		ctx,
		`UPDATE auth_sessions SET revoked_at = COALESCE(revoked_at, ?) WHERE user_id = ? AND revoked_at IS NULL`,
		mutation.CreatedAt,
		mutation.TargetUserID,
	)
	if err != nil {
		return CredentialLifecycleResult{}, err
	}
	revokedSessions, _ := revokeResult.RowsAffected()

	extraJSON, err := credentialLifecycleAuditExtraJSON(mutation, revokedSessions)
	if err != nil {
		return CredentialLifecycleResult{}, err
	}
	auditLogID, err := addAuditLog(ctx, tx, AuditLog{
		UserID:     mutation.ActorUserID,
		Action:     mutation.AuditAction,
		Resource:   mutation.Path,
		Method:     mutation.Method,
		Path:       mutation.Path,
		IP:         mutation.IP,
		UserAgent:  mutation.UserAgent,
		StatusCode: mutation.StatusCode,
		CreatedAt:  mutation.CreatedAt,
		ExtraJSON:  extraJSON,
	})
	if err != nil {
		return CredentialLifecycleResult{}, err
	}

	if mutation.IdempotencyKey != "" {
		if _, err := tx.ExecContext(
			ctx,
			`INSERT INTO credential_lifecycle_requests(
				operation, actor_user_id, target_user_id, idempotency_key, request_fingerprint, audit_log_id, created_at
			) VALUES (?, ?, ?, ?, ?, ?, ?)`,
			mutation.Operation,
			mutation.ActorUserID,
			mutation.TargetUserID,
			mutation.IdempotencyKey,
			mutation.RequestFingerprint,
			auditLogID,
			mutation.CreatedAt,
		); err != nil {
			return CredentialLifecycleResult{}, err
		}
	}

	if err := tx.Commit(); err != nil {
		return CredentialLifecycleResult{}, err
	}
	return CredentialLifecycleResult{AuditLogID: auditLogID, RevokedSessions: revokedSessions}, nil
}

func credentialLifecycleReplayResult(
	ctx context.Context,
	tx *sql.Tx,
	mutation CredentialLifecycleMutation,
) (CredentialLifecycleResult, bool, error) {
	record, lookupErr := lookupCredentialLifecycleRequest(ctx, tx, mutation)
	if lookupErr == nil {
		if record.targetUserID != mutation.TargetUserID || record.requestFingerprint != mutation.RequestFingerprint {
			return CredentialLifecycleResult{}, false, fmt.Errorf("%w: key was already used for a different request", ErrCredentialLifecycleConflict)
		}
		return CredentialLifecycleResult{Replayed: true, AuditLogID: record.auditLogID}, true, nil
	}
	if !errors.Is(lookupErr, sql.ErrNoRows) {
		return CredentialLifecycleResult{}, false, lookupErr
	}
	return CredentialLifecycleResult{}, false, nil
}

func (s *Store) PruneCredentialLifecycleRequests(ctx context.Context, now time.Time) error {
	if now.IsZero() {
		now = time.Now()
	}
	cutoff := now.Add(-credentialLifecycleRequestRetention).Unix()
	_, err := s.db.ExecContext(ctx, `DELETE FROM credential_lifecycle_requests WHERE created_at <= ?`, cutoff)
	return err
}

func normalizeCredentialLifecycleMutation(mutation CredentialLifecycleMutation) CredentialLifecycleMutation {
	mutation.Operation = strings.TrimSpace(mutation.Operation)
	if mutation.Operation == "" {
		mutation.Operation = CredentialLifecycleAdminPasswordReset
	}
	mutation.AuditAction = strings.TrimSpace(mutation.AuditAction)
	if mutation.AuditAction == "" {
		mutation.AuditAction = legacyUserPasswordAuditAction
	}
	mutation.ActorUserID = strings.TrimSpace(mutation.ActorUserID)
	mutation.TargetUserID = strings.TrimSpace(mutation.TargetUserID)
	mutation.PasswordHash = strings.TrimSpace(mutation.PasswordHash)
	mutation.IdempotencyKey = strings.TrimSpace(mutation.IdempotencyKey)
	mutation.RequestFingerprint = strings.TrimSpace(mutation.RequestFingerprint)
	if mutation.Path == "" {
		mutation.Path = "/api/admin/users/password"
	}
	if mutation.StatusCode == 0 {
		mutation.StatusCode = 200
	}
	if mutation.CreatedAt <= 0 {
		mutation.CreatedAt = nowUnix()
	}
	return mutation
}

func lookupCredentialLifecycleRequest(
	ctx context.Context,
	tx *sql.Tx,
	mutation CredentialLifecycleMutation,
) (credentialLifecycleRequestRecord, error) {
	var record credentialLifecycleRequestRecord
	err := tx.QueryRowContext(
		ctx,
		`SELECT target_user_id, request_fingerprint, audit_log_id
		FROM credential_lifecycle_requests
		WHERE operation = ? AND actor_user_id = ? AND idempotency_key = ?`,
		mutation.Operation,
		mutation.ActorUserID,
		mutation.IdempotencyKey,
	).Scan(&record.targetUserID, &record.requestFingerprint, &record.auditLogID)
	return record, err
}

func credentialLifecycleAuditExtraJSON(mutation CredentialLifecycleMutation, revokedSessions int64) (string, error) {
	payload := map[string]interface{}{
		"eventType":        mutation.Operation,
		"targetUserId":     mutation.TargetUserID,
		"outcome":          "success",
		"revokedSessions":  revokedSessions,
		"idempotent":       mutation.IdempotencyKey != "",
		"idempotencyKeyID": idempotencyKeyID(mutation.IdempotencyKey),
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	return string(encoded), nil
}

func idempotencyKeyID(key string) string {
	key = strings.TrimSpace(key)
	if key == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:8])
}
