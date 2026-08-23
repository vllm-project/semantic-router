package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func (store *Store) ApplyBackchannelLogout(
	ctx context.Context,
	request managementidentity.BackchannelLogout,
) (managementidentity.BackchannelLogoutResult, error) {
	identity := request.Identity
	if !canonicalUUID(identity.IssuerID) || identity.TokenID == "" ||
		(identity.Subject == "" && identity.IssuerSessionID == "") ||
		identity.IssuedAt.IsZero() || identity.ExpiresAt.IsZero() ||
		!identity.ExpiresAt.After(identity.IssuedAt) || request.RequestID == "" {
		return managementidentity.BackchannelLogoutResult{}, managementidentity.ErrInvalidLifecycleRequest
	}
	tokenDigest := sha256.Sum256([]byte("vllm-sr/backchannel-logout/v1\x00" + identity.IssuerID + "\x00" + identity.TokenID))
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.BackchannelLogoutResult, error) {
		var issuer, status string
		var revision uint64
		if err := tx.QueryRowContext(ctx, `SELECT issuer,status,revision
FROM trusted_identity_issuers WHERE id=$1 FOR SHARE`, identity.IssuerID).Scan(&issuer, &status, &revision); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.BackchannelLogoutResult{}, managementidentity.ErrNotFound
			}
			return managementidentity.BackchannelLogoutResult{}, err
		}
		if status != "active" {
			return managementidentity.BackchannelLogoutResult{}, managementidentity.ErrNotFound
		}
		if _, err := tx.ExecContext(ctx, `WITH expired AS (
  SELECT issuer_id,token_id_digest FROM management_backchannel_logout_replays
  WHERE expires_at<=clock_timestamp()-interval '2 minutes' ORDER BY expires_at LIMIT 1000
)
DELETE FROM management_backchannel_logout_replays replay
USING expired
WHERE replay.issuer_id=expired.issuer_id AND replay.token_id_digest=expired.token_id_digest`); err != nil {
			return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("expire Management back-channel logout replays: %w", err)
		}
		inserted, applyBackchannelLogoutErr := tx.ExecContext(ctx, `INSERT INTO management_backchannel_logout_replays
  (issuer_id,token_id_digest,claims_digest,expires_at)
VALUES ($1,$2,$3,$4)
ON CONFLICT (issuer_id,token_id_digest) DO NOTHING`, identity.IssuerID, tokenDigest[:], identity.ClaimsDigest[:], identity.ExpiresAt.UTC())
		if applyBackchannelLogoutErr != nil {
			return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("record Management back-channel logout replay: %w", applyBackchannelLogoutErr)
		}
		count, applyBackchannelLogoutErr := inserted.RowsAffected()
		if applyBackchannelLogoutErr != nil || count < 0 || count > 1 {
			return managementidentity.BackchannelLogoutResult{}, errors.New("management back-channel logout replay state is invalid")
		}
		replayed := count == 0
		if replayed {
			var stored []byte
			if err := tx.QueryRowContext(ctx, `SELECT claims_digest
FROM management_backchannel_logout_replays
WHERE issuer_id=$1 AND token_id_digest=$2 FOR UPDATE`, identity.IssuerID, tokenDigest[:]).Scan(&stored); err != nil {
				return managementidentity.BackchannelLogoutResult{}, managementidentity.ErrBackchannelReplay
			}
			if !slices.Equal(stored, identity.ClaimsDigest[:]) {
				return managementidentity.BackchannelLogoutResult{}, managementidentity.ErrBackchannelReplay
			}
		}

		selector, args := backchannelLogoutSelector(identity.IssuerID, issuer, identity.IssuerSessionID, identity.Subject)
		if !replayed {
			if _, err := tx.ExecContext(ctx, `UPDATE management_sessions SET status='expired'
WHERE `+selector+` AND status='active' AND expires_at<=clock_timestamp()`, args...); err != nil {
				return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("expire Management sessions for back-channel logout: %w", err)
			}
			if _, err := tx.ExecContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE `+selector+` AND status='active'`, args...); err != nil {
				return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("apply Management back-channel logout: %w", err)
			}
			if err := appendAudit(ctx, tx, auditMutation{
				Action: "management_session.backchannel_logout", ResourceType: "trusted_identity_issuer",
				ResourceID: identity.IssuerID, AfterRevision: revision, ExternalActor: true,
				Actor: managementidentity.MutationActor{
					RequestID: request.RequestID,
					Reason:    "Trusted issuer back-channel logout",
				},
			}); err != nil {
				return managementidentity.BackchannelLogoutResult{}, err
			}
		}
		rows, applyBackchannelLogoutErr := tx.QueryContext(ctx, `SELECT id::text FROM management_sessions
WHERE `+selector+` AND status='revoked' AND expires_at>clock_timestamp()
ORDER BY id`, args...)
		if applyBackchannelLogoutErr != nil {
			return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("list back-channel revoked Management sessions: %w", applyBackchannelLogoutErr)
		}
		defer rows.Close()
		ids := make([]string, 0)
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				return managementidentity.BackchannelLogoutResult{}, err
			}
			if !canonicalUUID(id) {
				return managementidentity.BackchannelLogoutResult{}, errors.New("stored Management session identifier is invalid")
			}
			ids = append(ids, id)
		}
		if err := rows.Err(); err != nil {
			return managementidentity.BackchannelLogoutResult{}, err
		}
		return managementidentity.BackchannelLogoutResult{SessionIDs: ids, Replayed: replayed}, nil
	})
}

func backchannelLogoutSelector(issuerID, issuer, issuerSessionID, subject string) (string, []any) {
	if issuerSessionID != "" {
		return `auth_source_kind='issuer' AND auth_source_id=$1 AND issuer_session_id=$2`,
			[]any{issuerID, issuerSessionID}
	}
	return `auth_source_kind='issuer' AND auth_source_id=$1 AND principal_id IN
  (SELECT id FROM management_principals WHERE issuer=$2 AND subject=$3)`,
		[]any{issuerID, issuer, subject}
}
