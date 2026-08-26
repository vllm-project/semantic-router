package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

func (store *Store) ApplyBackchannelLogout(
	ctx context.Context,
	request managementidentity.BackchannelLogout,
) (managementidentity.BackchannelLogoutResult, error) {
	identity, tokenDigest, err := validatedBackchannelLogout(request)
	if err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		result, err := store.applyBackchannelLogoutOnce(ctx, request, identity, tokenDigest)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if !retryableLogoutTransactionError(err) || ctx.Err() != nil {
			return managementidentity.BackchannelLogoutResult{}, err
		}
	}
	return managementidentity.BackchannelLogoutResult{},
		fmt.Errorf("apply Management back-channel logout after transaction retries: %w", lastErr)
}

func (store *Store) applyBackchannelLogoutOnce(
	ctx context.Context,
	request managementidentity.BackchannelLogout,
	identity managementauth.BackchannelLogoutIdentity,
	tokenDigest [sha256.Size]byte,
) (managementidentity.BackchannelLogoutResult, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.BackchannelLogoutResult, error) {
		return applyBackchannelLogoutTransaction(ctx, tx, request, identity, tokenDigest)
	})
}

func applyBackchannelLogoutTransaction(
	ctx context.Context,
	tx *sql.Tx,
	request managementidentity.BackchannelLogout,
	identity managementauth.BackchannelLogoutIdentity,
	tokenDigest [sha256.Size]byte,
) (managementidentity.BackchannelLogoutResult, error) {
	issuer, revision, err := lockBackchannelLogoutIssuer(ctx, tx, identity)
	if err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	replayed, err := recordBackchannelLogoutReplay(ctx, tx, identity, tokenDigest)
	if err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	effectiveLogoutIssuedAt, err := persistBackchannelLogoutTombstone(ctx, tx, identity)
	if err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	plan := backchannelLogoutPlanFor(
		identity.IssuerID, issuer, identity.IssuerSessionID, identity.Subject, effectiveLogoutIssuedAt.UTC(),
	)
	if _, err := tx.ExecContext(ctx, plan.expireQuery, plan.arguments...); err != nil {
		return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("expire Management sessions for back-channel logout: %w", err)
	}
	if _, err := tx.ExecContext(ctx, plan.revokeQuery, plan.arguments...); err != nil {
		return managementidentity.BackchannelLogoutResult{}, fmt.Errorf("apply Management back-channel logout: %w", err)
	}
	if err := appendBackchannelLogoutAudit(ctx, tx, request, identity, revision, replayed); err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	ids, err := listBackchannelRevokedSessionIDs(ctx, tx, plan)
	if err != nil {
		return managementidentity.BackchannelLogoutResult{}, err
	}
	return managementidentity.BackchannelLogoutResult{SessionIDs: ids, Replayed: replayed}, nil
}

func lockBackchannelLogoutIssuer(
	ctx context.Context,
	tx *sql.Tx,
	identity managementauth.BackchannelLogoutIdentity,
) (string, uint64, error) {
	_, selectorDigest := backchannelLogoutSelector(identity)
	if _, err := tx.ExecContext(ctx, lockBackchannelLogoutSelectorQuery, selectorDigest[:]); err != nil {
		return "", 0, fmt.Errorf("lock Management back-channel logout selector: %w", err)
	}
	var issuer, status string
	var revision uint64
	if err := tx.QueryRowContext(ctx, `SELECT issuer,status,revision
FROM trusted_identity_issuers WHERE id=$1 FOR SHARE`, identity.IssuerID).Scan(&issuer, &status, &revision); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", 0, managementidentity.ErrNotFound
		}
		return "", 0, err
	}
	if status != "active" {
		return "", 0, managementidentity.ErrNotFound
	}
	return issuer, revision, nil
}

func recordBackchannelLogoutReplay(
	ctx context.Context,
	tx *sql.Tx,
	identity managementauth.BackchannelLogoutIdentity,
	tokenDigest [sha256.Size]byte,
) (bool, error) {
	if _, err := tx.ExecContext(ctx, expireBackchannelLogoutReplaysQuery); err != nil {
		return false, fmt.Errorf("expire Management back-channel logout replays: %w", err)
	}
	inserted, err := tx.ExecContext(
		ctx, insertBackchannelLogoutReplayQuery,
		identity.IssuerID, tokenDigest[:], identity.ClaimsDigest[:], identity.ExpiresAt.UTC(),
	)
	if err != nil {
		return false, fmt.Errorf("record Management back-channel logout replay: %w", err)
	}
	count, err := inserted.RowsAffected()
	if err != nil || count < 0 || count > 1 {
		return false, errors.New("management back-channel logout replay state is invalid")
	}
	if count == 1 {
		return false, nil
	}
	var stored []byte
	if err := tx.QueryRowContext(
		ctx, selectBackchannelLogoutReplayQuery, identity.IssuerID, tokenDigest[:],
	).Scan(&stored); err != nil || !slices.Equal(stored, identity.ClaimsDigest[:]) {
		return false, managementidentity.ErrBackchannelReplay
	}
	return true, nil
}

func persistBackchannelLogoutTombstone(
	ctx context.Context,
	tx *sql.Tx,
	identity managementauth.BackchannelLogoutIdentity,
) (time.Time, error) {
	selectorKind, selectorDigest := backchannelLogoutSelector(identity)
	var effectiveLogoutIssuedAt time.Time
	err := tx.QueryRowContext(
		ctx, upsertBackchannelLogoutTombstoneQuery, identity.IssuerID, selectorKind, selectorDigest[:],
		identity.IssuedAt.UTC(), identity.ExpiresAt.UTC(),
	).Scan(&effectiveLogoutIssuedAt)
	if err != nil {
		return time.Time{}, fmt.Errorf("persist Management back-channel logout selector: %w", err)
	}
	return effectiveLogoutIssuedAt, nil
}

func appendBackchannelLogoutAudit(
	ctx context.Context,
	tx *sql.Tx,
	request managementidentity.BackchannelLogout,
	identity managementauth.BackchannelLogoutIdentity,
	revision uint64,
	replayed bool,
) error {
	if replayed {
		return nil
	}
	return appendAudit(ctx, tx, auditMutation{
		Action: "management_session.backchannel_logout", ResourceType: "trusted_identity_issuer",
		ResourceID: identity.IssuerID, AfterRevision: revision, ExternalActor: true,
		Actor: managementidentity.MutationActor{
			RequestID: request.RequestID,
			Reason:    "Trusted issuer back-channel logout",
		},
	})
}

func listBackchannelRevokedSessionIDs(
	ctx context.Context,
	tx *sql.Tx,
	plan backchannelLogoutPlan,
) ([]string, error) {
	rows, err := tx.QueryContext(ctx, plan.listQuery, plan.arguments...)
	if err != nil {
		return nil, fmt.Errorf("list back-channel revoked Management sessions: %w", err)
	}
	defer rows.Close()
	ids := make([]string, 0)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		if !canonicalUUID(id) {
			return nil, errors.New("stored Management session identifier is invalid")
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return ids, nil
}

const expireBackchannelLogoutReplaysQuery = `WITH expired AS (
  SELECT issuer_id,token_id_digest FROM management_backchannel_logout_replays
  WHERE expires_at<=clock_timestamp()-interval '2 minutes' ORDER BY expires_at LIMIT 1000
)
DELETE FROM management_backchannel_logout_replays replay
USING expired
WHERE replay.issuer_id=expired.issuer_id AND replay.token_id_digest=expired.token_id_digest`

const insertBackchannelLogoutReplayQuery = `INSERT INTO management_backchannel_logout_replays
  (issuer_id,token_id_digest,claims_digest,expires_at)
VALUES ($1,$2,$3,$4)
ON CONFLICT (issuer_id,token_id_digest) DO NOTHING`

const selectBackchannelLogoutReplayQuery = `SELECT claims_digest
FROM management_backchannel_logout_replays
WHERE issuer_id=$1 AND token_id_digest=$2 FOR UPDATE`

const lockBackchannelLogoutSelectorQuery = `SELECT pg_advisory_xact_lock(
  hashtextextended(encode($1::bytea, 'hex'), 0))`

const upsertBackchannelLogoutTombstoneQuery = `INSERT INTO management_issuer_logout_tombstones
  (issuer_id,selector_kind,selector_digest,logout_issued_at,logout_expires_at)
VALUES ($1,$2,$3,$4,$5)
ON CONFLICT (issuer_id,selector_kind,selector_digest) DO UPDATE SET
  logout_expires_at = CASE
    WHEN EXCLUDED.logout_issued_at > management_issuer_logout_tombstones.logout_issued_at
      THEN EXCLUDED.logout_expires_at
    WHEN EXCLUDED.logout_issued_at = management_issuer_logout_tombstones.logout_issued_at
      THEN GREATEST(management_issuer_logout_tombstones.logout_expires_at, EXCLUDED.logout_expires_at)
    ELSE management_issuer_logout_tombstones.logout_expires_at
  END,
  logout_issued_at = GREATEST(
    management_issuer_logout_tombstones.logout_issued_at,
    EXCLUDED.logout_issued_at
  )
RETURNING logout_issued_at`

func backchannelLogoutSelector(identity managementauth.BackchannelLogoutIdentity) (string, [sha256.Size]byte) {
	if identity.IssuerSessionID != "" {
		return "sid", managementauth.IssuerSessionLogoutDigest(identity.IssuerID, identity.IssuerSessionID)
	}
	return "subject", managementauth.IssuerSubjectLogoutDigest(identity.IssuerID, identity.Subject)
}

func retryableLogoutTransactionError(err error) bool {
	var databaseError *pq.Error
	return errors.As(err, &databaseError) &&
		(databaseError.Code == "40001" || databaseError.Code == "40P01")
}

func validatedBackchannelLogout(request managementidentity.BackchannelLogout) (managementauth.BackchannelLogoutIdentity, [sha256.Size]byte, error) {
	identity := request.Identity
	if !canonicalUUID(identity.IssuerID) || identity.TokenID == "" ||
		(identity.Subject == "" && identity.IssuerSessionID == "") ||
		identity.IssuedAt.IsZero() || identity.ExpiresAt.IsZero() ||
		!identity.ExpiresAt.After(identity.IssuedAt) || request.RequestID == "" {
		return managementauth.BackchannelLogoutIdentity{}, [sha256.Size]byte{}, managementidentity.ErrInvalidLifecycleRequest
	}
	digest := sha256.Sum256([]byte("vllm-sr/backchannel-logout/v1\x00" + identity.IssuerID + "\x00" + identity.TokenID))
	return identity, digest, nil
}

const backchannelLogoutSIDSelector = `auth_source_kind='issuer' AND auth_source_id=$1 AND issuer_session_id=$2`

const backchannelLogoutSubjectSelector = `auth_source_kind='issuer' AND auth_source_id=$1 AND principal_id IN
  (SELECT id FROM management_principals WHERE issuer=$2 AND subject=$3)
  AND authenticated_at<=$4`

const expireBackchannelLogoutBySIDQuery = `UPDATE management_sessions SET status='expired'
WHERE ` + backchannelLogoutSIDSelector + ` AND status='active' AND expires_at<=clock_timestamp()`

const expireBackchannelLogoutBySubjectQuery = `UPDATE management_sessions SET status='expired'
WHERE ` + backchannelLogoutSubjectSelector + ` AND status='active' AND expires_at<=clock_timestamp()`

const revokeBackchannelLogoutBySIDQuery = `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE ` + backchannelLogoutSIDSelector + ` AND status='active'`

const revokeBackchannelLogoutBySubjectQuery = `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE ` + backchannelLogoutSubjectSelector + ` AND status='active'`

const listBackchannelLogoutBySIDQuery = `SELECT id::text FROM management_sessions
WHERE ` + backchannelLogoutSIDSelector + ` AND status='revoked' AND expires_at>clock_timestamp()
ORDER BY id`

const listBackchannelLogoutBySubjectQuery = `SELECT id::text FROM management_sessions
WHERE ` + backchannelLogoutSubjectSelector + ` AND status='revoked' AND expires_at>clock_timestamp()
ORDER BY id`

type backchannelLogoutPlan struct {
	expireQuery string
	revokeQuery string
	listQuery   string
	arguments   []any
}

func backchannelLogoutPlanFor(
	issuerID, issuer, issuerSessionID, subject string,
	logoutIssuedAt time.Time,
) backchannelLogoutPlan {
	if issuerSessionID != "" {
		return backchannelLogoutPlan{
			expireQuery: expireBackchannelLogoutBySIDQuery,
			revokeQuery: revokeBackchannelLogoutBySIDQuery, listQuery: listBackchannelLogoutBySIDQuery,
			arguments: []any{issuerID, issuerSessionID},
		}
	}
	return backchannelLogoutPlan{
		expireQuery: expireBackchannelLogoutBySubjectQuery,
		revokeQuery: revokeBackchannelLogoutBySubjectQuery, listQuery: listBackchannelLogoutBySubjectQuery,
		arguments: []any{issuerID, issuer, subject, logoutIssuedAt},
	}
}
