package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const (
	lockIssuerLogoutSelectorQuery = `SELECT pg_advisory_xact_lock(
  hashtextextended(encode($1::bytea, 'hex'), 0))`

	loadIssuerLogoutTombstoneQuery = `SELECT logout_issued_at
FROM management_issuer_logout_tombstones
WHERE issuer_id=$1 AND selector_kind=$2 AND selector_digest=$3`
)

type issuerLogoutSelector struct {
	kind   string
	digest [32]byte
}

// RejectLoggedOutIssuerIdentityInTransaction applies the durable issuer
// logout fences before an exchange reads or creates any Management session.
// Callers must invoke it as the first statement in their serializable
// transaction and retry serialization failures so a waiter cannot act on the
// snapshot it took before acquiring the transaction-scoped selector lock.
func (s *Store) RejectLoggedOutIssuerIdentityInTransaction(
	ctx context.Context,
	tx *sql.Tx,
	identity managementauth.VerifiedExternalIdentity,
) error {
	if s == nil || s.db == nil || tx == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if !canonicalUUID(identity.IssuerID) || identity.Subject == "" || identity.AuthenticatedAt.IsZero() ||
		(identity.IssuerSessionID != nil && *identity.IssuerSessionID == "") {
		return managementauth.ErrAuthenticationDenied
	}
	selectors := []issuerLogoutSelector{{
		kind:   "subject",
		digest: managementauth.IssuerSubjectLogoutDigest(identity.IssuerID, identity.Subject),
	}}
	if identity.IssuerSessionID != nil {
		selectors = append(selectors, issuerLogoutSelector{
			kind:   "sid",
			digest: managementauth.IssuerSessionLogoutDigest(identity.IssuerID, *identity.IssuerSessionID),
		})
	}
	for _, selector := range selectors {
		if _, err := tx.ExecContext(ctx, lockIssuerLogoutSelectorQuery, selector.digest[:]); err != nil {
			return fmt.Errorf("lock issuer logout selector: %w", err)
		}
	}
	for _, selector := range selectors {
		var logoutIssuedAt time.Time
		err := tx.QueryRowContext(
			ctx,
			loadIssuerLogoutTombstoneQuery,
			identity.IssuerID,
			selector.kind,
			selector.digest[:],
		).Scan(&logoutIssuedAt)
		if errors.Is(err, sql.ErrNoRows) {
			continue
		}
		if err != nil {
			return fmt.Errorf("load issuer logout tombstone: %w", err)
		}
		if selector.kind == "sid" || !identity.AuthenticatedAt.UTC().After(logoutIssuedAt.UTC()) {
			return managementauth.ErrAuthenticationDenied
		}
	}
	return nil
}
