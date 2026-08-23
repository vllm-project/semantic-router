package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const durableRevocationsQuery = `SELECT barrier_kind, barrier_id
FROM (
  SELECT 'management_session'::text AS barrier_kind, id::text AS barrier_id
  FROM management_sessions WHERE status = 'revoked'
  UNION ALL
  SELECT 'management_principal', id::text
  FROM management_principals WHERE status = 'disabled'
  UNION ALL
  SELECT 'authentication_source', 'issuer:' || id::text
  FROM trusted_identity_issuers WHERE status = 'disabled'
  UNION ALL
  SELECT 'authentication_source', 'mtls:' || id::text
  FROM management_mtls_mappings WHERE status = 'disabled'
  UNION ALL
  SELECT 'authentication_source', 'service_credential:' || credential.id::text
  FROM management_service_account_credentials AS credential
  JOIN management_service_accounts AS account ON account.id = credential.service_account_id
  WHERE credential.status = 'revoked' OR account.status = 'disabled'
) AS barriers
ORDER BY barrier_kind, barrier_id`

// LoadRevocationBarriers reconstructs lifecycle deny facts from one
// repeatable-read snapshot. Cluster/Namespace security-policy barriers are
// operation fences and are restored by their publication coordinator; this
// loader owns only durable identity/session/source lifecycle facts.
func (store *Store) LoadRevocationBarriers(ctx context.Context) (_ []managementauth.RevocationBarrier, resultErr error) {
	if store == nil || store.db == nil {
		return nil, errors.New("management session PostgreSQL store is unavailable")
	}
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return nil, fmt.Errorf("begin Management revocation snapshot: %w", err)
	}
	defer func() {
		if resultErr != nil {
			_ = tx.Rollback()
		}
	}()
	rows, err := tx.QueryContext(ctx, durableRevocationsQuery)
	if err != nil {
		return nil, fmt.Errorf("load Management revocation snapshot: %w", err)
	}
	defer rows.Close()
	barriers := make([]managementauth.RevocationBarrier, 0)
	for rows.Next() {
		var kind managementauth.BarrierKind
		var id string
		if err := rows.Scan(&kind, &id); err != nil {
			return nil, fmt.Errorf("scan Management revocation snapshot: %w", err)
		}
		if err := validateRevocationBarrier(kind, id); err != nil {
			return nil, err
		}
		barriers = append(barriers, managementauth.RevocationBarrier{Kind: kind, ID: id})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Management revocation snapshot: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return nil, fmt.Errorf("commit Management revocation snapshot: %w", err)
	}
	return barriers, nil
}

func validateRevocationBarrier(kind managementauth.BarrierKind, id string) error {
	switch kind {
	case managementauth.BarrierManagementSession, managementauth.BarrierManagementPrincipal:
		if parsed, err := uuid.Parse(id); err == nil && parsed.String() == id {
			return nil
		}
	case managementauth.BarrierAuthenticationSource:
		for _, prefix := range []string{"issuer:", "service_credential:", "mtls:"} {
			if len(id) > len(prefix) && id[:len(prefix)] == prefix {
				parsed, err := uuid.Parse(id[len(prefix):])
				if err == nil && parsed.String() == id[len(prefix):] {
					return nil
				}
			}
		}
	}
	return errors.New("durable Management revocation snapshot contains an invalid barrier")
}
