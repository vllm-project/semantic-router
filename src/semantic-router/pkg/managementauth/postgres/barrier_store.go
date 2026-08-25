package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

var ErrInvalidBarrier = errors.New("management revocation barrier is invalid")

// BarrierStore uses PostgreSQL itself as the strongly consistent revocation
// authority when no shared runtime store is configured. Durable lifecycle
// state is checked directly, while explicit rows preserve the deny fence
// across the small interval between a lifecycle mutation and its acknowledgement.
type BarrierStore struct {
	db *sql.DB
}

func NewBarrierStore(db *sql.DB) (*BarrierStore, error) {
	if db == nil {
		return nil, errors.New("management revocation PostgreSQL database is required")
	}
	return &BarrierStore{db: db}, nil
}

func (store *BarrierStore) Ready(ctx context.Context) error {
	if store == nil || store.db == nil {
		return managementauth.ErrAuthenticationUnavailable
	}
	if err := store.db.PingContext(ctx); err != nil {
		return fmt.Errorf("check Management revocation PostgreSQL: %w", err)
	}
	return nil
}

func (store *BarrierStore) Check(
	ctx context.Context,
	check managementauth.BarrierCheck,
) (managementauth.BarrierState, error) {
	if store == nil || store.db == nil || validateBarrierCheck(check) != nil {
		return managementauth.BarrierState{}, ErrInvalidBarrier
	}
	authSourceID := string(check.AuthSourceKind) + ":" + check.AuthSourceID
	var state managementauth.BarrierState
	err := store.db.QueryRowContext(ctx, `SELECT
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='cluster_session_policy' AND barrier_id='singleton'),
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='namespace_security_policy' AND barrier_id=$1),
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='management_session' AND barrier_id=$2)
    OR EXISTS (SELECT 1 FROM management_sessions WHERE id=$2::uuid AND status <> 'active'),
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='management_principal' AND barrier_id=$3)
    OR EXISTS (SELECT 1 FROM management_principals WHERE id=$3::uuid AND status <> 'active'),
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='authentication_source' AND barrier_id=$4)
    OR CASE $5
      WHEN 'issuer' THEN EXISTS (
        SELECT 1 FROM trusted_identity_issuers WHERE id=$6::uuid AND status <> 'active')
      WHEN 'mtls' THEN EXISTS (
        SELECT 1 FROM management_mtls_mappings WHERE id=$6::uuid AND status <> 'active')
      WHEN 'service_credential' THEN EXISTS (
        SELECT 1 FROM management_service_account_credentials credential
        JOIN management_service_accounts account ON account.id=credential.service_account_id
        WHERE credential.id=$6::uuid AND (credential.status='revoked' OR account.status <> 'active'))
      ELSE TRUE
    END`, check.NamespaceID, check.SessionID, check.PrincipalID, authSourceID,
		string(check.AuthSourceKind), check.AuthSourceID,
	).Scan(&state.ClusterDenied, &state.NamespaceDenied, &state.SessionDenied,
		&state.PrincipalDenied, &state.AuthSourceDenied)
	if err != nil {
		return managementauth.BarrierState{}, fmt.Errorf("check Management revocation PostgreSQL: %w", err)
	}
	state.Ready = true
	return state, nil
}

func (store *BarrierStore) CheckDelegation(
	ctx context.Context,
	check managementauth.DelegationBarrierCheck,
) (managementauth.DelegationBarrierState, error) {
	if store == nil || store.db == nil || !canonicalUUID(check.SessionID) || !canonicalUUID(check.PrincipalID) {
		return managementauth.DelegationBarrierState{}, ErrInvalidBarrier
	}
	var state managementauth.DelegationBarrierState
	err := store.db.QueryRowContext(ctx, `SELECT
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='management_session' AND barrier_id=$1)
    OR EXISTS (SELECT 1 FROM management_sessions WHERE id=$1::uuid AND status <> 'active'),
  EXISTS (SELECT 1 FROM management_revocation_barriers
          WHERE barrier_kind='management_principal' AND barrier_id=$2)
    OR EXISTS (SELECT 1 FROM management_principals WHERE id=$2::uuid AND status <> 'active')`,
		check.SessionID, check.PrincipalID,
	).Scan(&state.SessionDenied, &state.PrincipalDenied)
	if err != nil {
		return managementauth.DelegationBarrierState{}, fmt.Errorf("check delegated revocation PostgreSQL: %w", err)
	}
	state.Ready = true
	return state, nil
}

func (store *BarrierStore) InstallDeny(
	ctx context.Context,
	kind managementauth.BarrierKind,
	id string,
) error {
	if store == nil || store.db == nil || validateBarrierValue(kind, id) != nil {
		return ErrInvalidBarrier
	}
	_, err := store.db.ExecContext(ctx, `INSERT INTO management_revocation_barriers
  (barrier_kind,barrier_id) VALUES ($1,$2)
ON CONFLICT (barrier_kind,barrier_id) DO NOTHING`, kind, id)
	if err != nil {
		return fmt.Errorf("install Management revocation barrier: %w", err)
	}
	return nil
}

func (store *BarrierStore) RemoveDeny(
	ctx context.Context,
	kind managementauth.BarrierKind,
	id string,
) error {
	if store == nil || store.db == nil || validateBarrierValue(kind, id) != nil {
		return ErrInvalidBarrier
	}
	_, err := store.db.ExecContext(ctx, `DELETE FROM management_revocation_barriers
WHERE barrier_kind=$1 AND barrier_id=$2`, kind, id)
	if err != nil {
		return fmt.Errorf("remove Management revocation barrier: %w", err)
	}
	return nil
}

func validateBarrierCheck(check managementauth.BarrierCheck) error {
	for _, value := range []string{check.SessionID, check.PrincipalID, check.AuthSourceID} {
		if !canonicalUUID(value) {
			return ErrInvalidBarrier
		}
	}
	if check.NamespaceID != "" && !canonicalUUID(check.NamespaceID) {
		return ErrInvalidBarrier
	}
	switch check.AuthSourceKind {
	case managementauth.AuthSourceIssuer, managementauth.AuthSourceServiceCredential, managementauth.AuthSourceMTLS:
		return nil
	default:
		return ErrInvalidBarrier
	}
}

func validateBarrierValue(kind managementauth.BarrierKind, id string) error {
	switch kind {
	case managementauth.BarrierClusterSessionPolicy:
		if id == "singleton" {
			return nil
		}
	case managementauth.BarrierNamespaceSecurityPolicy, managementauth.BarrierManagementSession,
		managementauth.BarrierManagementPrincipal:
		if canonicalUUID(id) {
			return nil
		}
	case managementauth.BarrierAuthenticationSource:
		sourceKind, sourceID, found := strings.Cut(id, ":")
		if found && canonicalUUID(sourceID) {
			switch managementauth.AuthSourceKind(sourceKind) {
			case managementauth.AuthSourceIssuer, managementauth.AuthSourceServiceCredential, managementauth.AuthSourceMTLS:
				return nil
			}
		}
	}
	return ErrInvalidBarrier
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed != uuid.Nil && parsed.String() == value
}

var (
	_ managementauth.RevocationBarrierStore           = (*BarrierStore)(nil)
	_ managementauth.DelegationRevocationBarrierStore = (*BarrierStore)(nil)
)
