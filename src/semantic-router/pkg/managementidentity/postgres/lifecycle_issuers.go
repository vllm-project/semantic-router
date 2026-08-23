package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const trustedIssuerColumns = `id::text,issuer,kind,discovery_url,jwks_url,audiences,
       claim_mapping,assurance_mapping,status,revision,created_at,updated_at`

func (store *Store) GetTrustedIdentityIssuer(
	ctx context.Context,
	issuerID string,
) (managementidentity.TrustedIdentityIssuer, error) {
	if !canonicalUUID(issuerID) {
		return managementidentity.TrustedIdentityIssuer{}, managementidentity.ErrNotFound
	}
	issuer, err := scanTrustedIdentityIssuer(store.database.QueryRowContext(ctx,
		`SELECT `+trustedIssuerColumns+` FROM trusted_identity_issuers WHERE id=$1`, issuerID))
	return trustedIssuerResult(issuer, err)
}

func (store *Store) ListTrustedIdentityIssuers(
	ctx context.Context,
	request managementidentity.ListRequest,
) (managementidentity.TrustedIdentityIssuerPage, error) {
	if err := validateList(request); err != nil {
		return managementidentity.TrustedIdentityIssuerPage{}, err
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+trustedIssuerColumns+`
FROM trusted_identity_issuers
WHERE ($1='' OR id>NULLIF($1,'')::uuid) ORDER BY id LIMIT $2`, request.AfterID, request.Limit+1)
	if err != nil {
		return managementidentity.TrustedIdentityIssuerPage{}, fmt.Errorf("list trusted identity issuers: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.TrustedIdentityIssuer, 0, request.Limit+1)
	for rows.Next() {
		issuer, err := scanTrustedIdentityIssuer(rows)
		if err != nil {
			return managementidentity.TrustedIdentityIssuerPage{}, err
		}
		items = append(items, issuer)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.TrustedIdentityIssuerPage{}, err
	}
	page := managementidentity.TrustedIdentityIssuerPage{Items: items}
	if len(items) > request.Limit {
		page.Items = items[:request.Limit]
		page.NextCursor = page.Items[len(page.Items)-1].ID
	}
	return page, nil
}

func (store *Store) CreateTrustedIdentityIssuer(
	ctx context.Context,
	request managementidentity.CreateTrustedIdentityIssuer,
) (managementidentity.IssuerMutation, error) {
	if request.Command.Scope.Kind != managementcommand.ScopeCluster ||
		request.Command.PrincipalID != request.Actor.PrincipalID {
		return managementidentity.IssuerMutation{}, managementidentity.ErrInvalidLifecycleRequest
	}
	audiences, claims, assurance, err := encodeIssuerDocuments(request.Issuer)
	if err != nil {
		return managementidentity.IssuerMutation{}, err
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.IssuerMutation, error) {
		if stored, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.IssuerMutation{}, err
		} else if found {
			result, err := replayMutation(stored, "trusted_identity_issuer")
			if err != nil {
				return managementidentity.IssuerMutation{}, managementcommand.ErrConflict
			}
			issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx,
				`SELECT `+trustedIssuerColumns+` FROM trusted_identity_issuers WHERE id=$1`, result.ID))
			if err != nil {
				return managementidentity.IssuerMutation{}, err
			}
			return managementidentity.IssuerMutation{Result: result, Issuer: issuer}, nil
		}
		issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx, `INSERT INTO trusted_identity_issuers
  (id,issuer,kind,discovery_url,jwks_url,audiences,claim_mapping,assurance_mapping,status,revision)
VALUES ($1,$2,$3,NULLIF($4,''),NULLIF($5,''),$6,$7,$8,'active',1)
RETURNING `+trustedIssuerColumns, request.Issuer.ID, request.Issuer.Issuer,
			request.Issuer.Kind, request.Issuer.DiscoveryURL, request.Issuer.JWKSURL,
			audiences, claims, assurance))
		if err != nil {
			return managementidentity.IssuerMutation{}, mapWriteError("create trusted identity issuer", err)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "trusted_identity_issuer.created", ResourceType: "trusted_identity_issuer",
			ResourceID: issuer.ID, AfterRevision: issuer.Revision, Actor: request.Actor,
		}); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		stored := managementcommand.ResourceResult{
			ResourceType: "trusted_identity_issuer", ResourceID: issuer.ID,
			ResourceRevision: issuer.Revision, ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, stored); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		return managementidentity.IssuerMutation{
			Result: managementidentity.MutationResult{
				Kind: stored.ResourceType, ID: stored.ResourceID, Revision: stored.ResourceRevision,
				ResponseStatus: stored.ResponseStatus,
			},
			Issuer: issuer,
		}, nil
	})
}

func (store *Store) UpdateTrustedIdentityIssuer(
	ctx context.Context,
	request managementidentity.UpdateTrustedIdentityIssuer,
) (managementidentity.IssuerMutation, error) {
	var audiences, claims, assurance any
	var err error
	if request.Audiences != nil {
		audiences, err = json.Marshal(*request.Audiences)
	}
	if err == nil && request.ClaimMapping != nil {
		claims, err = json.Marshal(*request.ClaimMapping)
	}
	if err == nil && request.AssuranceMapping != nil {
		assurance, err = json.Marshal(*request.AssuranceMapping)
	}
	if err != nil {
		return managementidentity.IssuerMutation{}, managementidentity.ErrInvalidLifecycleRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.IssuerMutation, error) {
		issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx, `UPDATE trusted_identity_issuers SET
  discovery_url=CASE WHEN $3::boolean THEN NULLIF($4,'') ELSE discovery_url END,
  jwks_url=CASE WHEN $5::boolean THEN NULLIF($6,'') ELSE jwks_url END,
  audiences=CASE WHEN $7::boolean THEN $8::jsonb ELSE audiences END,
  claim_mapping=CASE WHEN $9::boolean THEN $10::jsonb ELSE claim_mapping END,
  assurance_mapping=CASE WHEN $11::boolean THEN $12::jsonb ELSE assurance_mapping END,
  status=COALESCE($13,status),revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2
RETURNING `+trustedIssuerColumns,
			request.ID, request.ExpectedRevision,
			request.DiscoveryURL != nil, stringValue(request.DiscoveryURL),
			request.JWKSURL != nil, stringValue(request.JWKSURL),
			request.Audiences != nil, audiences,
			request.ClaimMapping != nil, claims,
			request.AssuranceMapping != nil, assurance,
			nullableResourceStatus(request.Status)))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.IssuerMutation{}, classifyRevision(ctx, tx, "trusted_identity_issuers", request.ID)
		}
		if err != nil {
			return managementidentity.IssuerMutation{}, mapWriteError("update trusted identity issuer", err)
		}
		sessionIDs, err := revokeIssuerSessions(ctx, tx, request.ID)
		if err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		before := request.ExpectedRevision
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "trusted_identity_issuer.updated", ResourceType: "trusted_identity_issuer",
			ResourceID: request.ID, BeforeRevision: &before,
			AfterRevision: issuer.Revision, Actor: request.Actor,
		}); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		return managementidentity.IssuerMutation{
			Result: managementidentity.MutationResult{
				Kind: "trusted_identity_issuer", ID: request.ID, Revision: issuer.Revision,
				ResponseStatus: 200,
			},
			Issuer: issuer, Sessions: sessionIDs,
		}, nil
	})
}

func (store *Store) DeleteTrustedIdentityIssuer(
	ctx context.Context,
	issuerID string,
	expectedRevision uint64,
	actor managementidentity.MutationActor,
) (managementidentity.IssuerMutation, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.IssuerMutation, error) {
		issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx,
			`SELECT `+trustedIssuerColumns+` FROM trusted_identity_issuers WHERE id=$1 FOR UPDATE`, issuerID))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.IssuerMutation{}, managementidentity.ErrNotFound
		}
		if err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		if issuer.Revision != expectedRevision {
			return managementidentity.IssuerMutation{}, managementidentity.ErrRevisionConflict
		}
		sessionIDs, err := revokeIssuerSessions(ctx, tx, issuerID)
		if err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		result, err := tx.ExecContext(ctx, `DELETE FROM trusted_identity_issuers
WHERE id=$1 AND revision=$2`, issuerID, expectedRevision)
		if err != nil {
			return managementidentity.IssuerMutation{}, mapWriteError("delete trusted identity issuer", err)
		}
		count, countErr := result.RowsAffected()
		if countErr != nil {
			return managementidentity.IssuerMutation{}, countErr
		}
		if count != 1 {
			return managementidentity.IssuerMutation{}, managementidentity.ErrRevisionConflict
		}
		after := expectedRevision + 1
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "trusted_identity_issuer.deleted", ResourceType: "trusted_identity_issuer",
			ResourceID: issuerID, BeforeRevision: &expectedRevision,
			AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		issuer.Revision = after
		return managementidentity.IssuerMutation{
			Result: managementidentity.MutationResult{
				Kind: "trusted_identity_issuer", ID: issuerID, Revision: after,
				ResponseStatus: 204,
			},
			Issuer: issuer, Sessions: sessionIDs,
		}, nil
	})
}

func (store *Store) RefreshTrustedIdentityIssuer(
	ctx context.Context,
	request managementidentity.RefreshTrustedIdentityIssuer,
) (managementidentity.IssuerMutation, error) {
	if request.Command.Scope.Kind != managementcommand.ScopeCluster ||
		request.Command.PrincipalID != request.Actor.PrincipalID {
		return managementidentity.IssuerMutation{}, managementidentity.ErrInvalidLifecycleRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.IssuerMutation, error) {
		if stored, found, err := commandpostgres.Lock(ctx, tx, request.Command); err != nil {
			return managementidentity.IssuerMutation{}, err
		} else if found {
			result, err := replayMutation(stored, "trusted_identity_issuer")
			if err != nil || result.ID != request.ID {
				return managementidentity.IssuerMutation{}, managementcommand.ErrConflict
			}
			issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx,
				`SELECT `+trustedIssuerColumns+` FROM trusted_identity_issuers WHERE id=$1`, request.ID))
			if err != nil {
				return managementidentity.IssuerMutation{}, err
			}
			sessionIDs, err := revokedIssuerSessionIDs(ctx, tx, request.ID)
			if err != nil {
				return managementidentity.IssuerMutation{}, err
			}
			return managementidentity.IssuerMutation{Result: result, Issuer: issuer, Sessions: sessionIDs}, nil
		}
		var before uint64
		if err := tx.QueryRowContext(ctx, `SELECT revision FROM trusted_identity_issuers
WHERE id=$1 AND status='active' FOR UPDATE`, request.ID).Scan(&before); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.IssuerMutation{}, managementidentity.ErrNotFound
			}
			return managementidentity.IssuerMutation{}, err
		}
		issuer, err := scanTrustedIdentityIssuer(tx.QueryRowContext(ctx, `UPDATE trusted_identity_issuers
SET revision=revision+1,updated_at=clock_timestamp() WHERE id=$1
RETURNING `+trustedIssuerColumns, request.ID))
		if err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		sessionIDs, err := revokedIssuerSessionIDs(ctx, tx, request.ID)
		if err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "trusted_identity_issuer.keys_refreshed", ResourceType: "trusted_identity_issuer",
			ResourceID: request.ID, BeforeRevision: &before,
			AfterRevision: issuer.Revision, Actor: request.Actor,
		}); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		stored := managementcommand.ResourceResult{
			ResourceType: "trusted_identity_issuer", ResourceID: request.ID,
			ResourceRevision: issuer.Revision, ResponseStatus: 200,
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command, stored); err != nil {
			return managementidentity.IssuerMutation{}, err
		}
		return managementidentity.IssuerMutation{
			Result: managementidentity.MutationResult{
				Kind: stored.ResourceType, ID: stored.ResourceID, Revision: stored.ResourceRevision,
				ResponseStatus: stored.ResponseStatus,
			},
			Issuer: issuer, Sessions: sessionIDs,
		}, nil
	})
}

func revokeIssuerSessions(ctx context.Context, tx *sql.Tx, issuerID string) ([]string, error) {
	if _, err := tx.ExecContext(ctx, `UPDATE management_sessions SET status='expired'
WHERE auth_source_kind='issuer' AND auth_source_id=$1
  AND status='active' AND expires_at<=clock_timestamp()`, issuerID); err != nil {
		return nil, fmt.Errorf("expire trusted-issuer Management sessions: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `UPDATE management_sessions
SET status='revoked',revoked_at=COALESCE(revoked_at,clock_timestamp())
WHERE auth_source_kind='issuer' AND auth_source_id=$1 AND status='active'`, issuerID); err != nil {
		return nil, fmt.Errorf("revoke trusted-issuer Management sessions: %w", err)
	}
	return revokedIssuerSessionIDs(ctx, tx, issuerID)
}

func revokedIssuerSessionIDs(ctx context.Context, tx *sql.Tx, issuerID string) ([]string, error) {
	rows, err := tx.QueryContext(ctx, `SELECT id::text FROM management_sessions
WHERE auth_source_kind='issuer' AND auth_source_id=$1
  AND status='revoked' AND expires_at>clock_timestamp()
ORDER BY id`, issuerID)
	if err != nil {
		return nil, err
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
	return ids, rows.Err()
}

func scanTrustedIdentityIssuer(row scanner) (managementidentity.TrustedIdentityIssuer, error) {
	var issuer managementidentity.TrustedIdentityIssuer
	var discoveryURL, jwksURL sql.NullString
	var audiences, claims, assurance []byte
	if err := row.Scan(
		&issuer.ID, &issuer.Issuer, &issuer.Kind, &discoveryURL, &jwksURL,
		&audiences, &claims, &assurance, &issuer.Status, &issuer.Revision,
		&issuer.CreatedAt, &issuer.UpdatedAt,
	); err != nil {
		return managementidentity.TrustedIdentityIssuer{}, err
	}
	issuer.DiscoveryURL, issuer.JWKSURL = discoveryURL.String, jwksURL.String
	if err := json.Unmarshal(audiences, &issuer.Audiences); err != nil {
		return managementidentity.TrustedIdentityIssuer{}, errors.New("stored trusted-issuer audiences are invalid")
	}
	if err := json.Unmarshal(claims, &issuer.ClaimMapping); err != nil {
		return managementidentity.TrustedIdentityIssuer{}, errors.New("stored trusted-issuer claim mapping is invalid")
	}
	if err := json.Unmarshal(assurance, &issuer.AssuranceMapping); err != nil {
		return managementidentity.TrustedIdentityIssuer{}, errors.New("stored trusted-issuer assurance mapping is invalid")
	}
	if issuer.ClaimMapping == nil {
		issuer.ClaimMapping = map[string]string{}
	}
	if issuer.AssuranceMapping == nil {
		issuer.AssuranceMapping = map[string]string{}
	}
	issuer.CreatedAt, issuer.UpdatedAt = issuer.CreatedAt.UTC(), issuer.UpdatedAt.UTC()
	if issuer.VerificationValue().Validate() != nil ||
		(issuer.Status != managementauth.ResourceActive && issuer.Status != managementauth.ResourceDisabled) ||
		issuer.CreatedAt.IsZero() || issuer.UpdatedAt.IsZero() {
		return managementidentity.TrustedIdentityIssuer{}, errors.New("stored trusted identity issuer is invalid")
	}
	return issuer, nil
}

func trustedIssuerResult(
	issuer managementidentity.TrustedIdentityIssuer,
	err error,
) (managementidentity.TrustedIdentityIssuer, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.TrustedIdentityIssuer{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.TrustedIdentityIssuer{}, fmt.Errorf("load trusted identity issuer: %w", err)
	}
	return issuer, nil
}

func encodeIssuerDocuments(issuer managementidentity.TrustedIdentityIssuer) ([]byte, []byte, []byte, error) {
	audiences, err := json.Marshal(issuer.Audiences)
	if err != nil {
		return nil, nil, nil, err
	}
	claims, err := json.Marshal(issuer.ClaimMapping)
	if err != nil {
		return nil, nil, nil, err
	}
	assurance, err := json.Marshal(issuer.AssuranceMapping)
	return audiences, claims, assurance, err
}

func nullableResourceStatus(status *managementauth.ResourceStatus) any {
	if status == nil {
		return nil
	}
	return string(*status)
}
