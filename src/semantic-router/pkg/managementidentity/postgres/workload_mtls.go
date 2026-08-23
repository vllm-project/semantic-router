package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const mtlsMappingColumns = `id::text,matcher_kind,matcher_value,principal_id::text,
       workload_class,source_assured_at,status,revision,created_at,updated_at`

func (store *Store) GetMTLSMapping(ctx context.Context, id string) (managementidentity.MTLSIdentityMapping, error) {
	if !canonicalUUID(id) {
		return managementidentity.MTLSIdentityMapping{}, managementidentity.ErrNotFound
	}
	mapping, err := scanMTLSMapping(store.database.QueryRowContext(ctx,
		`SELECT `+mtlsMappingColumns+` FROM management_mtls_mappings WHERE id=$1`, id))
	return mtlsMappingResult(mapping, err)
}

func (store *Store) ListMTLSMappings(
	ctx context.Context,
	query managementidentity.MTLSMappingQuery,
) (managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping], error) {
	if query.Limit < 1 || query.Limit > maximumPageSize ||
		(query.Status != "" && query.Status != string(managementauth.ResourceActive) && query.Status != string(managementauth.ResourceDisabled)) {
		return managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{}, managementidentity.ErrInvalidWorkloadRequest
	}
	var afterTime, afterID any
	if query.After != nil {
		if query.After.CreatedAt.IsZero() || !canonicalUUID(query.After.ID) {
			return managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{}, managementidentity.ErrInvalidWorkloadRequest
		}
		afterTime, afterID = query.After.CreatedAt.UTC(), query.After.ID
	}
	rows, err := store.database.QueryContext(ctx, `SELECT `+mtlsMappingColumns+`
FROM management_mtls_mappings
WHERE ($1='' OR status=$1)
  AND ($2::timestamptz IS NULL OR (created_at,id)>($2,$3::uuid))
ORDER BY created_at,id LIMIT $4`, query.Status, afterTime, afterID, query.Limit+1)
	if err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{}, fmt.Errorf("list Management mTLS identity mappings: %w", err)
	}
	defer rows.Close()
	items := make([]managementidentity.MTLSIdentityMapping, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanMTLSMapping(rows)
		if err != nil {
			return managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{}, fmt.Errorf("scan Management mTLS mapping page: %w", err)
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{}, fmt.Errorf("iterate Management mTLS mapping page: %w", err)
	}
	page := managementidentity.WorkloadRepositoryPage[managementidentity.MTLSIdentityMapping]{Items: items}
	if len(items) > query.Limit {
		page.Items, page.HasMore = items[:query.Limit], true
	}
	return page, nil
}

func (store *Store) CreateMTLSMapping(
	ctx context.Context,
	mutation managementidentity.MTLSMappingCreateMutation,
) (managementidentity.WorkloadMutationResult, error) {
	mapping := mutation.Mapping
	if !canonicalUUID(mapping.ID) || !canonicalUUID(mapping.PrincipalID) || mapping.Revision != 1 ||
		mutation.Command.Scope.Kind != managementcommand.ScopeCluster ||
		mutation.Command.PrincipalID != mutation.Actor.PrincipalID {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		if stored, found, err := commandpostgres.Lock(ctx, tx, mutation.Command); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapCommandError(err)
		} else if found {
			result, err := replayWorkloadMutation(stored, "mtls_identity_mapping")
			if err != nil {
				return managementidentity.WorkloadMutationResult{}, err
			}
			return result, nil
		}
		var status string
		if err := tx.QueryRowContext(ctx,
			`SELECT status FROM management_principals WHERE id=$1 FOR SHARE`, mapping.PrincipalID,
		).Scan(&status); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
			}
			return managementidentity.WorkloadMutationResult{}, err
		}
		if status != "active" {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_mtls_mappings
  (id,matcher_kind,matcher_value,principal_id,workload_class,source_assured_at,
   status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,1,$8,$8)`, mapping.ID, mapping.MatcherKind,
			mapping.MatcherValue, mapping.PrincipalID, mapping.WorkloadClass,
			mapping.SourceAssuredAt, mapping.Status, mapping.CreatedAt); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWriteError("create Management mTLS identity mapping", err)
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "mtls_identity_mapping.created", ResourceType: "mtls_identity_mapping",
			ResourceID: mapping.ID, AfterRevision: 1, Actor: mutation.Actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		result := managementcommand.ResourceResult{
			ResourceType: "mtls_identity_mapping", ResourceID: mapping.ID,
			ResourceRevision: 1, ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteResource(ctx, tx, mutation.Command, result); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: result.ResourceType, ID: result.ResourceID, Revision: 1, HTTPStatus: 201,
		}, nil
	})
}

func (store *Store) PatchMTLSMapping(
	ctx context.Context,
	updated managementidentity.MTLSIdentityMapping,
	expected uint64,
	actor managementidentity.MutationActor,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(updated.ID) || !canonicalUUID(updated.PrincipalID) || expected == 0 {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		current, patchMTLSMappingErr := scanMTLSMapping(tx.QueryRowContext(ctx,
			`SELECT `+mtlsMappingColumns+` FROM management_mtls_mappings WHERE id=$1 FOR UPDATE`, updated.ID))
		if errors.Is(patchMTLSMappingErr, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if patchMTLSMappingErr != nil {
			return managementidentity.WorkloadMutationResult{}, patchMTLSMappingErr
		}
		if current.Revision != expected {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		if current.MatcherKind != updated.MatcherKind || current.MatcherValue != updated.MatcherValue ||
			current.PrincipalID != updated.PrincipalID {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
		}
		var principalStatus string
		if err := tx.QueryRowContext(ctx,
			`SELECT status FROM management_principals WHERE id=$1 FOR SHARE`, updated.PrincipalID,
		).Scan(&principalStatus); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
			}
			return managementidentity.WorkloadMutationResult{}, err
		}
		if principalStatus != "active" {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		var revision uint64
		if err := tx.QueryRowContext(ctx, `UPDATE management_mtls_mappings
SET workload_class=$3,source_assured_at=$4,status=$5,revision=revision+1,updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2 RETURNING revision`, updated.ID, expected, updated.WorkloadClass,
			updated.SourceAssuredAt, updated.Status).Scan(&revision); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
			}
			return managementidentity.WorkloadMutationResult{}, err
		}
		sessions, patchMTLSMappingErr := revokeSourceSessions(ctx, tx, managementauth.AuthSourceMTLS, []string{updated.ID})
		if patchMTLSMappingErr != nil {
			return managementidentity.WorkloadMutationResult{}, patchMTLSMappingErr
		}
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "mtls_identity_mapping.updated", ResourceType: "mtls_identity_mapping",
			ResourceID: updated.ID, BeforeRevision: &expected, AfterRevision: revision, Actor: actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: "mtls_identity_mapping", ID: updated.ID, Revision: revision,
			HTTPStatus: 200, SessionIDs: sessions,
		}, nil
	})
}

func (store *Store) DeleteMTLSMapping(
	ctx context.Context,
	id string,
	expected uint64,
	actor managementidentity.MutationActor,
) (managementidentity.WorkloadMutationResult, error) {
	if !canonicalUUID(id) || expected == 0 {
		return managementidentity.WorkloadMutationResult{}, managementidentity.ErrInvalidWorkloadRequest
	}
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (managementidentity.WorkloadMutationResult, error) {
		mapping, err := scanMTLSMapping(tx.QueryRowContext(ctx,
			`SELECT `+mtlsMappingColumns+` FROM management_mtls_mappings WHERE id=$1 FOR UPDATE`, id))
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrNotFound
		}
		if err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if mapping.Revision != expected {
			return managementidentity.WorkloadMutationResult{}, managementidentity.ErrRevisionConflict
		}
		sessions, err := revokeSourceSessions(ctx, tx, managementauth.AuthSourceMTLS, []string{id})
		if err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM management_mtls_mappings WHERE id=$1`, id); err != nil {
			return managementidentity.WorkloadMutationResult{}, mapWorkloadDependency("delete Management mTLS identity mapping", err)
		}
		after := expected + 1
		if err := appendAudit(ctx, tx, auditMutation{
			Action: "mtls_identity_mapping.deleted", ResourceType: "mtls_identity_mapping",
			ResourceID: id, BeforeRevision: &expected, AfterRevision: after, Actor: actor,
		}); err != nil {
			return managementidentity.WorkloadMutationResult{}, err
		}
		return managementidentity.WorkloadMutationResult{
			Kind: "mtls_identity_mapping", ID: id, Revision: after,
			HTTPStatus: 204, SessionIDs: sessions,
		}, nil
	})
}

func (store *Store) ResolveMTLSIdentity(
	ctx context.Context,
	matcherKind string,
	matcherValue string,
	now time.Time,
) (managementidentity.VerifiedMTLSMapping, error) {
	if store == nil || store.database == nil || matcherKind == "" || matcherValue == "" || now.IsZero() {
		return managementidentity.VerifiedMTLSMapping{}, managementauth.ErrAuthenticationDenied
	}
	var mapping managementidentity.VerifiedMTLSMapping
	var mappingStatus, principalStatus string
	if err := store.database.QueryRowContext(ctx, `SELECT mapping.id::text,mapping.principal_id::text,
       mapping.workload_class,mapping.source_assured_at,mapping.status,principal.status
FROM management_mtls_mappings mapping
JOIN management_principals principal ON principal.id=mapping.principal_id
WHERE mapping.matcher_kind=$1 AND mapping.matcher_value=$2`, matcherKind, matcherValue).Scan(
		&mapping.MappingID, &mapping.PrincipalID, &mapping.WorkloadClass, &mapping.SourceAssuredAt,
		&mappingStatus, &principalStatus,
	); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return managementidentity.VerifiedMTLSMapping{}, managementauth.ErrAuthenticationDenied
		}
		return managementidentity.VerifiedMTLSMapping{}, managementauth.ErrAuthenticationUnavailable
	}
	if !canonicalUUID(mapping.MappingID) || !canonicalUUID(mapping.PrincipalID) ||
		(mapping.WorkloadClass != string(managementidentity.WorkloadStandard) && mapping.WorkloadClass != string(managementidentity.WorkloadStrong)) ||
		mappingStatus != string(managementauth.ResourceActive) || principalStatus != "active" ||
		mapping.SourceAssuredAt.IsZero() || mapping.SourceAssuredAt.After(now) {
		return managementidentity.VerifiedMTLSMapping{}, managementauth.ErrAuthenticationDenied
	}
	mapping.SourceAssuredAt = mapping.SourceAssuredAt.UTC()
	return mapping, nil
}

func scanMTLSMapping(scanner scanner) (managementidentity.MTLSIdentityMapping, error) {
	var mapping managementidentity.MTLSIdentityMapping
	if err := scanner.Scan(&mapping.ID, &mapping.MatcherKind, &mapping.MatcherValue,
		&mapping.PrincipalID, &mapping.WorkloadClass, &mapping.SourceAssuredAt,
		&mapping.Status, &mapping.Revision, &mapping.CreatedAt, &mapping.UpdatedAt); err != nil {
		return managementidentity.MTLSIdentityMapping{}, err
	}
	if !canonicalUUID(mapping.ID) || !canonicalUUID(mapping.PrincipalID) || mapping.MatcherValue == "" ||
		(mapping.MatcherKind != managementidentity.MTLSMatcherSPIFFEID && mapping.MatcherKind != managementidentity.MTLSMatcherSANURI &&
			mapping.MatcherKind != managementidentity.MTLSMatcherSANDNS && mapping.MatcherKind != managementidentity.MTLSMatcherSubjectDNDigest) ||
		(mapping.WorkloadClass != managementidentity.WorkloadStandard && mapping.WorkloadClass != managementidentity.WorkloadStrong) ||
		(mapping.Status != managementauth.ResourceActive && mapping.Status != managementauth.ResourceDisabled) ||
		mapping.Revision == 0 || mapping.SourceAssuredAt.IsZero() {
		return managementidentity.MTLSIdentityMapping{}, errors.New("stored Management mTLS identity mapping is invalid")
	}
	mapping.SourceAssuredAt = mapping.SourceAssuredAt.UTC()
	mapping.CreatedAt, mapping.UpdatedAt = mapping.CreatedAt.UTC(), mapping.UpdatedAt.UTC()
	return mapping, nil
}

func mtlsMappingResult(
	mapping managementidentity.MTLSIdentityMapping,
	err error,
) (managementidentity.MTLSIdentityMapping, error) {
	if errors.Is(err, sql.ErrNoRows) {
		return managementidentity.MTLSIdentityMapping{}, managementidentity.ErrNotFound
	}
	if err != nil {
		return managementidentity.MTLSIdentityMapping{}, fmt.Errorf("load Management mTLS identity mapping: %w", err)
	}
	return mapping, nil
}

func replayWorkloadMutation(
	stored managementcommand.StoredResult,
	kind string,
) (managementidentity.WorkloadMutationResult, error) {
	if stored.Resource == nil || stored.Operation != nil || stored.Secret != nil || stored.Resource.ResourceType != kind {
		return managementidentity.WorkloadMutationResult{}, managementcommand.ErrConflict
	}
	return managementidentity.WorkloadMutationResult{
		Kind: stored.Resource.ResourceType, ID: stored.Resource.ResourceID,
		Revision: stored.Resource.ResourceRevision, HTTPStatus: stored.Resource.ResponseStatus, Replayed: true,
	}, nil
}
