package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/namespacemanagement"
)

const (
	getManagedNamespaceQuery = `SELECT id, name, quota_partition_id, billing_currency, status,
       revision, runtime_epoch, created_at, updated_at
FROM access_namespaces WHERE id = $1`
	listManagedNamespacesQuery = `SELECT id, name, quota_partition_id, billing_currency, status,
       revision, runtime_epoch, created_at, updated_at
FROM access_namespaces
WHERE ($1 = '' OR status = $1)
  AND ($2 OR id = ANY($3::uuid[]))
  AND ($4::timestamptz IS NULL OR (created_at, id) < ($4::timestamptz, $5::uuid))
ORDER BY created_at DESC, id DESC
LIMIT $6`
	insertManagedNamespaceQuery = `INSERT INTO access_namespaces
  (id,name,quota_partition_id,billing_currency,status,revision,runtime_epoch,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`
	insertManagedSelfServiceQuery = `INSERT INTO self_service_policies
  (namespace_id,max_keys_per_user,max_delegated_sessions,delegated_session_ttl_seconds,
   allow_team_key_delegation,automatic_first_key,team_admin_capabilities,
   default_access_policy_id,default_rate_limit_policy_id,revision,seed_version,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,NULLIF($8,'')::uuid,NULLIF($9,'')::uuid,$10,$11,$12)`
	insertManagedSecurityQuery = `INSERT INTO management_security_policies
  (namespace_id,action_requirements,seed_version,revision,updated_at)
VALUES ($1,$2,$3,$4,$5)`
	insertManagedClaimsQuery = `INSERT INTO routing_claim_schemas
  (namespace_id,definitions,revision,updated_at) VALUES ($1,$2,$3,$4)`
	updateManagedNamespaceQuery = `UPDATE access_namespaces
SET status=$3, revision=revision+1, updated_at=clock_timestamp()
WHERE id=$1 AND revision=$2
RETURNING id,name,quota_partition_id,billing_currency,status,revision,runtime_epoch,created_at,updated_at`
	getManagedSelfServiceQuery = `SELECT namespace_id,max_keys_per_user,max_delegated_sessions,
       delegated_session_ttl_seconds,allow_team_key_delegation,automatic_first_key,
       team_admin_capabilities,default_access_policy_id,default_rate_limit_policy_id,
       revision,seed_version,updated_at
FROM self_service_policies WHERE namespace_id=$1`
	updateManagedSelfServiceQuery = `UPDATE self_service_policies SET
  max_keys_per_user=$3,max_delegated_sessions=$4,delegated_session_ttl_seconds=$5,
  allow_team_key_delegation=$6,automatic_first_key=$7,team_admin_capabilities=$8,
  default_access_policy_id=NULLIF($9,'')::uuid,default_rate_limit_policy_id=NULLIF($10,'')::uuid,
  revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND revision=$2
RETURNING namespace_id,max_keys_per_user,max_delegated_sessions,delegated_session_ttl_seconds,
          allow_team_key_delegation,automatic_first_key,team_admin_capabilities,
          default_access_policy_id,default_rate_limit_policy_id,revision,seed_version,updated_at`
	getManagedSecurityQuery = `SELECT namespace_id,action_requirements,seed_version,revision,updated_at
FROM management_security_policies WHERE namespace_id=$1`
	updateManagedSecurityQuery = `UPDATE management_security_policies SET
  action_requirements=$3,seed_version=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND revision=$2
RETURNING namespace_id,action_requirements,seed_version,revision,updated_at`
	getManagedClaimsQuery = `SELECT namespace_id,definitions,revision,updated_at
FROM routing_claim_schemas WHERE namespace_id=$1`
	updateManagedClaimsQuery = `UPDATE routing_claim_schemas SET
  definitions=$3,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND revision=$2
RETURNING namespace_id,definitions,revision,updated_at`
	activeNamespaceCompanionsQuery = `SELECT namespace.id,
       self_service.namespace_id IS NOT NULL,
       security.namespace_id IS NOT NULL,
       claims.namespace_id IS NOT NULL
FROM access_namespaces AS namespace
LEFT JOIN self_service_policies AS self_service ON self_service.namespace_id=namespace.id
LEFT JOIN management_security_policies AS security ON security.namespace_id=namespace.id
LEFT JOIN routing_claim_schemas AS claims ON claims.namespace_id=namespace.id
WHERE namespace.status='active'`
	activeNamespaceDependenciesQuery = `SELECT EXISTS (
  SELECT 1 FROM access_users WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM access_teams WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM access_api_keys WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM access_policies WHERE namespace_id=$1 AND status IN ('draft','active')
  UNION ALL SELECT 1 FROM rate_limit_policies WHERE namespace_id=$1 AND status IN ('draft','active')
  UNION ALL SELECT 1 FROM provider_credentials WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM routing_models WHERE namespace_id=$1 AND status IN ('draft','active')
  UNION ALL SELECT 1 FROM routing_recipes AS recipe
    WHERE recipe.namespace_id=$1 AND recipe.status IN ('draft','active')
      AND NOT EXISTS (SELECT 1 FROM routing_recipe_provenance AS provenance
                      WHERE provenance.namespace_id=recipe.namespace_id AND provenance.recipe_id=recipe.id)
  UNION ALL SELECT 1 FROM routing_entrypoints WHERE namespace_id=$1 AND status IN ('draft','active')
  UNION ALL SELECT 1 FROM management_service_accounts WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM management_role_bindings WHERE namespace_id=$1 AND status='active'
  UNION ALL SELECT 1 FROM management_invitations WHERE namespace_id=$1 AND status='pending'
)`
	validateDefaultAccessPolicyQuery = `SELECT EXISTS(SELECT 1 FROM access_policies
WHERE id=$1 AND namespace_id=$2 AND status='active')`
	validateDefaultRatePolicyQuery = `SELECT EXISTS(SELECT 1 FROM rate_limit_policies
WHERE id=$1 AND namespace_id=$2 AND status='active')`
)

type namespaceManagementRepository struct{ store *Store }

func NewNamespaceManagementRepository(store *Store) (namespacemanagement.Repository, error) {
	if store == nil || store.db == nil {
		return nil, namespacemanagement.ErrUnavailable
	}
	return &namespaceManagementRepository{store: store}, nil
}

func (repository *namespaceManagementRepository) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	if repository == nil || repository.store == nil || repository.store.db == nil || codec == nil {
		return namespacemanagement.ErrUnavailable
	}
	if err := repository.store.db.PingContext(ctx); err != nil {
		return err
	}
	if err := commandpostgres.ValidateReferencedHMACVersions(ctx, repository.store.db, codec); err != nil {
		return err
	}
	rows, err := repository.store.db.QueryContext(ctx, activeNamespaceCompanionsQuery)
	if err != nil {
		return fmt.Errorf("verify active Namespace companions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var id string
		var selfService, security, claims bool
		if err := rows.Scan(&id, &selfService, &security, &claims); err != nil {
			return err
		}
		if !selfService || !security || !claims {
			return fmt.Errorf("active Namespace %s is missing a mandatory companion", id)
		}
	}
	return rows.Err()
}

func (repository *namespaceManagementRepository) Replay(ctx context.Context, command managementcommand.Command) (namespacemanagement.MutationResult, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, repository.store.db, command)
	if err != nil || !found {
		return namespacemanagement.MutationResult{}, found, mapNamespaceCommandError(err)
	}
	result, err := namespaceMutationResult(stored)
	return result, true, err
}

func (repository *namespaceManagementRepository) GetNamespace(ctx context.Context, id string) (namespacemanagement.Namespace, error) {
	value, err := scanManagedNamespace(repository.store.db.QueryRowContext(ctx, getManagedNamespaceQuery, id))
	return value, mapNamespaceReadError(err)
}

func (repository *namespaceManagementRepository) ListNamespaces(ctx context.Context, query namespacemanagement.NamespaceQuery) (namespacemanagement.RepositoryPage[namespacemanagement.Namespace], error) {
	var afterTime any
	var afterID any
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	rows, err := repository.store.db.QueryContext(ctx, listManagedNamespacesQuery, query.Status, query.Scope.All,
		pq.Array(query.Scope.NamespaceIDs), afterTime, afterID, query.Limit+1)
	if err != nil {
		return namespacemanagement.RepositoryPage[namespacemanagement.Namespace]{}, err
	}
	defer rows.Close()
	items := make([]namespacemanagement.Namespace, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanManagedNamespace(rows)
		if err != nil {
			return namespacemanagement.RepositoryPage[namespacemanagement.Namespace]{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return namespacemanagement.RepositoryPage[namespacemanagement.Namespace]{}, err
	}
	more := len(items) > query.Limit
	if more {
		items = items[:query.Limit]
	}
	return namespacemanagement.RepositoryPage[namespacemanagement.Namespace]{Items: items, HasMore: more}, nil
}

func (repository *namespaceManagementRepository) CreateNamespace(ctx context.Context, mutation namespacemanagement.CreateNamespaceMutation) (namespacemanagement.MutationResult, error) {
	security, err := json.Marshal(mutation.Security.ActionRequirements)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	capabilities, err := json.Marshal(mutation.SelfService.TeamAdminCapabilities)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	definitions, err := json.Marshal(mutation.RoutingClaims.Definitions)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	meta, err := namespaceMutationMeta(mutation.Actor, "namespace.create")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return withSerializableRetry(ctx, repository.store.db, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		if replay, found, err := lockNamespaceCommand(ctx, tx, mutation.Command); err != nil || found {
			return replay, err
		}
		namespace := mutation.Namespace
		if _, err := tx.ExecContext(ctx, insertManagedNamespaceQuery, namespace.ID, namespace.Name,
			namespace.QuotaPartitionID, namespace.BillingCurrency, namespace.Status, namespace.Revision,
			namespace.RuntimeEpoch, namespace.CreatedAt, namespace.UpdatedAt); err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCreateError(err)
		}
		self := mutation.SelfService
		if _, err := tx.ExecContext(ctx, insertManagedSelfServiceQuery, self.NamespaceID, self.MaxKeysPerUser,
			self.MaxDelegatedSessions, int64(self.DelegatedSessionTTL/time.Second), self.AllowTeamKeyDelegation,
			self.AutomaticFirstKey, capabilities, self.DefaultAccessPolicyID, self.DefaultRateLimitPolicyID,
			self.Revision, self.SeedVersion, self.UpdatedAt); err != nil {
			return namespacemanagement.MutationResult{}, fmt.Errorf("insert Namespace self-service policy: %w", err)
		}
		if _, err := tx.ExecContext(ctx, insertManagedSecurityQuery, mutation.Security.NamespaceID, security,
			mutation.Security.SeedVersion, mutation.Security.Revision, mutation.Security.UpdatedAt); err != nil {
			return namespacemanagement.MutationResult{}, fmt.Errorf("insert Namespace security policy: %w", err)
		}
		if _, err := tx.ExecContext(ctx, insertManagedClaimsQuery, mutation.RoutingClaims.NamespaceID,
			definitions, mutation.RoutingClaims.Revision, mutation.RoutingClaims.UpdatedAt); err != nil {
			return namespacemanagement.MutationResult{}, fmt.Errorf("insert Namespace routing claim schema: %w", err)
		}
		if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespace.ID), outboxMutation{
			AggregateType: "namespace", AggregateID: namespace.ID,
			AggregateRevision: accesscontrol.Revision(namespace.Revision), Operation: outboxCreated,
		}, meta); err != nil {
			return namespacemanagement.MutationResult{}, err
		}
		return completeNamespaceCommand(ctx, tx, mutation.Command, "namespace", namespace.ID, namespace.Revision, 201)
	})
}

func (repository *namespaceManagementRepository) PatchNamespace(ctx context.Context, namespace namespacemanagement.Namespace, expected uint64, actor namespacemanagement.Actor) (namespacemanagement.MutationResult, error) {
	meta, err := namespaceMutationMeta(actor, "namespace.update")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return inTransaction(ctx, repository.store, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		updated, err := scanManagedNamespace(tx.QueryRowContext(ctx, updateManagedNamespaceQuery, namespace.ID, expected, namespace.Status))
		if err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCASErr(err)
		}
		return appendNamespaceMutation(ctx, tx, updated.ID, "namespace", updated.Revision, outboxUpdated, meta, 200)
	})
}

func (repository *namespaceManagementRepository) DeleteNamespace(ctx context.Context, id string, expected uint64, actor namespacemanagement.Actor) (namespacemanagement.MutationResult, error) {
	meta, err := namespaceMutationMeta(actor, "namespace.delete")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return inTransaction(ctx, repository.store, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		var dependencies bool
		if err := tx.QueryRowContext(ctx, activeNamespaceDependenciesQuery, id).Scan(&dependencies); err != nil {
			return namespacemanagement.MutationResult{}, err
		}
		if dependencies {
			return namespacemanagement.MutationResult{}, namespacemanagement.ErrDependency
		}
		updated, err := scanManagedNamespace(tx.QueryRowContext(ctx, updateManagedNamespaceQuery, id, expected, accesscontrol.NamespaceStatusDisabled))
		if err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCASErr(err)
		}
		return appendNamespaceMutation(ctx, tx, updated.ID, "namespace", updated.Revision, outboxDeleted, meta, 204)
	})
}

func (repository *namespaceManagementRepository) GetSelfServicePolicy(ctx context.Context, namespaceID string) (namespacemanagement.SelfServicePolicy, error) {
	value, err := scanManagedSelfService(repository.store.db.QueryRowContext(ctx, getManagedSelfServiceQuery, namespaceID))
	return value, mapNamespaceReadError(err)
}

func (repository *namespaceManagementRepository) PatchSelfServicePolicy(ctx context.Context, policy namespacemanagement.SelfServicePolicy, expected uint64, actor namespacemanagement.Actor) (namespacemanagement.MutationResult, error) {
	capabilities, err := json.Marshal(policy.TeamAdminCapabilities)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	meta, err := namespaceMutationMeta(actor, "namespace.self_service_policy.update")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return inTransaction(ctx, repository.store, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		if err := validateNamespacePolicyReference(ctx, tx, validateDefaultAccessPolicyQuery, policy.DefaultAccessPolicyID, policy.NamespaceID); err != nil {
			return namespacemanagement.MutationResult{}, err
		}
		if err := validateNamespacePolicyReference(ctx, tx, validateDefaultRatePolicyQuery, policy.DefaultRateLimitPolicyID, policy.NamespaceID); err != nil {
			return namespacemanagement.MutationResult{}, err
		}
		updated, err := scanManagedSelfService(tx.QueryRowContext(ctx, updateManagedSelfServiceQuery,
			policy.NamespaceID, expected, policy.MaxKeysPerUser, policy.MaxDelegatedSessions,
			int64(policy.DelegatedSessionTTL/time.Second), policy.AllowTeamKeyDelegation,
			policy.AutomaticFirstKey, capabilities, policy.DefaultAccessPolicyID, policy.DefaultRateLimitPolicyID))
		if err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCASErr(err)
		}
		return appendNamespaceMutation(ctx, tx, updated.NamespaceID, "self_service_policy", updated.Revision, outboxUpdated, meta, 200)
	})
}

func (repository *namespaceManagementRepository) GetManagementSecurityPolicy(ctx context.Context, namespaceID string) (namespacemanagement.ManagementSecurityPolicy, error) {
	value, err := scanManagedSecurity(repository.store.db.QueryRowContext(ctx, getManagedSecurityQuery, namespaceID))
	return value, mapNamespaceReadError(err)
}

func (repository *namespaceManagementRepository) PatchManagementSecurityPolicy(ctx context.Context, policy namespacemanagement.ManagementSecurityPolicy, expected uint64, actor namespacemanagement.Actor) (namespacemanagement.MutationResult, error) {
	requirements, err := json.Marshal(policy.ActionRequirements)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	meta, err := namespaceMutationMeta(actor, "namespace.management_security_policy.update")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return inTransaction(ctx, repository.store, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		updated, err := scanManagedSecurity(tx.QueryRowContext(ctx, updateManagedSecurityQuery,
			policy.NamespaceID, expected, requirements, policy.SeedVersion))
		if err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCASErr(err)
		}
		return appendNamespaceMutation(ctx, tx, updated.NamespaceID, "management_security_policy", updated.Revision, outboxUpdated, meta, 200)
	})
}

func (repository *namespaceManagementRepository) GetRoutingClaimSchema(ctx context.Context, namespaceID string) (namespacemanagement.RoutingClaimSchema, error) {
	value, err := scanManagedClaims(repository.store.db.QueryRowContext(ctx, getManagedClaimsQuery, namespaceID))
	return value, mapNamespaceReadError(err)
}

func (repository *namespaceManagementRepository) PatchRoutingClaimSchema(ctx context.Context, schema namespacemanagement.RoutingClaimSchema, expected uint64, actor namespacemanagement.Actor) (namespacemanagement.MutationResult, error) {
	definitions, err := json.Marshal(schema.Definitions)
	if err != nil {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrInvalidRequest
	}
	meta, err := namespaceMutationMeta(actor, "namespace.routing_claim_schema.update")
	if err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return inTransaction(ctx, repository.store, func(tx *sql.Tx) (namespacemanagement.MutationResult, error) {
		updated, err := scanManagedClaims(tx.QueryRowContext(ctx, updateManagedClaimsQuery, schema.NamespaceID, expected, definitions))
		if err != nil {
			return namespacemanagement.MutationResult{}, mapNamespaceCASErr(err)
		}
		return appendNamespaceMutation(ctx, tx, updated.NamespaceID, "routing_claim_schema", updated.Revision, outboxUpdated, meta, 200)
	})
}

func scanManagedNamespace(scanner rowScanner) (namespacemanagement.Namespace, error) {
	var value namespacemanagement.Namespace
	if err := scanner.Scan(&value.ID, &value.Name, &value.QuotaPartitionID, &value.BillingCurrency,
		&value.Status, &value.Revision, &value.RuntimeEpoch, &value.CreatedAt, &value.UpdatedAt); err != nil {
		return value, err
	}
	value.CreatedAt, value.UpdatedAt = value.CreatedAt.UTC(), value.UpdatedAt.UTC()
	return value, nil
}

func scanManagedSelfService(scanner rowScanner) (namespacemanagement.SelfServicePolicy, error) {
	var value namespacemanagement.SelfServicePolicy
	var ttl int64
	var capabilities []byte
	var accessPolicy, ratePolicy sql.NullString
	if err := scanner.Scan(&value.NamespaceID, &value.MaxKeysPerUser, &value.MaxDelegatedSessions, &ttl,
		&value.AllowTeamKeyDelegation, &value.AutomaticFirstKey, &capabilities, &accessPolicy, &ratePolicy,
		&value.Revision, &value.SeedVersion, &value.UpdatedAt); err != nil {
		return value, err
	}
	if err := json.Unmarshal(capabilities, &value.TeamAdminCapabilities); err != nil {
		return value, err
	}
	value.DelegatedSessionTTL, value.UpdatedAt = time.Duration(ttl)*time.Second, value.UpdatedAt.UTC()
	if accessPolicy.Valid {
		value.DefaultAccessPolicyID = accessPolicy.String
	}
	if ratePolicy.Valid {
		value.DefaultRateLimitPolicyID = ratePolicy.String
	}
	return value, nil
}

func scanManagedSecurity(scanner rowScanner) (namespacemanagement.ManagementSecurityPolicy, error) {
	var value namespacemanagement.ManagementSecurityPolicy
	var requirements []byte
	if err := scanner.Scan(&value.NamespaceID, &requirements, &value.SeedVersion, &value.Revision, &value.UpdatedAt); err != nil {
		return value, err
	}
	if err := json.Unmarshal(requirements, &value.ActionRequirements); err != nil {
		return value, err
	}
	value.UpdatedAt = value.UpdatedAt.UTC()
	return value, nil
}

func scanManagedClaims(scanner rowScanner) (namespacemanagement.RoutingClaimSchema, error) {
	var value namespacemanagement.RoutingClaimSchema
	var definitions []byte
	if err := scanner.Scan(&value.NamespaceID, &definitions, &value.Revision, &value.UpdatedAt); err != nil {
		return value, err
	}
	if err := json.Unmarshal(definitions, &value.Definitions); err != nil {
		return value, err
	}
	value.UpdatedAt = value.UpdatedAt.UTC()
	if err := accessmanagement.ValidateSchema(accessmanagement.RoutingClaimSchema{Revision: value.Revision, Definitions: value.Definitions}); err != nil {
		return value, err
	}
	return value, nil
}

func namespaceMutationMeta(actor namespacemanagement.Actor, action string) (MutationMeta, error) {
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index, id := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(id)
	}
	meta := MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: action, Reason: actor.Reason, Details: AuditDetails{},
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationMeta{}, namespacemanagement.ErrInvalidRequest
	}
	return meta, nil
}

func appendNamespaceMutation(ctx context.Context, tx *sql.Tx, namespaceID, kind string, revision uint64,
	operation outboxOperation, meta MutationMeta, status int,
) (namespacemanagement.MutationResult, error) {
	if _, err := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(namespaceID), outboxMutation{
		AggregateType: kind, AggregateID: namespaceID, AggregateRevision: accesscontrol.Revision(revision), Operation: operation,
	}, meta); err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return namespacemanagement.MutationResult{Kind: kind, ID: namespaceID, Revision: revision, HTTPStatus: status}, nil
}

func validateNamespacePolicyReference(ctx context.Context, tx *sql.Tx, query, policyID, namespaceID string) error {
	if policyID == "" {
		return nil
	}
	var found bool
	if err := tx.QueryRowContext(ctx, query, policyID, namespaceID).Scan(&found); err != nil {
		return err
	}
	if !found {
		return namespacemanagement.ErrInvalidRequest
	}
	return nil
}

func lockNamespaceCommand(ctx context.Context, tx *sql.Tx, command managementcommand.Command) (namespacemanagement.MutationResult, bool, error) {
	stored, replayed, err := commandpostgres.Lock(ctx, tx, command)
	if err != nil || !replayed {
		return namespacemanagement.MutationResult{}, false, mapNamespaceCommandError(err)
	}
	result, err := namespaceMutationResult(stored)
	return result, true, err
}

func completeNamespaceCommand(ctx context.Context, tx *sql.Tx, command managementcommand.Command,
	kind, id string, revision uint64, status int,
) (namespacemanagement.MutationResult, error) {
	if err := commandpostgres.CompleteResource(ctx, tx, command, managementcommand.ResourceResult{
		ResourceType: kind, ResourceID: id, ResourceRevision: revision, ResponseStatus: status,
	}); err != nil {
		return namespacemanagement.MutationResult{}, err
	}
	return namespacemanagement.MutationResult{Kind: kind, ID: id, Revision: revision, HTTPStatus: status}, nil
}

func namespaceMutationResult(stored managementcommand.StoredResult) (namespacemanagement.MutationResult, error) {
	if stored.Resource == nil || stored.Resource.ResourceType != "namespace" {
		return namespacemanagement.MutationResult{}, namespacemanagement.ErrUnavailable
	}
	resource := stored.Resource
	return namespacemanagement.MutationResult{
		Kind: resource.ResourceType, ID: resource.ResourceID,
		Revision: resource.ResourceRevision, Replayed: true, HTTPStatus: resource.ResponseStatus,
	}, nil
}

func mapNamespaceReadError(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return namespacemanagement.ErrNotFound
	}
	return err
}

func mapNamespaceCreateError(err error) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return namespacemanagement.ErrAlreadyExists
	}
	return err
}

func mapNamespaceCASErr(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return namespacemanagement.ErrRevisionConflict
	}
	return mapNamespaceCreateError(err)
}

func mapNamespaceCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return namespacemanagement.ErrIdempotencyConflict
	}
	return err
}

func withSerializableRetry[T any](ctx context.Context, database *sql.DB, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	for attempt := 0; attempt < 3; attempt++ {
		tx, err := database.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
		if err != nil {
			return zero, err
		}
		value, operationErr := operation(tx)
		if operationErr != nil {
			_ = tx.Rollback()
			if retryableNamespaceTransaction(operationErr) {
				continue
			}
			return zero, operationErr
		}
		if err := tx.Commit(); err != nil {
			if retryableNamespaceTransaction(err) {
				continue
			}
			return zero, err
		}
		return value, nil
	}
	return zero, namespacemanagement.ErrUnavailable
}

func retryableNamespaceTransaction(err error) bool {
	var databaseError *pq.Error
	return errors.As(err, &databaseError) && (databaseError.Code == "40001" || databaseError.Code == "40P01")
}

var _ namespacemanagement.Repository = (*namespaceManagementRepository)(nil)
