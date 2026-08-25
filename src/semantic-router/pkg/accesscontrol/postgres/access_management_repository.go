package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

var _ accessmanagement.Repository = (*Store)(nil)

func (s *Store) Ready(ctx context.Context) error {
	if s == nil || s.db == nil {
		return accessmanagement.ErrUnavailable
	}
	if err := s.db.PingContext(ctx); err != nil {
		return fmt.Errorf("%w: PostgreSQL access state: %w", accessmanagement.ErrUnavailable, err)
	}
	return nil
}

func (s *Store) LoadPolicySnapshot(
	ctx context.Context,
	namespaceID string,
	subject accessmanagement.Subject,
) (accessmanagement.PolicySnapshot, error) {
	if err := validateIdentityIDs(accesscontrol.NamespaceID(namespaceID), subject.ID); err != nil || subject.Validate() != nil {
		return accessmanagement.PolicySnapshot{}, accessmanagement.ErrInvalidRequest
	}
	return inReadTransaction(ctx, s, func(tx *sql.Tx) (accessmanagement.PolicySnapshot, error) {
		return loadAccessManagementSnapshot(ctx, tx, accesscontrol.NamespaceID(namespaceID), subject)
	})
}

func loadAccessManagementSnapshot(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	subject accessmanagement.Subject,
) (accessmanagement.PolicySnapshot, error) {
	namespace, loadAccessManagementSnapshotErr := scanNamespace(tx.QueryRowContext(ctx, getNamespaceQuery, namespaceID))
	if errors.Is(loadAccessManagementSnapshotErr, sql.ErrNoRows) {
		return accessmanagement.PolicySnapshot{}, accessmanagement.ErrNotFound
	}
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, fmt.Errorf("load effective-policy namespace: %w", loadAccessManagementSnapshotErr)
	}
	desired, applied, revisionTime, loadAccessManagementSnapshotErr := loadPolicyRevisionState(ctx, tx, namespaceID)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	candidate, subjectRevision, layerSubjects, claimSubjectIDs, loadAccessManagementSnapshotErr := loadSubjectCandidateBase(ctx, tx, namespace, subject)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	accessBindings, rateBindings, loadAccessManagementSnapshotErr := loadSubjectBindings(ctx, tx, namespace, claimSubjectIDs)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	candidate.Revision = desired
	candidate.KeyAccessBindings = accessBindings[layerSubjectID(layerSubjects.Key)]
	candidate.UserAccessBindings = accessBindings[layerSubjectID(layerSubjects.User)]
	candidate.TeamAccessBindings = accessBindings[layerSubjectID(layerSubjects.Team)]
	candidate.KeyRateBindings = rateBindings[layerSubjectID(layerSubjects.Key)]
	candidate.UserRateBindings = rateBindings[layerSubjectID(layerSubjects.User)]
	candidate.TeamRateBindings = rateBindings[layerSubjectID(layerSubjects.Team)]
	candidate.AccessPolicies, loadAccessManagementSnapshotErr = loadScopedAccessPolicies(ctx, tx, accessBindings)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	candidate.RatePolicies, loadAccessManagementSnapshotErr = loadScopedRatePolicies(ctx, tx, rateBindings)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	schema, claims, loadAccessManagementSnapshotErr := loadScopedRoutingClaims(ctx, tx, namespaceID, claimSubjectIDs)
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, loadAccessManagementSnapshotErr
	}
	if err := accessmanagement.ValidateSchema(schema); err != nil {
		return accessmanagement.PolicySnapshot{}, fmt.Errorf("stored routing claim schema: %w", err)
	}
	context := compileRoutingContext(subject, subjectRevision, schema, claims, layerSubjects)
	candidate.RoutingClaims = effectiveClaimValues(context.Effective)
	projection, loadAccessManagementSnapshotErr := accessprojection.Compile(candidate, accessprojection.CompileOptions{
		CalendarScheduleStart: revisionTime.UTC().Truncate(time.Millisecond),
	})
	if loadAccessManagementSnapshotErr != nil {
		return accessmanagement.PolicySnapshot{}, fmt.Errorf("compile effective policy: %w", loadAccessManagementSnapshotErr)
	}
	return accessmanagement.PolicySnapshot{
		NamespaceID: string(namespace.ID), QuotaPartition: string(namespace.QuotaPartitionID),
		BillingCurrency: namespace.BillingCurrency, Subject: subject, SubjectRevision: subjectRevision,
		DesiredRevision: desired, AppliedRevision: applied, RevisionTime: revisionTime,
		Projection: projection, LayerSubjects: layerSubjects, Schema: schema, Context: context,
	}, nil
}

func loadPolicyRevisionState(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (uint64, uint64, time.Time, error) {
	var desired, applied int64
	if err := tx.QueryRowContext(ctx, `SELECT
  COALESCE((SELECT MAX(revision) FROM policy_revisions WHERE namespace_id = $1), 0),
  COALESCE((SELECT MAX(routing_revision) FROM routing_snapshots WHERE namespace_id = $1 AND status = 'active'), 0)`,
		namespaceID).Scan(&desired, &applied); err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("read effective-policy revisions: %w", err)
	}
	if desired <= 0 || applied < 0 || applied > desired {
		return 0, 0, time.Time{}, accessmanagement.ErrUnavailable
	}
	var revisionTime time.Time
	if err := tx.QueryRowContext(ctx, `SELECT created_at FROM policy_revisions WHERE namespace_id = $1 AND revision = $2`,
		namespaceID, desired).Scan(&revisionTime); err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("read effective-policy revision time: %w", err)
	}
	return uint64(desired), uint64(applied), revisionTime.UTC(), nil
}

func loadSubjectCandidateBase(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	subject accessmanagement.Subject,
) (accessprojection.Candidate, uint64, accessmanagement.LayerSubjects, []string, error) {
	switch subject.Kind {
	case accesscontrol.SubjectKindAPIKey:
		key, err := scanAPIKey(tx.QueryRowContext(ctx, getAPIKeyQuery, namespace.ID, subject.ID))
		if errors.Is(err, sql.ErrNoRows) || (err == nil && key.DeletedAt != nil) {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrNotFound
		}
		if err != nil {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, fmt.Errorf("load effective API key: %w", err)
		}
		relationships, layers, ids, err := loadKeyRelationships(ctx, tx, key)
		if err != nil {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, err
		}
		return accessprojection.Candidate{Namespace: namespace, Key: key, Relationships: relationships},
			uint64(key.Revision), layers, ids, nil
	case accesscontrol.SubjectKindUser:
		record, err := scanUser(tx.QueryRowContext(ctx, getUserQuery, namespace.ID, subject.ID))
		if errors.Is(err, sql.ErrNoRows) || (err == nil && record.DeletedAt != nil) {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrNotFound
		}
		if err != nil {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, fmt.Errorf("load effective user: %w", err)
		}
		if record.User.Status != accesscontrol.UserStatusActive {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrNotFound
		}
		key := syntheticUserKey(namespace, record.User)
		userSubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindUser, ID: subject.ID}
		layers := accessmanagement.LayerSubjects{User: &userSubject}
		return accessprojection.Candidate{
				Namespace: namespace, Key: key,
				Relationships: accesscontrol.APIKeyRelationships{OwnerUser: &record.User},
			},
			uint64(record.Revision), layers, []string{subject.ID}, nil
	case accesscontrol.SubjectKindTeam:
		record, err := scanTeam(tx.QueryRowContext(ctx, getTeamQuery, namespace.ID, subject.ID))
		if errors.Is(err, sql.ErrNoRows) || (err == nil && record.DeletedAt != nil) {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrNotFound
		}
		if err != nil {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, fmt.Errorf("load effective team: %w", err)
		}
		if record.Team.Status != accesscontrol.TeamStatusActive {
			return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrNotFound
		}
		key := syntheticTeamKey(namespace, record.Team)
		teamSubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindTeam, ID: subject.ID}
		layers := accessmanagement.LayerSubjects{Team: &teamSubject}
		return accessprojection.Candidate{
				Namespace: namespace, Key: key,
				Relationships: accesscontrol.APIKeyRelationships{OwnerTeam: &record.Team},
			},
			uint64(record.Revision), layers, []string{subject.ID}, nil
	default:
		return accessprojection.Candidate{}, 0, accessmanagement.LayerSubjects{}, nil, accessmanagement.ErrInvalidRequest
	}
}

func loadKeyRelationships(
	ctx context.Context,
	tx *sql.Tx,
	key accesscontrol.APIKey,
) (accesscontrol.APIKeyRelationships, accessmanagement.LayerSubjects, []string, error) {
	keySubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindAPIKey, ID: string(key.ID)}
	layers := accessmanagement.LayerSubjects{Key: &keySubject}
	ids := []string{string(key.ID)}
	relationships := accesscontrol.APIKeyRelationships{}
	switch key.Owner.Kind {
	case accesscontrol.SubjectKindUser:
		owner, err := scanUser(tx.QueryRowContext(ctx, getUserQuery, key.NamespaceID, key.Owner.ID))
		if err != nil || owner.DeletedAt != nil {
			return relationships, layers, nil, accessmanagement.ErrNotFound
		}
		relationships.OwnerUser = &owner.User
		userSubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindUser, ID: string(owner.User.ID)}
		layers.User = &userSubject
		ids = append(ids, userSubject.ID)
		if key.ContextTeamID != "" {
			team, teamErr := scanTeam(tx.QueryRowContext(ctx, getTeamQuery, key.NamespaceID, key.ContextTeamID))
			membership, membershipErr := scanMembership(tx.QueryRowContext(ctx, getMembershipQuery,
				key.NamespaceID, key.ContextTeamID, owner.User.ID))
			if teamErr != nil || membershipErr != nil || team.DeletedAt != nil {
				return relationships, layers, nil, accessmanagement.ErrNotFound
			}
			relationships.ContextTeam, relationships.ContextMembership = &team.Team, &membership.Membership
			teamSubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindTeam, ID: string(team.Team.ID)}
			layers.Team = &teamSubject
			ids = append(ids, teamSubject.ID)
		}
	case accesscontrol.SubjectKindTeam:
		owner, err := scanTeam(tx.QueryRowContext(ctx, getTeamQuery, key.NamespaceID, key.Owner.ID))
		if err != nil || owner.DeletedAt != nil {
			return relationships, layers, nil, accessmanagement.ErrNotFound
		}
		relationships.OwnerTeam = &owner.Team
		teamSubject := accessmanagement.Subject{Kind: accesscontrol.SubjectKindTeam, ID: string(owner.Team.ID)}
		layers.Team = &teamSubject
		ids = append(ids, teamSubject.ID)
	default:
		return relationships, layers, nil, accessmanagement.ErrUnavailable
	}
	if err := accesscontrol.ValidateAPIKeyRelationships(key, relationships); err != nil {
		return relationships, layers, nil, accessmanagement.ErrNotFound
	}
	return relationships, layers, ids, nil
}

func syntheticUserKey(namespace accesscontrol.Namespace, user accesscontrol.User) accesscontrol.APIKey {
	return accesscontrol.APIKey{
		ID: accesscontrol.APIKeyID(user.ID), NamespaceID: namespace.ID,
		Name: "effective-user-policy", Owner: user.SubjectRef(), Status: accesscontrol.APIKeyStatusActive,
		PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1, CreatedAt: user.CreatedAt, UpdatedAt: user.UpdatedAt,
	}
}

func syntheticTeamKey(namespace accesscontrol.Namespace, team accesscontrol.Team) accesscontrol.APIKey {
	return accesscontrol.APIKey{
		ID: accesscontrol.APIKeyID(team.ID), NamespaceID: namespace.ID,
		Name: "effective-team-policy", Owner: team.SubjectRef(), Status: accesscontrol.APIKeyStatusActive,
		PolicyEpoch: 1, DelegationEpoch: 1, Revision: 1, CreatedAt: team.CreatedAt, UpdatedAt: team.UpdatedAt,
	}
}

func layerSubjectID(subject *accessmanagement.Subject) string {
	if subject == nil {
		return ""
	}
	return subject.ID
}

func loadSubjectBindings(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	subjectIDs []string,
) (map[string][]accesscontrol.AccessPolicyBinding, map[string][]accesscontrol.RateLimitBinding, error) {
	accessBindings := make(map[string][]accesscontrol.AccessPolicyBinding)
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT b.id, b.namespace_id, b.subject_id, s.kind,
 b.policy_id, b.status, b.revision
FROM access_policy_bindings b JOIN access_subjects s
 ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND b.subject_id = ANY($2)
ORDER BY b.subject_id,b.id`, namespace.ID, pq.Array(subjectIDs))
	if queryContextErr != nil {
		return nil, nil, fmt.Errorf("load scoped access bindings: %w", queryContextErr)
	}
	for rows.Next() {
		binding, scanErr := scanAccessPolicyBinding(rows)
		if scanErr != nil {
			rows.Close()
			return nil, nil, scanErr
		}
		accessBindings[string(binding.Subject.ID)] = append(accessBindings[string(binding.Subject.ID)], binding)
	}
	if err := rows.Close(); err != nil {
		return nil, nil, err
	}
	rateBindings := make(map[string][]accesscontrol.RateLimitBinding)
	rows, queryContextErr = tx.QueryContext(ctx, `SELECT b.id, b.namespace_id, b.subject_id, s.kind,
 b.policy_id, b.binding_mode, b.quota_partition_id, b.status, b.revision
FROM rate_limit_bindings b JOIN access_subjects s
 ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
WHERE b.namespace_id=$1 AND b.subject_id = ANY($2)
ORDER BY b.subject_id,b.id`, namespace.ID, pq.Array(subjectIDs))
	if queryContextErr != nil {
		return nil, nil, fmt.Errorf("load scoped rate bindings: %w", queryContextErr)
	}
	defer rows.Close()
	for rows.Next() {
		binding, scanErr := scanRateLimitBinding(rows)
		if scanErr != nil {
			return nil, nil, scanErr
		}
		if binding.QuotaPartitionID != namespace.QuotaPartitionID {
			return nil, nil, accessmanagement.ErrUnavailable
		}
		rateBindings[string(binding.Subject.ID)] = append(rateBindings[string(binding.Subject.ID)], binding)
	}
	return accessBindings, rateBindings, rows.Err()
}

func loadScopedAccessPolicies(
	ctx context.Context,
	tx *sql.Tx,
	bindings map[string][]accesscontrol.AccessPolicyBinding,
) (map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy, error) {
	ids := make(map[accesscontrol.AccessPolicyID]struct{})
	for _, values := range bindings {
		for _, binding := range values {
			if binding.Status == accesscontrol.BindingStatusActive {
				ids[binding.PolicyID] = struct{}{}
			}
		}
	}
	result := make(map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy, len(ids))
	for id := range ids {
		policy, err := scanAccessPolicy(tx.QueryRowContext(ctx, getAccessPolicyQuery,
			bindingsNamespace(bindings), id))
		if err != nil {
			return nil, fmt.Errorf("load scoped access policy %s: %w", id, err)
		}
		policy.Grants, err = listAccessPolicyGrants(ctx, tx, id)
		if err != nil {
			return nil, err
		}
		result[id] = policy
	}
	return result, nil
}

func loadScopedRatePolicies(
	ctx context.Context,
	tx *sql.Tx,
	bindings map[string][]accesscontrol.RateLimitBinding,
) (map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy, error) {
	ids := make(map[accesscontrol.RateLimitPolicyID]struct{})
	for _, values := range bindings {
		for _, binding := range values {
			if binding.Status == accesscontrol.BindingStatusActive {
				ids[binding.PolicyID] = struct{}{}
			}
		}
	}
	result := make(map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy, len(ids))
	for id := range ids {
		policy, err := scanRateLimitPolicy(tx.QueryRowContext(ctx, getRateLimitPolicyQuery,
			bindingsRateNamespace(bindings), id))
		if err != nil {
			return nil, fmt.Errorf("load scoped rate policy %s: %w", id, err)
		}
		policy.Rules, err = listRateLimitRules(ctx, tx, id)
		if err != nil {
			return nil, err
		}
		result[id] = policy
	}
	return result, nil
}

func bindingsNamespace(bindings map[string][]accesscontrol.AccessPolicyBinding) accesscontrol.NamespaceID {
	for _, values := range bindings {
		if len(values) != 0 {
			return values[0].NamespaceID
		}
	}
	return ""
}

func bindingsRateNamespace(bindings map[string][]accesscontrol.RateLimitBinding) accesscontrol.NamespaceID {
	for _, values := range bindings {
		if len(values) != 0 {
			return values[0].NamespaceID
		}
	}
	return ""
}

type storedRoutingClaim struct {
	Value     routingsnapshot.ClaimValue
	Revision  uint64
	UpdatedAt time.Time
}

func loadScopedRoutingClaims(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	subjectIDs []string,
) (accessmanagement.RoutingClaimSchema, map[string]map[string]storedRoutingClaim, error) {
	schema := accessmanagement.RoutingClaimSchema{Definitions: make(map[string]accessmanagement.ClaimDefinition)}
	var definitions []byte
	var revision int64
	err := tx.QueryRowContext(ctx, `SELECT definitions,revision FROM routing_claim_schemas WHERE namespace_id=$1`, namespaceID).
		Scan(&definitions, &revision)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return schema, nil, fmt.Errorf("load routing claim schema: %w", err)
	}
	if err == nil {
		if revision <= 0 || json.Unmarshal(definitions, &schema.Definitions) != nil {
			return schema, nil, accessmanagement.ErrUnavailable
		}
		schema.Revision = uint64(revision)
	}
	rows, err := tx.QueryContext(ctx, `SELECT subject_id,claim_name,value,revision,updated_at
FROM routing_subject_claims WHERE namespace_id=$1 AND subject_id = ANY($2)
ORDER BY subject_id,claim_name`, namespaceID, pq.Array(subjectIDs))
	if err != nil {
		return schema, nil, fmt.Errorf("load scoped routing claims: %w", err)
	}
	defer rows.Close()
	result := make(map[string]map[string]storedRoutingClaim)
	for rows.Next() {
		var subjectID, name string
		var raw []byte
		var claim storedRoutingClaim
		var claimRevision int64
		if err := rows.Scan(&subjectID, &name, &raw, &claimRevision, &claim.UpdatedAt); err != nil {
			return schema, nil, err
		}
		if claimRevision <= 0 || json.Unmarshal(raw, &claim.Value) != nil || claim.Value.Validate() != nil {
			return schema, nil, accessmanagement.ErrUnavailable
		}
		claim.Revision = uint64(claimRevision)
		if result[subjectID] == nil {
			result[subjectID] = make(map[string]storedRoutingClaim)
		}
		result[subjectID][name] = claim
	}
	return schema, result, rows.Err()
}

func compileRoutingContext(
	subject accessmanagement.Subject,
	revision uint64,
	schema accessmanagement.RoutingClaimSchema,
	claims map[string]map[string]storedRoutingClaim,
	layers accessmanagement.LayerSubjects,
) accessmanagement.RoutingContext {
	result := accessmanagement.RoutingContext{Subject: subject, Revision: revision, SchemaRevision: schema.Revision}
	for name, claim := range claims[subject.ID] {
		result.Stored = append(result.Stored, accessmanagement.StoredClaim{
			Name: name, Value: claim.Value,
			Revision: claim.Revision, UpdatedAt: claim.UpdatedAt.UTC(),
		})
	}
	sort.Slice(result.Stored, func(i, j int) bool { return result.Stored[i].Name < result.Stored[j].Name })
	resolved := make(map[string]accessmanagement.EffectiveClaim)
	for _, source := range []*accessmanagement.Subject{layers.Team, layers.User, layers.Key} {
		if source == nil {
			continue
		}
		for name, claim := range claims[source.ID] {
			resolved[name] = accessmanagement.EffectiveClaim{StoredClaim: accessmanagement.StoredClaim{
				Name: name, Value: claim.Value, Revision: claim.Revision, UpdatedAt: claim.UpdatedAt.UTC(),
			}, Source: *source}
		}
	}
	for _, claim := range resolved {
		result.Effective = append(result.Effective, claim)
	}
	sort.Slice(result.Effective, func(i, j int) bool { return result.Effective[i].Name < result.Effective[j].Name })
	return result
}

func effectiveClaimValues(claims []accessmanagement.EffectiveClaim) map[string]routingsnapshot.ClaimValue {
	if len(claims) == 0 {
		return nil
	}
	result := make(map[string]routingsnapshot.ClaimValue, len(claims))
	for _, claim := range claims {
		result[claim.Name] = claim.Value
	}
	return result
}

func (s *Store) ResourceExists(
	ctx context.Context,
	namespaceID string,
	resource accesscontrol.GrantResource,
) (bool, error) {
	if validateUUID("namespace id", namespaceID) != nil || resource.Validate() != nil {
		return false, accessmanagement.ErrInvalidRequest
	}
	const modelExistsQuery = `SELECT EXISTS(SELECT 1 FROM routing_models
	WHERE namespace_id=$1 AND id=$2 AND status='active' AND deleted_at IS NULL)`
	const entrypointExistsQuery = `SELECT EXISTS(SELECT 1 FROM routing_entrypoints
	WHERE namespace_id=$1 AND id=$2 AND status='active' AND deleted_at IS NULL)`
	var query string
	switch resource.Type {
	case accesscontrol.GrantResourceModel:
		query = modelExistsQuery
	case accesscontrol.GrantResourceEntrypoint:
		query = entrypointExistsQuery
	default:
		return false, accessmanagement.ErrInvalidRequest
	}
	var exists bool
	if err := s.db.QueryRowContext(ctx, query, namespaceID, resource.ID).Scan(&exists); err != nil {
		return false, fmt.Errorf("load access-check resource: %w", err)
	}
	return exists, nil
}

func (s *Store) UpdateRoutingContext(
	ctx context.Context,
	request accessmanagement.UpdateRoutingContextRequest,
) (accessmanagement.RoutingContextMutation, error) {
	if request.ExpectedRevision == 0 || request.Subject.Validate() != nil ||
		validateIdentityIDs(accesscontrol.NamespaceID(request.NamespaceID), request.Subject.ID) != nil {
		return accessmanagement.RoutingContextMutation{}, accessmanagement.ErrInvalidRequest
	}
	return inTransaction(ctx, s, func(tx *sql.Tx) (accessmanagement.RoutingContextMutation, error) {
		schema, _, updateRoutingContextErr := loadScopedRoutingClaims(ctx, tx, accesscontrol.NamespaceID(request.NamespaceID), []string{request.Subject.ID})
		if updateRoutingContextErr != nil {
			return accessmanagement.RoutingContextMutation{}, updateRoutingContextErr
		}
		if err := accessmanagement.ValidateContextValues(schema, request.Values); err != nil {
			return accessmanagement.RoutingContextMutation{}, err
		}
		newRevision, updateRoutingContextErr := advanceRoutingContextSubject(ctx, tx, request)
		if updateRoutingContextErr != nil {
			return accessmanagement.RoutingContextMutation{}, updateRoutingContextErr
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM routing_subject_claims WHERE namespace_id=$1 AND subject_id=$2`,
			request.NamespaceID, request.Subject.ID); err != nil {
			return accessmanagement.RoutingContextMutation{}, fmt.Errorf("replace routing context: %w", err)
		}
		names := make([]string, 0, len(request.Values))
		for name := range request.Values {
			names = append(names, name)
		}
		sort.Strings(names)
		for _, name := range names {
			encoded, encodeErr := json.Marshal(request.Values[name])
			if encodeErr != nil {
				return accessmanagement.RoutingContextMutation{}, accessmanagement.ErrInvalidRequest
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_subject_claims
 (namespace_id,subject_id,claim_name,value,revision,updated_at)
VALUES ($1,$2,$3,$4,$5,clock_timestamp())`, request.NamespaceID, request.Subject.ID, name, encoded, newRevision); err != nil {
				return accessmanagement.RoutingContextMutation{}, fmt.Errorf("insert routing context claim: %w", err)
			}
		}
		actor, actorErr := routingContextMutationMeta(request.Actor, request.Subject, len(request.Values))
		if actorErr != nil {
			return accessmanagement.RoutingContextMutation{}, actorErr
		}
		receipt, updateRoutingContextErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(request.NamespaceID), outboxMutation{
			AggregateType: "routing_context", AggregateID: request.Subject.ID,
			AggregateRevision: accesscontrol.Revision(newRevision), Operation: outboxUpdated,
			References: map[string]string{"subjectType": string(request.Subject.Kind)},
		}, actor)
		if updateRoutingContextErr != nil {
			return accessmanagement.RoutingContextMutation{}, updateRoutingContextErr
		}
		var partition string
		if err := tx.QueryRowContext(ctx, `SELECT quota_partition_id FROM access_namespaces WHERE id=$1`, request.NamespaceID).
			Scan(&partition); err != nil {
			return accessmanagement.RoutingContextMutation{}, fmt.Errorf("read routing context partition: %w", err)
		}
		return accessmanagement.RoutingContextMutation{DesiredRevision: uint64(receipt.DesiredRevision), QuotaPartition: partition}, nil
	})
}

func advanceRoutingContextSubject(
	ctx context.Context,
	tx *sql.Tx,
	request accessmanagement.UpdateRoutingContextRequest,
) (uint64, error) {
	const advanceUserRevisionQuery = `UPDATE access_users SET revision=revision+1,updated_at=clock_timestamp()
	WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL RETURNING revision`
	const advanceTeamRevisionQuery = `UPDATE access_teams SET revision=revision+1,updated_at=clock_timestamp()
	WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL RETURNING revision`
	const advanceRoutingContextAPIKeyRevisionQuery = `UPDATE access_api_keys SET revision=revision+1,updated_at=clock_timestamp()
	WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL RETURNING revision`
	var query string
	switch request.Subject.Kind {
	case accesscontrol.SubjectKindUser:
		query = advanceUserRevisionQuery
	case accesscontrol.SubjectKindTeam:
		query = advanceTeamRevisionQuery
	case accesscontrol.SubjectKindAPIKey:
		query = advanceRoutingContextAPIKeyRevisionQuery
	default:
		return 0, accessmanagement.ErrInvalidRequest
	}
	var revision int64
	if err := tx.QueryRowContext(ctx, query, request.NamespaceID, request.Subject.ID, request.ExpectedRevision).Scan(&revision); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return 0, accessmanagement.ErrRevisionConflict
		}
		return 0, fmt.Errorf("advance routing context subject revision: %w", err)
	}
	if revision <= 0 {
		return 0, accessmanagement.ErrUnavailable
	}
	revisionValue, err := positiveUint64(revision, "routing-context subject revision")
	if err != nil {
		return 0, accessmanagement.ErrUnavailable
	}
	return revisionValue, nil
}

func routingContextMutationMeta(
	actor accessmanagement.Actor,
	subject accessmanagement.Subject,
	claimCount int,
) (MutationMeta, error) {
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index, value := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(value)
	}
	return MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: "routing_context.updated", Reason: "update routing context",
		Details: AuditDetails{"subjectType": string(subject.Kind), "claimCount": fmt.Sprintf("%d", claimCount)},
	}, nil
}
