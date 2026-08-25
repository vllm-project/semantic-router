package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
)

func selectAgentInferenceKey(
	ctx context.Context, tx *sql.Tx, namespaceID, principalID, keyID, effectiveTeamID string, _ string,
	target resolvedAgentTarget,
) (delegationmanagement.EligibleKey, error) {
	if uuid.Validate(keyID) != nil {
		return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
	}
	var team any
	if effectiveTeamID != "" {
		if uuid.Validate(effectiveTeamID) != nil {
			return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
		}
		team = effectiveTeamID
	}
	return queryAgentInferenceKey(
		ctx, tx, namespaceID, principalID, team, true, keyID, target,
		[]accesscontrol.GrantPermission{
			accesscontrol.GrantPermissionDiscover, accesscontrol.GrantPermissionInvoke,
		}, true,
	)
}

func selectAgentInferenceKeyRead(
	ctx context.Context, tx *sql.Tx, namespaceID, principalID, keyID, effectiveTeamID string, _ string,
	target resolvedAgentTarget,
) (delegationmanagement.EligibleKey, error) {
	if uuid.Validate(keyID) != nil {
		return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
	}
	var team any
	if effectiveTeamID != "" {
		if uuid.Validate(effectiveTeamID) != nil {
			return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
		}
		team = effectiveTeamID
	}
	return queryAgentInferenceKey(
		ctx, tx, namespaceID, principalID, team, true, keyID, target,
		[]accesscontrol.GrantPermission{
			accesscontrol.GrantPermissionDiscover, accesscontrol.GrantPermissionInvoke,
		}, false,
	)
}

func queryAgentInferenceKey(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	principalID string,
	team any,
	requireExactTeam bool,
	keyID any,
	target resolvedAgentTarget,
	permissions []accesscontrol.GrantPermission,
	lock bool,
) (delegationmanagement.EligibleKey, error) {
	values := make([]string, len(permissions))
	for index, permission := range permissions {
		if !permission.Valid() {
			return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
		}
		values[index] = string(permission)
	}
	if len(values) == 0 {
		return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
	}
	statement := eligibleAgentKeyForTargetSQL
	if lock {
		statement += ` FOR UPDATE OF k,l,u,p`
	}
	key, err := scanEligibleKey(tx.QueryRowContext(
		ctx, statement, namespaceID, principalID, team, requireExactTeam,
		target.Kind, target.ResourceID, pq.Array(values), keyID,
	))
	if errors.Is(err, sql.ErrNoRows) {
		return delegationmanagement.EligibleKey{}, agentmanagement.ErrNotFound
	}
	if err != nil {
		return delegationmanagement.EligibleKey{}, fmt.Errorf("select Agent inference key: %w", err)
	}
	return key, nil
}

func keyCanAccessTarget(
	ctx context.Context, tx *sql.Tx, namespaceID string, key delegationmanagement.EligibleKey,
	userID string, target resolvedAgentTarget, permission accesscontrol.GrantPermission,
) (bool, error) {
	subjects := []string{key.KeyID, userID}
	if key.TeamID != "" {
		subjects = append(subjects, key.TeamID)
	}
	var allowed bool
	err := tx.QueryRowContext(ctx, `SELECT COALESCE((
  SELECT EXISTS (
    SELECT 1
    FROM access_policy_bindings binding
    JOIN access_policies policy ON policy.namespace_id=binding.namespace_id AND policy.id=binding.policy_id
    JOIN access_policy_grants grant_record ON grant_record.policy_id=policy.id
    WHERE binding.namespace_id=$1 AND binding.subject_id=effective_access.subject_id
      AND binding.status='active' AND policy.status='active'
      AND grant_record.resource_type=$3 AND grant_record.resource_id=$4
      AND grant_record.permission=$5 AND grant_record.effect='allow'
  ) AND NOT EXISTS (
    SELECT 1
    FROM access_policy_bindings binding
    JOIN access_policies policy ON policy.namespace_id=binding.namespace_id AND policy.id=binding.policy_id
    JOIN access_policy_grants grant_record ON grant_record.policy_id=policy.id
    WHERE binding.namespace_id=$1 AND binding.subject_id=effective_access.subject_id
      AND binding.status='active' AND policy.status='active'
      AND grant_record.resource_type=$3 AND grant_record.resource_id=$4
      AND grant_record.permission=$5 AND grant_record.effect='deny'
  )
  FROM (
    SELECT candidate.subject_id
    FROM unnest($2::uuid[]) WITH ORDINALITY AS candidate(subject_id,priority)
    WHERE EXISTS (
      SELECT 1 FROM access_policy_bindings inheritance_binding
      WHERE inheritance_binding.namespace_id=$1
        AND inheritance_binding.subject_id=candidate.subject_id
        AND inheritance_binding.status='active'
    )
    ORDER BY candidate.priority LIMIT 1
  ) effective_access
),false)`, namespaceID, pq.Array(subjects), target.Kind, target.ResourceID, permission).Scan(&allowed)
	if err != nil {
		return false, fmt.Errorf("evaluate Agent target access: %w", err)
	}
	return allowed, nil
}

func resolveAgentTarget(
	ctx context.Context, tx *sql.Tx, namespaceID string, target agentmanagement.Target,
) (resolvedAgentTarget, error) {
	if target.ID == "" || strings.TrimSpace(target.ID) != target.ID {
		return resolvedAgentTarget{}, agentmanagement.ErrInvalid
	}
	var statement string
	switch target.Kind {
	case agentmanagement.TargetModel:
		statement = `SELECT id::text,name FROM routing_models
WHERE namespace_id=$1 AND status='active' AND (id::text=$2 OR name=$2 OR aliases ? $2)
ORDER BY CASE WHEN id::text=$2 THEN 0 WHEN name=$2 THEN 1 ELSE 2 END,id LIMIT 2`
	case agentmanagement.TargetEntrypoint:
		statement = `SELECT id::text,name FROM routing_entrypoints
WHERE namespace_id=$1 AND status='active' AND published_revision IS NOT NULL
  AND (id::text=$2 OR name=$2 OR aliases ? $2)
ORDER BY CASE WHEN id::text=$2 THEN 0 WHEN name=$2 THEN 1 ELSE 2 END,id LIMIT 2`
	default:
		return resolvedAgentTarget{}, agentmanagement.ErrInvalid
	}
	rows, err := tx.QueryContext(ctx, statement, namespaceID, target.ID)
	if err != nil {
		return resolvedAgentTarget{}, fmt.Errorf("resolve Agent target: %w", err)
	}
	defer rows.Close()
	matches := make([]resolvedAgentTarget, 0, 2)
	for rows.Next() {
		match := resolvedAgentTarget{Kind: target.Kind}
		if err := rows.Scan(&match.ResourceID, &match.PublicID); err != nil {
			return resolvedAgentTarget{}, fmt.Errorf("scan Agent target: %w", err)
		}
		matches = append(matches, match)
	}
	if err := rows.Err(); err != nil {
		return resolvedAgentTarget{}, fmt.Errorf("iterate Agent target: %w", err)
	}
	// Missing and ambiguous request-facing identifiers are intentionally
	// indistinguishable. A session pins exactly one canonical public name and
	// immutable resource UUID; it never retains a mutable alias.
	if len(matches) != 1 {
		return resolvedAgentTarget{}, agentmanagement.ErrNotFound
	}
	return matches[0], nil
}

func verifyTargetCapabilities(
	ctx context.Context, tx *sql.Tx, namespaceID string, target resolvedAgentTarget, required []string,
) error {
	if len(required) == 0 {
		return nil
	}
	var satisfied bool
	var err error
	if target.Kind == agentmanagement.TargetModel {
		err = tx.QueryRowContext(ctx, `SELECT mr.capabilities ?& $3::text[]
FROM routing_models m JOIN routing_model_revisions mr ON mr.model_id=m.id AND mr.revision=m.current_revision
WHERE m.namespace_id=$1 AND m.id=$2 AND m.status='active'`, namespaceID, target.ResourceID,
			pq.Array(required)).Scan(&satisfied)
	} else {
		err = tx.QueryRowContext(ctx, `SELECT EXISTS (
  SELECT 1 FROM routing_assignment_models assignment
  WHERE assignment.entrypoint_id=e.id AND assignment.entrypoint_revision=e.published_revision
) AND NOT EXISTS (
  SELECT 1 FROM routing_assignment_models assignment
  JOIN routing_model_revisions model
    ON model.model_id=assignment.model_id AND model.revision=assignment.model_revision
  WHERE assignment.entrypoint_id=e.id AND assignment.entrypoint_revision=e.published_revision
    AND NOT (model.capabilities ?& $3::text[])
)
FROM routing_entrypoints e
WHERE e.namespace_id=$1 AND e.id=$2 AND e.status='active' AND e.published_revision IS NOT NULL`,
			namespaceID, target.ResourceID, pq.Array(required)).Scan(&satisfied)
	}
	if errors.Is(err, sql.ErrNoRows) || !satisfied {
		return agentmanagement.ErrNotFound
	}
	if err != nil {
		return fmt.Errorf("verify Agent target capabilities: %w", err)
	}
	return nil
}

func loadActiveAgentDelegation(
	ctx context.Context, tx *sql.Tx, session agentmanagement.Session,
) (delegationmanagement.Session, delegationmanagement.EligibleKey, error) {
	delegation, err := scanDelegatedSession(tx.QueryRowContext(ctx, `SELECT `+delegatedSessionColumns+`
FROM delegated_inference_sessions d JOIN access_namespaces n ON n.id=d.namespace_id
JOIN agent_sessions s ON s.namespace_id=d.namespace_id AND s.delegated_inference_session_id=d.id
JOIN management_sessions ms ON ms.id=d.management_session_id
JOIN management_principals principal ON principal.id=d.principal_id
JOIN access_users user_record ON user_record.namespace_id=d.namespace_id AND user_record.id=d.user_id
WHERE d.namespace_id=$1 AND s.id=$2 AND s.status='active' AND d.status='active'
  AND d.expires_at>clock_timestamp() AND ms.status='active' AND ms.expires_at>clock_timestamp()
  AND principal.status='active' AND user_record.status='active'
FOR UPDATE OF d,s`, session.NamespaceID, session.ID))
	if err != nil {
		return delegationmanagement.Session{}, delegationmanagement.EligibleKey{}, agentDenied(err)
	}
	key, err := lockEligibleKey(ctx, tx, session.NamespaceID, session.OwnerPrincipalID, delegation.APIKeyID)
	if err != nil || key.DelegationEpoch != delegation.DelegationEpoch || key.TeamID != delegation.TeamID {
		return delegationmanagement.Session{}, delegationmanagement.EligibleKey{}, agentmanagement.ErrDenied
	}
	if session.EffectiveUserID != "" && session.EffectiveUserID != delegation.UserID {
		return delegationmanagement.Session{}, delegationmanagement.EligibleKey{}, agentmanagement.ErrDenied
	}
	if session.EffectiveTeamID != delegation.TeamID {
		return delegationmanagement.Session{}, delegationmanagement.EligibleKey{}, agentmanagement.ErrDenied
	}
	return delegation, key, nil
}

func loadAgentSessionBootstrapReplay(
	ctx context.Context, tx *sql.Tx, namespaceID, sessionID string,
) (agentmanagement.Session, delegationmanagement.Session, uint64, error) {
	var session agentmanagement.Session
	var userID, teamID, targetModelID, targetEntrypointID sql.NullString
	err := tx.QueryRowContext(ctx, `SELECT id::text,namespace_id::text,owner_principal_id::text,
       effective_user_id::text,effective_team_id::text,delegated_inference_session_id::text,
       profile_id::text,profile_revision,target_model_id,target_entrypoint_id,target_public_id,authority_digest,
       mode,title,status,revision,created_at,updated_at
FROM agent_sessions WHERE namespace_id=$1 AND id=$2 AND status='active'`, namespaceID, sessionID).Scan(
		&session.ID, &session.NamespaceID, &session.OwnerPrincipalID, &userID, &teamID,
		&session.DelegatedInferenceSessionID, &session.ProfileID, &session.ProfileRevision,
		&targetModelID, &targetEntrypointID, &session.Target.ID, &session.AuthorityDigest,
		&session.Mode, &session.Title,
		&session.Status, &session.Revision, &session.CreatedAt, &session.UpdatedAt,
	)
	if err != nil {
		return agentmanagement.Session{}, delegationmanagement.Session{}, 0, agentNotFound(err)
	}
	if userID.Valid {
		session.EffectiveUserID = userID.String
	}
	if teamID.Valid {
		session.EffectiveTeamID = teamID.String
	}
	if targetModelID.Valid {
		session.Target.Kind, session.TargetResourceID = agentmanagement.TargetModel, targetModelID.String
	} else if targetEntrypointID.Valid {
		session.Target.Kind, session.TargetResourceID = agentmanagement.TargetEntrypoint, targetEntrypointID.String
	} else {
		return agentmanagement.Session{}, delegationmanagement.Session{}, 0, agentmanagement.ErrConflict
	}
	delegation, err := getDelegatedSessionTx(ctx, tx, namespaceID, session.DelegatedInferenceSessionID)
	if err != nil {
		return agentmanagement.Session{}, delegationmanagement.Session{}, 0, err
	}
	desired, err := latestAggregateDesiredRevision(ctx, tx, delegation.ID)
	return session, delegation, desired, err
}

func agentMutationMeta(
	mutation agentmanagement.MutationContext, action, reason string,
) (MutationMeta, error) {
	principal := accesscontrol.ManagementPrincipalID(mutation.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(mutation.ActorChain))
	for index, actor := range mutation.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(actor)
	}
	if len(chain) == 0 {
		chain = []accesscontrol.ManagementPrincipalID{principal}
	}
	meta := MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: mutation.RequestID,
		SourceIP: mutation.SourceIP, Action: action, Reason: reason, Details: AuditDetails{},
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationMeta{}, agentmanagement.ErrInvalid
	}
	return meta, nil
}

func agentNotFound(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return agentmanagement.ErrNotFound
	}
	return err
}

func agentDenied(err error) error {
	if errors.Is(err, sql.ErrNoRows) || errors.Is(err, delegationmanagement.ErrNotEligible) {
		return agentmanagement.ErrDenied
	}
	return err
}

func mapAgentWrite(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return agentmanagement.ErrConflict
	}
	var databaseError *pq.Error
	if errors.As(err, &databaseError) {
		switch databaseError.Code {
		case "23505", "40001", "40P01":
			return agentmanagement.ErrConflict
		case "23503", "23514", "22P02":
			return agentmanagement.ErrInvalid
		}
	}
	return err
}

var (
	_ agentmanagement.SessionAuthority = (*AgentSessionAuthority)(nil)
	_ agentmanagement.TargetVisibility = (*AgentSessionAuthority)(nil)
)
