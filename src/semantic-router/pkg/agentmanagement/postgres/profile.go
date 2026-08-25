package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func (store *Store) ListProfiles(
	ctx context.Context, namespaceID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.Profile], returnErr error) {
	ids := scopedIDs(query.Scope, accesscontrol.ScopeResourceAgentProfile)
	statement := profileCurrentSelect + `
 WHERE p.namespace_id=$1 AND p.status<>'deleted'
   AND ($2 OR p.id=ANY($3::uuid[]))
	AND ($4='' OR lower(p.name) LIKE $4 ESCAPE '\' OR lower(p.description) LIKE $4 ESCAPE '\')
	AND ($5::timestamptz IS NULL OR (p.created_at,p.id)<($5,$6::uuid))
 ORDER BY p.created_at DESC,p.id DESC LIMIT $7`
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.Timestamp, query.After.ID
	}
	rows, err := store.db.QueryContext(
		ctx, statement, namespaceID, query.Scope.All, pq.Array(ids),
		managementsearch.PrefixPattern(query.Search), afterTime, afterID, query.Limit,
	)
	if err != nil {
		return agentmanagement.ListResult[agentmanagement.Profile]{}, fmt.Errorf("list Agent Profiles: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Profile, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanProfile(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.Profile]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.Profile]{}, fmt.Errorf("iterate Agent Profiles: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.Profile]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetProfile(ctx context.Context, namespaceID, id string) (agentmanagement.Profile, error) {
	statement := profileCurrentSelect + `
 WHERE p.namespace_id=$1 AND p.id=$2 AND p.status<>'deleted'`
	return scanProfile(store.db.QueryRowContext(ctx, statement, namespaceID, id))
}

func (store *Store) GetProfileRevision(
	ctx context.Context, namespaceID, id string, revision int64,
) (agentmanagement.Profile, error) {
	statement := profileRevisionSelect + ` WHERE p.namespace_id=$1 AND p.id=$2`
	return scanProfile(store.db.QueryRowContext(ctx, statement, namespaceID, id, revision))
}

func (store *Store) GetDefaultProfile(
	ctx context.Context, namespaceID string, mode agentmanagement.SessionMode,
) (agentmanagement.Profile, error) {
	var id string
	var revision int64
	err := store.db.QueryRowContext(ctx, `SELECT defaults.profile_id::text,defaults.profile_revision
FROM agent_profile_defaults defaults
JOIN agent_profiles profile ON profile.namespace_id=defaults.namespace_id AND profile.id=defaults.profile_id
WHERE defaults.namespace_id=$1 AND defaults.mode=$2 AND profile.status='active'`,
		namespaceID, mode).Scan(&id, &revision)
	if err != nil {
		return agentmanagement.Profile{}, mapNotFound(err)
	}
	return store.GetProfileRevision(ctx, namespaceID, id, revision)
}

func (store *Store) CreateProfile(
	ctx context.Context, namespaceID, id string, input agentmanagement.ProfileInput,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentProfileResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		targetModel, targetEntrypoint, err := resolveTarget(ctx, tx, namespaceID, input.DefaultTarget)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		encoded, digest, err := encodeProfileRevision(input)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profiles
  (id,namespace_id,name,description,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'active',1,1)`, id, namespaceID, input.Name, input.Description); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_revisions
  (profile_id,namespace_id,revision,target_model_id,target_entrypoint_id,
   minimum_target_capabilities,supported_modes,tool_policy,approval_policy,
   maximum_turn_seconds,maximum_tool_steps,context_token_budget,content_digest,created_by)
VALUES ($1,$2,1,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13)`,
			id, namespaceID, targetModel, targetEntrypoint, encoded.minimumCapabilities,
			encoded.supportedModes, encoded.toolPolicy, input.ApprovalPolicy,
			input.MaximumTurnSeconds, input.MaximumToolSteps, input.ContextTokenBudget,
			digest[:], nullableString(mutation.Mutation.PrincipalID)); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if err := replaceProfileSkills(ctx, tx, namespaceID, id, 1, input.Skills); err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		if err := replaceProfileDefaults(ctx, tx, namespaceID, id, 1, input.DefaultForModes, mutation.Mutation.PrincipalID); err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		return completeResourceCommand(ctx, tx, mutation, agentProfileResourceType, id, 1, 201)
	})
}

func (store *Store) PatchProfile(
	ctx context.Context, namespaceID, id string, expected int64, patch agentmanagement.ProfilePatch,
	mutation agentmanagement.MutationContext,
) (agentmanagement.Profile, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Profile, error) {
		current, patchProfileErr := store.lockProfile(ctx, tx, namespaceID, id, expected)
		if patchProfileErr != nil {
			return agentmanagement.Profile{}, patchProfileErr
		}
		input := applyProfilePatch(current, patch)
		targetModel, targetEntrypoint, patchProfileErr := resolveTarget(ctx, tx, namespaceID, input.DefaultTarget)
		if patchProfileErr != nil {
			return agentmanagement.Profile{}, patchProfileErr
		}
		encoded, digest, patchProfileErr := encodeProfileRevision(input)
		if patchProfileErr != nil {
			return agentmanagement.Profile{}, patchProfileErr
		}
		contentRevision := current.ContentRevision + 1
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_revisions
  (profile_id,namespace_id,revision,target_model_id,target_entrypoint_id,
   minimum_target_capabilities,supported_modes,tool_policy,approval_policy,
   maximum_turn_seconds,maximum_tool_steps,context_token_budget,content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`,
			id, namespaceID, contentRevision, targetModel, targetEntrypoint,
			encoded.minimumCapabilities, encoded.supportedModes, encoded.toolPolicy,
			input.ApprovalPolicy, input.MaximumTurnSeconds, input.MaximumToolSteps,
			input.ContextTokenBudget, digest[:], nullableString(mutation.PrincipalID)); err != nil {
			return agentmanagement.Profile{}, classifyWriteError(err)
		}
		if err := replaceProfileSkills(ctx, tx, namespaceID, id, contentRevision, input.Skills); err != nil {
			return agentmanagement.Profile{}, err
		}
		if err := replaceProfileDefaults(ctx, tx, namespaceID, id, contentRevision, input.DefaultForModes, mutation.PrincipalID); err != nil {
			return agentmanagement.Profile{}, err
		}
		result, patchProfileErr := tx.ExecContext(ctx, `UPDATE agent_profiles
SET name=$4,description=$5,current_revision=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status<>'deleted'`,
			namespaceID, id, expected, input.Name, input.Description, contentRevision)
		if patchProfileErr != nil {
			return agentmanagement.Profile{}, classifyWriteError(patchProfileErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.Profile{}, err
		}
		statement := profileCurrentSelect + ` WHERE p.namespace_id=$1 AND p.id=$2`
		return scanProfile(tx.QueryRowContext(ctx, statement, namespaceID, id))
	})
}

func (store *Store) DeleteProfile(
	ctx context.Context, namespaceID, id string, expected int64, _ agentmanagement.MutationContext,
) (int64, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (int64, error) {
		result, err := tx.ExecContext(ctx, `UPDATE agent_profiles p
SET status='deleted',revision=revision+1,updated_at=clock_timestamp(),deleted_at=clock_timestamp()
WHERE p.namespace_id=$1 AND p.id=$2 AND p.revision=$3 AND p.status<>'deleted'
  AND NOT EXISTS (SELECT 1 FROM agent_profile_defaults d
                   WHERE d.namespace_id=p.namespace_id AND d.profile_id=p.id)
  AND NOT EXISTS (SELECT 1 FROM agent_sessions s
                   WHERE s.namespace_id=p.namespace_id AND s.profile_id=p.id AND s.status='active')`,
			namespaceID, id, expected)
		if err != nil {
			return 0, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return 0, err
		}
		return expected + 1, nil
	})
}

type encodedProfileRevision struct {
	minimumCapabilities []byte
	supportedModes      []byte
	toolPolicy          []byte
}

func encodeProfileRevision(input agentmanagement.ProfileInput) (encodedProfileRevision, [sha256.Size]byte, error) {
	minimum, err := json.Marshal(input.MinimumTargetCapabilities)
	if err != nil {
		return encodedProfileRevision{}, [sha256.Size]byte{}, err
	}
	modes, err := json.Marshal(input.SupportedModes)
	if err != nil {
		return encodedProfileRevision{}, [sha256.Size]byte{}, err
	}
	policy, err := json.Marshal(input.ToolPolicy)
	if err != nil {
		return encodedProfileRevision{}, [sha256.Size]byte{}, err
	}
	canonical, err := json.Marshal(input)
	if err != nil {
		return encodedProfileRevision{}, [sha256.Size]byte{}, err
	}
	return encodedProfileRevision{minimumCapabilities: minimum, supportedModes: modes, toolPolicy: policy}, sha256.Sum256(canonical), nil
}

func (store *Store) lockProfile(
	ctx context.Context, tx *sql.Tx, namespaceID, id string, expected int64,
) (agentmanagement.Profile, error) {
	var revision int64
	if err := tx.QueryRowContext(ctx, `SELECT revision FROM agent_profiles
WHERE namespace_id=$1 AND id=$2 AND status<>'deleted' FOR UPDATE`, namespaceID, id).Scan(&revision); err != nil {
		return agentmanagement.Profile{}, mapNotFound(err)
	}
	if revision != expected {
		return agentmanagement.Profile{}, agentmanagement.ErrConflict
	}
	statement := profileCurrentSelect + ` WHERE p.namespace_id=$1 AND p.id=$2`
	return scanProfile(tx.QueryRowContext(ctx, statement, namespaceID, id))
}

func applyProfilePatch(current agentmanagement.Profile, patch agentmanagement.ProfilePatch) agentmanagement.ProfileInput {
	input := agentmanagement.ProfileInput{
		Name: current.Name, Description: current.Description, DefaultTarget: current.DefaultTarget,
		MinimumTargetCapabilities: current.MinimumTargetCapabilities, SupportedModes: current.SupportedModes,
		DefaultForModes: current.DefaultForModes, Skills: current.Skills, ToolPolicy: current.ToolPolicy,
		ApprovalPolicy: current.ApprovalPolicy, MaximumTurnSeconds: current.MaximumTurnSeconds,
		MaximumToolSteps: current.MaximumToolSteps, ContextTokenBudget: current.ContextTokenBudget,
	}
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.DefaultTarget.Present {
		input.DefaultTarget = patch.DefaultTarget.Value
	}
	if patch.MinimumTargetCapabilities != nil {
		input.MinimumTargetCapabilities = *patch.MinimumTargetCapabilities
	}
	if patch.SupportedModes != nil {
		input.SupportedModes = *patch.SupportedModes
	}
	if patch.DefaultForModes != nil {
		input.DefaultForModes = *patch.DefaultForModes
	}
	if patch.Skills != nil {
		input.Skills = *patch.Skills
	}
	if patch.ToolPolicy != nil {
		input.ToolPolicy = *patch.ToolPolicy
	}
	if patch.ApprovalPolicy != nil {
		input.ApprovalPolicy = *patch.ApprovalPolicy
	}
	if patch.MaximumTurnSeconds != nil {
		input.MaximumTurnSeconds = *patch.MaximumTurnSeconds
	}
	if patch.MaximumToolSteps != nil {
		input.MaximumToolSteps = *patch.MaximumToolSteps
	}
	if patch.ContextTokenBudget != nil {
		input.ContextTokenBudget = *patch.ContextTokenBudget
	}
	return input
}

func replaceProfileSkills(
	ctx context.Context, tx *sql.Tx, namespaceID, profileID string, revision int64,
	skills []agentmanagement.SkillReference,
) error {
	for ordinal, skill := range skills {
		var skillNamespace sql.NullString
		if err := tx.QueryRowContext(ctx, `SELECT namespace_id::text FROM agent_skills
WHERE id=$1 AND status='active' AND (namespace_id=$2 OR namespace_id IS NULL)`,
			skill.ID, namespaceID).Scan(&skillNamespace); err != nil && err != sql.ErrNoRows {
			return fmt.Errorf("resolve Agent Skill: %w", err)
		} else if err == sql.ErrNoRows {
			return agentmanagement.ErrNotFound
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_skills
  (namespace_id,profile_id,profile_revision,ordinal,skill_id,skill_namespace_id,skill_revision)
VALUES ($1,$2,$3,$4,$5,$6,$7)`, namespaceID, profileID, revision, ordinal,
			skill.ID, nullableNullString(skillNamespace), skill.Revision); err != nil {
			return classifyWriteError(err)
		}
	}
	return nil
}

func replaceProfileDefaults(
	ctx context.Context, tx *sql.Tx, namespaceID, profileID string, revision int64,
	modes []agentmanagement.SessionMode, principalID string,
) error {
	if _, err := tx.ExecContext(ctx, `DELETE FROM agent_profile_defaults
WHERE namespace_id=$1 AND profile_id=$2`, namespaceID, profileID); err != nil {
		return fmt.Errorf("clear Agent Profile defaults: %w", err)
	}
	for _, mode := range modes {
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_defaults
  (namespace_id,mode,profile_id,profile_revision,updated_by)
VALUES ($1,$2,$3,$4,$5)
ON CONFLICT (namespace_id,mode) DO UPDATE SET
  profile_id=EXCLUDED.profile_id,profile_revision=EXCLUDED.profile_revision,
  updated_by=EXCLUDED.updated_by,updated_at=clock_timestamp()`,
			namespaceID, mode, profileID, revision, nullableString(principalID)); err != nil {
			return classifyWriteError(err)
		}
	}
	return nil
}

type sqlQueryer interface {
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

const resolveModelTargetQuery = `SELECT id FROM routing_models
WHERE namespace_id=$1 AND status='active' AND (id::text=$2 OR name=$2 OR aliases ? $2) LIMIT 2`

const resolveEntrypointTargetQuery = `SELECT id FROM routing_entrypoints
WHERE namespace_id=$1 AND status='active' AND published_revision IS NOT NULL
  AND (id::text=$2 OR name=$2 OR aliases ? $2) LIMIT 2`

func resolveTarget(
	ctx context.Context, queryer sqlQueryer, namespaceID string, target *agentmanagement.Target,
) (_ any, _ any, returnErr error) {
	if target == nil {
		return nil, nil, nil
	}
	var statement string
	switch target.Kind {
	case agentmanagement.TargetModel:
		statement = resolveModelTargetQuery
	case agentmanagement.TargetEntrypoint:
		statement = resolveEntrypointTargetQuery
	default:
		return nil, nil, agentmanagement.ErrInvalid
	}
	rows, err := queryer.QueryContext(ctx, statement, namespaceID, target.ID)
	if err != nil {
		return nil, nil, fmt.Errorf("resolve Agent target: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	ids := make([]string, 0, 2)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, nil, err
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	if len(ids) == 0 {
		return nil, nil, agentmanagement.ErrNotFound
	}
	if len(ids) != 1 {
		return nil, nil, agentmanagement.ErrConflict
	}
	if target.Kind == agentmanagement.TargetModel {
		return ids[0], nil, nil
	}
	return nil, ids[0], nil
}

func scopedIDs(scope accesscontrol.ResultScope, kind accesscontrol.ScopeResourceType) []string {
	values := scope.IDs(kind)
	result := make([]string, len(values))
	for index, value := range values {
		result[index] = string(value)
	}
	return result
}

func requireOneRow(result sql.Result) error {
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows != 1 {
		return agentmanagement.ErrConflict
	}
	return nil
}

func nullableString(value string) any {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func nullableNullString(value sql.NullString) any {
	if !value.Valid {
		return nil
	}
	return value.String
}
