package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

func (store *Store) ListSkills(
	ctx context.Context, namespaceID string, query agentmanagement.ListQuery,
) (_ agentmanagement.ListResult[agentmanagement.Skill], returnErr error) {
	ids := scopedIDs(query.Scope, accesscontrol.ScopeResourceAgentSkill)
	statement := skillSelect + `
 WHERE (s.namespace_id=$1 OR s.namespace_id IS NULL) AND s.status<>'deleted'
   AND (s.builtin OR $2 OR s.id=ANY($3::uuid[]))
	AND ($4='' OR lower(s.name) LIKE $4 ESCAPE '\' OR lower(s.description) LIKE $4 ESCAPE '\')
	AND ($5::timestamptz IS NULL OR (s.created_at,s.id)<($5,$6::uuid))
 ORDER BY s.created_at DESC,s.id DESC LIMIT $7`
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
		return agentmanagement.ListResult[agentmanagement.Skill]{}, fmt.Errorf("list Agent Skills: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]agentmanagement.Skill, 0, query.Limit)
	for rows.Next() {
		value, scanErr := scanSkill(rows)
		if scanErr != nil {
			return agentmanagement.ListResult[agentmanagement.Skill]{}, scanErr
		}
		items = append(items, value)
	}
	if err := rows.Err(); err != nil {
		return agentmanagement.ListResult[agentmanagement.Skill]{}, fmt.Errorf("iterate Agent Skills: %w", err)
	}
	return agentmanagement.ListResult[agentmanagement.Skill]{Items: items, HasMore: len(items) == query.Limit}, nil
}

func (store *Store) GetSkill(ctx context.Context, namespaceID, id string) (agentmanagement.Skill, error) {
	return scanSkill(store.db.QueryRowContext(ctx, skillSelect+`
 WHERE s.id=$2 AND (s.namespace_id=$1 OR s.namespace_id IS NULL) AND s.status<>'deleted'`, namespaceID, id))
}

// GetSkillRevision reads the exact immutable content pinned by an Agent
// Profile. It remains available after the mutable Skill root is edited,
// disabled, or deleted so an already-created Session cannot drift.
func (store *Store) GetSkillRevision(
	ctx context.Context, namespaceID, id string, revision int64,
) (agentmanagement.Skill, error) {
	return scanSkill(store.db.QueryRowContext(ctx, skillRevisionSelect+`
 WHERE s.id=$2 AND (s.namespace_id=$1 OR s.namespace_id IS NULL) AND r.revision=$3`,
		namespaceID, id, revision))
}

func (store *Store) CreateSkill(
	ctx context.Context, namespaceID, id string, input agentmanagement.SkillInput,
	mutation agentmanagement.ResourceCommand,
) (agentmanagement.ResourceMutationResult, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.ResourceMutationResult, error) {
		if replay, found, err := lockResourceCommand(
			ctx, tx, namespaceID, agentSkillResourceType, mutation,
		); err != nil || found {
			return replay, err
		}
		tools, capabilities, digest, err := encodeSkillRevision(input)
		if err != nil {
			return agentmanagement.ResourceMutationResult{}, err
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_skills
  (id,namespace_id,name,description,builtin,status,current_revision,revision)
VALUES ($1,$2,$3,$4,FALSE,'active',1,1)`, id, namespaceID, input.Name, input.Description); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_skill_revisions
  (skill_id,namespace_id,revision,instructions,required_tools,minimum_capabilities,content_digest,created_by)
VALUES ($1,$2,1,$3,$4,$5,$6,$7)`, id, namespaceID, input.Instructions, tools,
			capabilities, digest[:], nullableString(mutation.Mutation.PrincipalID)); err != nil {
			return agentmanagement.ResourceMutationResult{}, classifyWriteError(err)
		}
		return completeResourceCommand(ctx, tx, mutation, agentSkillResourceType, id, 1, 201)
	})
}

func (store *Store) PatchSkill(
	ctx context.Context, namespaceID, id string, expected int64, patch agentmanagement.SkillPatch,
	mutation agentmanagement.MutationContext,
) (agentmanagement.Skill, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.Skill, error) {
		current, patchSkillErr := lockSkill(ctx, tx, namespaceID, id, expected)
		if patchSkillErr != nil {
			return agentmanagement.Skill{}, patchSkillErr
		}
		input := applySkillPatch(current, patch)
		tools, capabilities, digest, patchSkillErr := encodeSkillRevision(input)
		if patchSkillErr != nil {
			return agentmanagement.Skill{}, patchSkillErr
		}
		contentRevision := current.ContentRevision + 1
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_skill_revisions
  (skill_id,namespace_id,revision,instructions,required_tools,minimum_capabilities,content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, id, namespaceID, contentRevision,
			input.Instructions, tools, capabilities, digest[:], nullableString(mutation.PrincipalID)); err != nil {
			return agentmanagement.Skill{}, classifyWriteError(err)
		}
		result, patchSkillErr := tx.ExecContext(ctx, `UPDATE agent_skills
SET name=$4,description=$5,current_revision=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND builtin=FALSE AND status<>'deleted'`,
			namespaceID, id, expected, input.Name, input.Description, contentRevision)
		if patchSkillErr != nil {
			return agentmanagement.Skill{}, classifyWriteError(patchSkillErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.Skill{}, err
		}
		return scanSkill(tx.QueryRowContext(ctx, skillSelect+` WHERE s.namespace_id=$1 AND s.id=$2`, namespaceID, id))
	})
}

func (store *Store) DeleteSkill(
	ctx context.Context, namespaceID, id string, expected int64, _ agentmanagement.MutationContext,
) (int64, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (int64, error) {
		result, err := tx.ExecContext(ctx, `UPDATE agent_skills skill
SET status='deleted',revision=revision+1,updated_at=clock_timestamp(),deleted_at=clock_timestamp()
WHERE skill.namespace_id=$1 AND skill.id=$2 AND skill.revision=$3
  AND skill.builtin=FALSE AND skill.status<>'deleted'
  AND NOT EXISTS (
    SELECT 1 FROM agent_profile_skills pin
    JOIN agent_profiles profile ON profile.id=pin.profile_id
    WHERE pin.skill_id=skill.id AND pin.profile_revision=profile.current_revision
      AND profile.status='active')`, namespaceID, id, expected)
		if err != nil {
			return 0, classifyWriteError(err)
		}
		if err := requireOneRow(result); err != nil {
			return 0, err
		}
		return expected + 1, nil
	})
}

func encodeSkillRevision(input agentmanagement.SkillInput) ([]byte, []byte, [sha256.Size]byte, error) {
	tools, err := json.Marshal(input.RequiredTools)
	if err != nil {
		return nil, nil, [sha256.Size]byte{}, err
	}
	capabilities, err := json.Marshal(input.MinimumCapabilities)
	if err != nil {
		return nil, nil, [sha256.Size]byte{}, err
	}
	canonical, err := json.Marshal(input)
	if err != nil {
		return nil, nil, [sha256.Size]byte{}, err
	}
	return tools, capabilities, sha256.Sum256(canonical), nil
}

func lockSkill(
	ctx context.Context, tx *sql.Tx, namespaceID, id string, expected int64,
) (agentmanagement.Skill, error) {
	var revision int64
	var builtin bool
	err := tx.QueryRowContext(ctx, `SELECT revision,builtin FROM agent_skills
WHERE namespace_id=$1 AND id=$2 AND status<>'deleted' FOR UPDATE`, namespaceID, id).Scan(&revision, &builtin)
	if err != nil {
		return agentmanagement.Skill{}, mapNotFound(err)
	}
	if builtin {
		return agentmanagement.Skill{}, agentmanagement.ErrDenied
	}
	if revision != expected {
		return agentmanagement.Skill{}, agentmanagement.ErrConflict
	}
	return scanSkill(tx.QueryRowContext(ctx, skillSelect+` WHERE s.namespace_id=$1 AND s.id=$2`, namespaceID, id))
}

func applySkillPatch(current agentmanagement.Skill, patch agentmanagement.SkillPatch) agentmanagement.SkillInput {
	input := agentmanagement.SkillInput{
		Name: current.Name, Description: current.Description, Instructions: current.Instructions,
		RequiredTools: current.RequiredTools, MinimumCapabilities: current.MinimumCapabilities,
	}
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.Instructions != nil {
		input.Instructions = *patch.Instructions
	}
	if patch.RequiredTools != nil {
		input.RequiredTools = *patch.RequiredTools
	}
	if patch.MinimumCapabilities != nil {
		input.MinimumCapabilities = *patch.MinimumCapabilities
	}
	return input
}
