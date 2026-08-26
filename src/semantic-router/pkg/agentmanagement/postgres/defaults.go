package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const (
	builtinBuilderSkillName   = "Build with vLLM Semantic Router"
	defaultChatProfileName    = "Chat"
	defaultBuilderProfileName = "Builder"
	defaultMutationAttempts   = 4

	builtinBuilderSkillRevision int64 = 2
	defaultProfileRevision      int64 = 2
)

var builtinBuilderSkillID = uuid.NewSHA1(
	uuid.NameSpaceURL,
	[]byte("vllm-sr/router-agent-skill/builder/v1"),
).String()

func builtinBuilderToolNames() []string {
	return agentmanagement.BuiltinBuilderToolNames()
}

const builtinBuilderInstructions = `Design and tune a Mixture-of-Models through Router-native tools. Start by reading the live catalog and examples; never assume a component schema. Search the public web only when current external evidence would improve the design. Keep drafts unpublished while validating, probing, and evaluating. Prepare an Entrypoint and publication plan only when the dependency closure and gates are valid. Publication always requires a separate human approval.`

func builtinBuilderSkillInput() (agentmanagement.SkillInput, error) {
	return agentmanagement.NormalizeSkillInput(agentmanagement.SkillInput{
		Name:          builtinBuilderSkillName,
		Description:   "Guides the Agent through a safe Builder workflow.",
		Instructions:  builtinBuilderInstructions,
		RequiredTools: builtinBuilderToolNames(),
	})
}

type DefaultReconciler struct {
	store    *Store
	interval time.Duration
	now      func() time.Time
}

func NewDefaultReconciler(store *Store, interval time.Duration, now func() time.Time) (*DefaultReconciler, error) {
	if store == nil || store.db == nil {
		return nil, errors.New("agent default Profile store is unavailable")
	}
	if interval == 0 {
		interval = 30 * time.Second
	}
	if interval < time.Second || interval > 10*time.Minute {
		return nil, errors.New("agent default Profile reconcile interval is invalid")
	}
	if now == nil {
		now = time.Now
	}
	return &DefaultReconciler{store: store, interval: interval, now: now}, nil
}

func (reconciler *DefaultReconciler) Reconcile(ctx context.Context) error {
	if reconciler == nil || reconciler.store == nil {
		return errors.New("agent defaults are unavailable")
	}
	if err := reconciler.ensureBuiltinSkill(ctx); err != nil {
		return err
	}
	rows, err := reconciler.store.db.QueryContext(ctx, `SELECT id::text FROM access_namespaces
WHERE status='active' ORDER BY id`)
	if err != nil {
		return fmt.Errorf("list Namespaces for Agent defaults: %w", err)
	}
	var namespaces []string
	for rows.Next() {
		var namespaceID string
		if err := rows.Scan(&namespaceID); err != nil {
			return errors.Join(err, rows.Close())
		}
		namespaces = append(namespaces, namespaceID)
	}
	if err := rows.Close(); err != nil {
		return err
	}
	for _, namespaceID := range namespaces {
		if err := reconciler.ensureNamespaceDefaults(ctx, namespaceID); err != nil {
			return err
		}
	}
	return nil
}

func (reconciler *DefaultReconciler) Ready(ctx context.Context) error {
	if reconciler == nil || reconciler.store == nil || reconciler.store.db == nil {
		return errors.New("agent defaults are unavailable")
	}
	var missing int
	err := reconciler.store.db.QueryRowContext(ctx, `SELECT count(*) FROM access_namespaces namespace
WHERE namespace.status='active' AND EXISTS (
  SELECT 1 FROM (VALUES ('chat'),('builder')) mode(value)
  WHERE NOT EXISTS (
    SELECT 1 FROM agent_profile_defaults defaults
    JOIN agent_profiles profile ON profile.namespace_id=defaults.namespace_id AND profile.id=defaults.profile_id
    WHERE defaults.namespace_id=namespace.id AND defaults.mode=mode.value AND profile.status='active'
  )
)`).Scan(&missing)
	if err != nil {
		return fmt.Errorf("verify Agent defaults: %w", err)
	}
	if missing != 0 {
		return fmt.Errorf("%d active Namespaces are missing default Agent Profiles", missing)
	}
	return nil
}

func (reconciler *DefaultReconciler) Run(ctx context.Context) error {
	if err := reconciler.Reconcile(ctx); err != nil {
		return err
	}
	ticker := time.NewTicker(reconciler.interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if err := reconciler.Reconcile(ctx); err != nil {
				return err
			}
		}
	}
}

func (reconciler *DefaultReconciler) ensureBuiltinSkill(ctx context.Context) error {
	input, err := builtinBuilderSkillInput()
	if err != nil {
		return fmt.Errorf("validate built-in Agent Skill: %w", err)
	}
	required, minimum, digest, err := encodeSkillRevision(input)
	if err != nil {
		return fmt.Errorf("encode built-in Agent Skill: %w", err)
	}
	return retryDefaultMutation(ctx, func() error {
		_, err := inTransaction(ctx, reconciler.store, func(tx *sql.Tx) (struct{}, error) {
			if _, err := tx.ExecContext(ctx, `INSERT INTO agent_skills
  (id,namespace_id,name,description,builtin,status,current_revision,revision)
VALUES ($1,NULL,$2,$3,TRUE,'active',$4,1)
ON CONFLICT (id) DO NOTHING`, builtinBuilderSkillID, input.Name, input.Description,
				builtinBuilderSkillRevision); err != nil {
				return struct{}{}, classifyWriteError(err)
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO agent_skill_revisions
  (skill_id,namespace_id,revision,instructions,required_tools,minimum_capabilities,content_digest)
VALUES ($1,NULL,$2,$3,$4,$5,$6) ON CONFLICT (skill_id,revision) DO NOTHING`,
				builtinBuilderSkillID, builtinBuilderSkillRevision, input.Instructions,
				required, minimum, digest[:]); err != nil {
				return struct{}{}, classifyWriteError(err)
			}
			var storedName, storedDescription, storedStatus string
			var storedBuiltin bool
			var storedRevision int64
			var storedDigest []byte
			if err := tx.QueryRowContext(ctx, `SELECT skill.name,skill.description,skill.builtin,
       skill.status,skill.current_revision,revision.content_digest
FROM agent_skills skill
JOIN agent_skill_revisions revision ON revision.skill_id=skill.id AND revision.revision=$2
WHERE skill.id=$1 AND skill.namespace_id IS NULL`,
				builtinBuilderSkillID, builtinBuilderSkillRevision).Scan(
				&storedName, &storedDescription, &storedBuiltin, &storedStatus, &storedRevision, &storedDigest,
			); err != nil {
				return struct{}{}, mapNotFound(err)
			}
			if storedName != input.Name || storedDescription != input.Description || !storedBuiltin ||
				storedStatus != string(agentmanagement.StatusActive) ||
				storedRevision != builtinBuilderSkillRevision || !bytes.Equal(storedDigest, digest[:]) {
				return struct{}{}, fmt.Errorf("%w: built-in Agent Skill identity conflicts with Router defaults", agentmanagement.ErrConflict)
			}
			return struct{}{}, nil
		})
		return err
	})
}

func (reconciler *DefaultReconciler) ensureNamespaceDefaults(ctx context.Context, namespaceID string) error {
	definitions, err := defaultProfileDefinitions(namespaceID)
	if err != nil {
		return err
	}
	return retryDefaultMutation(ctx, func() error {
		_, err := inTransaction(ctx, reconciler.store, func(tx *sql.Tx) (struct{}, error) {
			for _, definition := range definitions {
				if err := insertDefaultProfile(ctx, tx, namespaceID, definition.id, definition.input); err != nil {
					return struct{}{}, err
				}
				for _, mode := range definition.input.DefaultForModes {
					if _, err := tx.ExecContext(
						ctx,
						insertDefaultProfileMapping,
						namespaceID,
						mode,
						definition.id,
						defaultProfileRevision,
						reconciler.now().UTC(),
					); err != nil {
						return struct{}{}, classifyWriteError(err)
					}
				}
			}
			return struct{}{}, nil
		})
		return err
	})
}

type defaultProfileDefinition struct {
	id    string
	input agentmanagement.ProfileInput
}

const insertDefaultProfileMapping = `INSERT INTO agent_profile_defaults
  (namespace_id,mode,profile_id,profile_revision,updated_by,updated_at)
VALUES ($1,$2,$3,$4,NULL,$5)
ON CONFLICT (namespace_id,mode) DO NOTHING`

func defaultProfileDefinitions(namespaceID string) ([]defaultProfileDefinition, error) {
	namespaceUUID, err := uuid.Parse(namespaceID)
	if err != nil {
		return nil, fmt.Errorf("parse Namespace for Agent defaults: %w", err)
	}
	chatInput := agentmanagement.ProfileInput{
		Name:            defaultChatProfileName,
		Description:     "Router-managed default.",
		SupportedModes:  []agentmanagement.SessionMode{agentmanagement.SessionChat},
		DefaultForModes: []agentmanagement.SessionMode{agentmanagement.SessionChat},
		ToolPolicy: agentmanagement.ToolPolicy{Allow: []string{
			"router.skills.read",
			agentmanagement.ToolWebSearch,
		}},
		ApprovalPolicy:     "required",
		MaximumTurnSeconds: 900,
		MaximumToolSteps:   64,
		ContextTokenBudget: 65536,
	}
	builderInput := agentmanagement.ProfileInput{
		Name:            defaultBuilderProfileName,
		Description:     "Router-managed default.",
		SupportedModes:  []agentmanagement.SessionMode{agentmanagement.SessionBuilder},
		DefaultForModes: []agentmanagement.SessionMode{agentmanagement.SessionBuilder},
		Skills: []agentmanagement.SkillReference{{
			ID:       builtinBuilderSkillID,
			Revision: builtinBuilderSkillRevision,
		}},
		ToolPolicy:         agentmanagement.ToolPolicy{Allow: builtinBuilderToolNames()},
		ApprovalPolicy:     "required",
		MaximumTurnSeconds: 900,
		MaximumToolSteps:   64,
		ContextTokenBudget: 65536,
	}
	definitions := []defaultProfileDefinition{
		{
			id:    uuid.NewSHA1(namespaceUUID, []byte("router-agent-profile/chat/v1")).String(),
			input: chatInput,
		},
		{
			id:    uuid.NewSHA1(namespaceUUID, []byte("router-agent-profile/builder/v1")).String(),
			input: builderInput,
		},
	}
	for index := range definitions {
		definitions[index].input, err = agentmanagement.NormalizeProfileInput(definitions[index].input)
		if err != nil {
			return nil, fmt.Errorf("validate default Agent Profile %q: %w", definitions[index].input.Name, err)
		}
	}
	return definitions, nil
}

func insertDefaultProfile(
	ctx context.Context, tx *sql.Tx, namespaceID, profileID string, input agentmanagement.ProfileInput,
) error {
	if input.DefaultTarget != nil {
		return fmt.Errorf("%w: Router default Profile cannot pin a target", agentmanagement.ErrInvalid)
	}
	encoded, digest, err := encodeProfileRevision(input)
	if err != nil {
		return err
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profiles
  (id,namespace_id,name,description,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'active',$5,1)
ON CONFLICT (id) DO NOTHING`, profileID, namespaceID, input.Name, input.Description,
		defaultProfileRevision); err != nil {
		return classifyWriteError(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_revisions
  (profile_id,namespace_id,revision,target_model_id,target_entrypoint_id,minimum_target_capabilities,
   supported_modes,tool_policy,approval_policy,maximum_turn_seconds,maximum_tool_steps,
   context_token_budget,content_digest)
VALUES ($1,$2,$3,NULL,NULL,$4,$5,$6,$7,$8,$9,$10,$11)
ON CONFLICT (profile_id,revision) DO NOTHING`, profileID, namespaceID,
		defaultProfileRevision, encoded.minimumCapabilities, encoded.supportedModes, encoded.toolPolicy,
		input.ApprovalPolicy, input.MaximumTurnSeconds, input.MaximumToolSteps,
		input.ContextTokenBudget, digest[:]); err != nil {
		return classifyWriteError(err)
	}
	for ordinal, skill := range input.Skills {
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_profile_skills
  (namespace_id,profile_id,profile_revision,ordinal,skill_id,skill_namespace_id,skill_revision)
VALUES ($1,$2,$3,$4,$5,NULL,$6) ON CONFLICT (profile_id,profile_revision,ordinal) DO NOTHING`,
			namespaceID, profileID, defaultProfileRevision, ordinal, skill.ID, skill.Revision); err != nil {
			return classifyWriteError(err)
		}
	}
	var storedName, storedDescription, storedStatus string
	var storedRevision int64
	var storedDigest []byte
	if err := tx.QueryRowContext(ctx, `SELECT profile.name,profile.description,profile.status,
       profile.current_revision,revision.content_digest
FROM agent_profiles profile
JOIN agent_profile_revisions revision ON revision.profile_id=profile.id AND revision.revision=$3
WHERE profile.namespace_id=$1 AND profile.id=$2`, namespaceID, profileID, defaultProfileRevision).Scan(
		&storedName, &storedDescription, &storedStatus, &storedRevision, &storedDigest,
	); err != nil {
		return mapNotFound(err)
	}
	if storedName != input.Name || storedDescription != input.Description ||
		storedStatus != string(agentmanagement.StatusActive) || storedRevision != defaultProfileRevision ||
		!bytes.Equal(storedDigest, digest[:]) {
		return fmt.Errorf("%w: default Agent Profile identity conflicts with Router defaults", agentmanagement.ErrConflict)
	}
	var storedSkillCount int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM agent_profile_skills
WHERE namespace_id=$1 AND profile_id=$2 AND profile_revision=$3`,
		namespaceID, profileID, defaultProfileRevision).Scan(&storedSkillCount); err != nil {
		return fmt.Errorf("verify default Agent Profile Skills: %w", err)
	}
	if storedSkillCount != len(input.Skills) {
		return fmt.Errorf("%w: default Agent Profile Skill pins conflict with Router defaults", agentmanagement.ErrConflict)
	}
	for ordinal, skill := range input.Skills {
		var storedSkillID string
		var storedSkillRevision int64
		if err := tx.QueryRowContext(ctx, `SELECT skill_id::text,skill_revision
FROM agent_profile_skills
WHERE namespace_id=$1 AND profile_id=$2 AND profile_revision=$3 AND ordinal=$4`,
			namespaceID, profileID, defaultProfileRevision, ordinal).Scan(&storedSkillID, &storedSkillRevision); err != nil {
			return mapNotFound(err)
		}
		if storedSkillID != skill.ID || storedSkillRevision != skill.Revision {
			return fmt.Errorf("%w: default Agent Profile Skill pins conflict with Router defaults", agentmanagement.ErrConflict)
		}
	}
	return nil
}

func retryDefaultMutation(ctx context.Context, operation func() error) error {
	var err error
	for attempt := 0; attempt < defaultMutationAttempts; attempt++ {
		err = operation()
		if !errors.Is(err, agentmanagement.ErrConflict) {
			return err
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
	return fmt.Errorf("reconcile Router Agent defaults after %d attempts: %w", defaultMutationAttempts, err)
}
