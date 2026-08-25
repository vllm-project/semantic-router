package postgres

import (
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

type rowScanner interface {
	Scan(...any) error
}

const profileSelectPrefix = `SELECT p.id::text,p.namespace_id::text,p.name,p.description,p.status,p.revision,
       p.created_at,p.updated_at,r.revision,r.target_model_id,r.target_entrypoint_id,
       COALESCE(model.name,entrypoint.name,''),r.minimum_target_capabilities,r.supported_modes,
       r.tool_policy,r.approval_policy,r.maximum_turn_seconds,r.maximum_tool_steps,r.context_token_budget,
       COALESCE((SELECT jsonb_agg(jsonb_build_object('id',pins.skill_id::text,'revision',pins.skill_revision)
                         ORDER BY pins.ordinal)
                 FROM agent_profile_skills pins
                WHERE pins.profile_id=p.id AND pins.profile_revision=r.revision),'[]'::jsonb),
       COALESCE((SELECT jsonb_agg(defaults.mode ORDER BY defaults.mode)
	                 FROM agent_profile_defaults defaults
	                WHERE defaults.namespace_id=p.namespace_id AND defaults.profile_id=p.id
	                  AND defaults.profile_revision=r.revision),'[]'::jsonb)
	  FROM agent_profiles p
	  JOIN agent_profile_revisions r ON r.profile_id=p.id AND r.revision=`

const profileSelectSuffix = `
	  LEFT JOIN routing_models model ON model.id=r.target_model_id AND model.namespace_id=p.namespace_id
	  LEFT JOIN routing_entrypoints entrypoint ON entrypoint.id=r.target_entrypoint_id AND entrypoint.namespace_id=p.namespace_id`

const (
	profileCurrentSelect  = profileSelectPrefix + "p.current_revision" + profileSelectSuffix
	profileRevisionSelect = profileSelectPrefix + "$3" + profileSelectSuffix
)

func scanProfile(scanner rowScanner) (agentmanagement.Profile, error) {
	var (
		value                                    agentmanagement.Profile
		targetModel, targetEntrypoint            sql.NullString
		targetPublic                             string
		minimum, modes, policy, skills, defaults []byte
	)
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.Name, &value.Description, &value.Status,
		&value.Revision, &value.CreatedAt, &value.UpdatedAt, &value.ContentRevision,
		&targetModel, &targetEntrypoint, &targetPublic, &minimum, &modes, &policy,
		&value.ApprovalPolicy, &value.MaximumTurnSeconds, &value.MaximumToolSteps,
		&value.ContextTokenBudget, &skills, &defaults,
	)
	if err != nil {
		return agentmanagement.Profile{}, mapNotFound(err)
	}
	if err := decodeJSON(minimum, &value.MinimumTargetCapabilities); err != nil {
		return agentmanagement.Profile{}, err
	}
	if err := decodeJSON(modes, &value.SupportedModes); err != nil {
		return agentmanagement.Profile{}, err
	}
	if err := decodeJSON(policy, &value.ToolPolicy); err != nil {
		return agentmanagement.Profile{}, err
	}
	if err := decodeJSON(skills, &value.Skills); err != nil {
		return agentmanagement.Profile{}, err
	}
	if err := decodeJSON(defaults, &value.DefaultForModes); err != nil {
		return agentmanagement.Profile{}, err
	}
	switch {
	case targetModel.Valid:
		value.DefaultTarget = &agentmanagement.Target{Kind: agentmanagement.TargetModel, ID: targetPublic}
	case targetEntrypoint.Valid:
		value.DefaultTarget = &agentmanagement.Target{Kind: agentmanagement.TargetEntrypoint, ID: targetPublic}
	}
	return value, nil
}

const skillSelect = `SELECT s.id::text,COALESCE(s.namespace_id::text,''),s.name,s.description,s.status,s.revision,
       s.created_at,s.updated_at,r.revision,s.builtin,r.instructions,r.required_tools,
       r.minimum_capabilities,r.content_digest
  FROM agent_skills s
  JOIN agent_skill_revisions r ON r.skill_id=s.id AND r.revision=s.current_revision`

const skillRevisionSelect = `SELECT s.id::text,COALESCE(s.namespace_id::text,''),s.name,s.description,s.status,s.revision,
       s.created_at,s.updated_at,r.revision,s.builtin,r.instructions,r.required_tools,
       r.minimum_capabilities,r.content_digest
  FROM agent_skills s
  JOIN agent_skill_revisions r ON r.skill_id=s.id`

func scanSkill(scanner rowScanner) (agentmanagement.Skill, error) {
	var value agentmanagement.Skill
	var tools, capabilities, digest []byte
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.Name, &value.Description, &value.Status,
		&value.Revision, &value.CreatedAt, &value.UpdatedAt, &value.ContentRevision,
		&value.Builtin, &value.Instructions, &tools, &capabilities, &digest,
	)
	if err != nil {
		return agentmanagement.Skill{}, mapNotFound(err)
	}
	if err := decodeJSON(tools, &value.RequiredTools); err != nil {
		return agentmanagement.Skill{}, err
	}
	if err := decodeJSON(capabilities, &value.MinimumCapabilities); err != nil {
		return agentmanagement.Skill{}, err
	}
	value.ContentDigest = "sha256:" + hex.EncodeToString(digest)
	return value, nil
}

const sourceSelect = `SELECT s.id::text,s.namespace_id::text,s.name,s.description,s.status,s.revision,
       s.created_at,s.updated_at,r.revision,s.source_kind,r.transport,r.endpoint,
       COALESCE(r.credential_id::text,''),r.egress_policy,r.discovered_tools,r.discovery_digest,
       s.approved_discovery_digest
  FROM agent_tool_sources s
  JOIN agent_tool_source_revisions r ON r.source_id=s.id AND r.revision=s.current_revision`

const sourceRevisionSelect = `SELECT s.id::text,s.namespace_id::text,s.name,s.description,s.status,s.revision,
       s.created_at,s.updated_at,r.revision,s.source_kind,r.transport,r.endpoint,
       COALESCE(r.credential_id::text,''),r.egress_policy,r.discovered_tools,r.discovery_digest,
       s.approved_discovery_digest
  FROM agent_tool_sources s
  JOIN agent_tool_source_revisions r ON r.namespace_id=s.namespace_id AND r.source_id=s.id`

func scanToolSource(scanner rowScanner) (agentmanagement.ToolSource, error) {
	var value agentmanagement.ToolSource
	var egress, tools, digest, approved []byte
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.Name, &value.Description, &value.Status,
		&value.Revision, &value.CreatedAt, &value.UpdatedAt, &value.ContentRevision,
		&value.Kind, &value.Transport, &value.Endpoint, &value.CredentialID,
		&egress, &tools, &digest, &approved,
	)
	if err != nil {
		return agentmanagement.ToolSource{}, mapNotFound(err)
	}
	if err := decodeJSON(egress, &value.EgressPolicy); err != nil {
		return agentmanagement.ToolSource{}, err
	}
	if err := decodeJSON(tools, &value.DiscoveredTools); err != nil {
		return agentmanagement.ToolSource{}, err
	}
	if len(digest) > 0 {
		value.DiscoveryDigest = "sha256:" + hex.EncodeToString(digest)
	}
	if len(approved) > 0 {
		value.ApprovedDiscoveryDigest = "sha256:" + hex.EncodeToString(approved)
	}
	switch {
	case value.Status != agentmanagement.StatusActive:
		value.Availability = agentmanagement.ToolSourceDisabled
	case value.DiscoveryDigest == "":
		value.Availability = agentmanagement.ToolSourceUndiscovered
	case value.ApprovedDiscoveryDigest == "":
		value.Availability = agentmanagement.ToolSourcePendingApproval
	case value.DiscoveryDigest != value.ApprovedDiscoveryDigest:
		value.Availability = agentmanagement.ToolSourceDrifted
	default:
		value.Availability = agentmanagement.ToolSourceReady
	}
	return value, nil
}

// #nosec G101 -- this constant selects metadata only and contains no credential value.
const credentialSelect = `SELECT id::text,namespace_id::text,name,''::text,status,revision,
       created_at,updated_at,COALESCE(active_version_id::text,'')
  FROM agent_tool_credentials`

func scanToolCredential(scanner rowScanner) (agentmanagement.ToolCredential, error) {
	var value agentmanagement.ToolCredential
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.Name, &value.Description, &value.Status,
		&value.Revision, &value.CreatedAt, &value.UpdatedAt, &value.ActiveVersionID,
	)
	if err != nil {
		return agentmanagement.ToolCredential{}, mapNotFound(err)
	}
	return value, nil
}

const sessionSelect = `SELECT session.id::text,session.namespace_id::text,session.owner_principal_id::text,
       COALESCE(session.effective_user_id::text,''),COALESCE(session.effective_team_id::text,''),
       delegation.api_key_id::text,session.delegated_inference_session_id::text,session.profile_id::text,session.profile_revision,
       session.target_model_id,session.target_entrypoint_id,session.target_public_id,session.authority_digest,session.mode,
       session.title,session.status,session.revision,session.created_at,session.updated_at
  FROM agent_sessions session
  JOIN delegated_inference_sessions delegation
    ON delegation.namespace_id=session.namespace_id AND delegation.id=session.delegated_inference_session_id`

func scanSession(scanner rowScanner) (agentmanagement.Session, error) {
	var value agentmanagement.Session
	var model, entrypoint sql.NullString
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.OwnerPrincipalID, &value.EffectiveUserID,
		&value.EffectiveTeamID, &value.KeyID, &value.DelegatedInferenceSessionID, &value.ProfileID,
		&value.ProfileRevision, &model, &entrypoint, &value.Target.ID, &value.AuthorityDigest, &value.Mode,
		&value.Title, &value.Status, &value.Revision, &value.CreatedAt, &value.UpdatedAt,
	)
	if err != nil {
		return agentmanagement.Session{}, mapNotFound(err)
	}
	if model.Valid {
		value.Target.Kind, value.TargetResourceID = agentmanagement.TargetModel, model.String
	} else if entrypoint.Valid {
		value.Target.Kind, value.TargetResourceID = agentmanagement.TargetEntrypoint, entrypoint.String
	} else {
		return agentmanagement.Session{}, fmt.Errorf("stored Agent Session target is invalid")
	}
	return value, nil
}

const turnSelect = `SELECT turn.id::text,turn.session_id::text,turn.ordinal,turn.status,
       COALESCE(turn.registry_revision,''),turn.fence,turn.input,turn.revision,
       turn.cancel_requested_at,turn.failure_code,turn.failure_message,
       turn.created_at,turn.updated_at
  FROM agent_turns turn`

func scanTurn(scanner rowScanner) (agentmanagement.Turn, error) {
	var value agentmanagement.Turn
	var input []byte
	var cancelledAt sql.NullTime
	var failureCode, failureMessage sql.NullString
	err := scanner.Scan(
		&value.ID, &value.SessionID, &value.Ordinal, &value.Status, &value.RegistryRevision,
		&value.Fence, &input, &value.Revision, &cancelledAt, &failureCode, &failureMessage,
		&value.CreatedAt, &value.UpdatedAt,
	)
	if err != nil {
		return agentmanagement.Turn{}, mapNotFound(err)
	}
	if err := decodeJSON(input, &value.Input); err != nil {
		return agentmanagement.Turn{}, err
	}
	if cancelledAt.Valid {
		value.CancelRequestedAt = &cancelledAt.Time
	}
	if failureCode.Valid {
		value.Failure = &agentmanagement.Failure{Code: failureCode.String, Message: failureMessage.String}
	}
	return value, nil
}

func scanEvent(scanner rowScanner) (agentmanagement.Event, error) {
	var value agentmanagement.Event
	err := scanner.Scan(
		&value.SessionID, &value.TurnID, &value.Sequence, &value.Type, &value.Payload, &value.CreatedAt,
	)
	if err != nil {
		return agentmanagement.Event{}, mapNotFound(err)
	}
	return value, nil
}

func scanInvocation(scanner rowScanner) (agentmanagement.InvocationRecord, error) {
	var value agentmanagement.InvocationRecord
	var result []byte
	var artifactID, errorCode sql.NullString
	var completedAt sql.NullTime
	err := scanner.Scan(
		&value.ID, &value.NamespaceID, &value.SessionID, &value.TurnID, &value.Fence,
		&value.RegistryRevision, &value.ToolName, &value.CredentialVersionID, &value.InputDigest, &value.Input,
		&value.Idempotency, &value.Class, &value.Status, &result, &artifactID,
		&errorCode, &value.StartedAt, &completedAt,
	)
	if err != nil {
		return agentmanagement.InvocationRecord{}, mapNotFound(err)
	}
	if len(result) > 0 {
		value.Result = result
	}
	if artifactID.Valid {
		value.ArtifactID = artifactID.String
	}
	if errorCode.Valid {
		value.ErrorCode = errorCode.String
	}
	if completedAt.Valid {
		value.CompletedAt = &completedAt.Time
	}
	return value, nil
}

func scanArtifact(scanner rowScanner) (agentmanagement.Artifact, error) {
	var value agentmanagement.Artifact
	var digest []byte
	err := scanner.Scan(
		&value.ID, &value.SessionID, &value.TurnID, &value.Kind, &value.MediaType,
		&value.Content, &digest, &value.SafePreview, &value.ExpiresAt, &value.CreatedAt,
	)
	if err != nil {
		return agentmanagement.Artifact{}, mapNotFound(err)
	}
	value.Digest = "sha256:" + hex.EncodeToString(digest)
	return value, nil
}

func scanCheckpoint(scanner rowScanner) (agentmanagement.Checkpoint, error) {
	var value agentmanagement.Checkpoint
	var goals, resources, toolResults, decisions, state, digest []byte
	err := scanner.Scan(
		&value.ID, &value.SessionID, &value.TurnID, &value.ThroughSequence,
		&value.Summary, &goals, &resources, &toolResults, &decisions, &state, &digest, &value.CreatedAt,
	)
	if err != nil {
		return agentmanagement.Checkpoint{}, mapNotFound(err)
	}
	if err := decodeJSON(goals, &value.UnresolvedGoals); err != nil {
		return agentmanagement.Checkpoint{}, err
	}
	if err := decodeJSON(resources, &value.ResourceReferences); err != nil {
		return agentmanagement.Checkpoint{}, err
	}
	if err := decodeJSON(toolResults, &value.ToolResultReferences); err != nil {
		return agentmanagement.Checkpoint{}, err
	}
	if err := decodeJSON(decisions, &value.Decisions); err != nil {
		return agentmanagement.Checkpoint{}, err
	}
	value.State = append(json.RawMessage(nil), state...)
	value.Digest = "sha256:" + hex.EncodeToString(digest)
	return value, nil
}

func scanPublicationPlan(scanner rowScanner) (agentmanagement.PublicationPlan, error) {
	var value agentmanagement.PublicationPlan
	var digest []byte
	err := scanner.Scan(
		&value.ID, &value.SessionID, &value.TurnID, &value.RecipeID,
		&value.RecipeContentRevision, &value.RecipeResourceRevision, &value.EntrypointID,
		&value.EntrypointContentRevision, &value.EntrypointResourceRevision,
		&value.CatalogRevision, &value.ExactDiff, &value.Diagnostics, &value.GateResults,
		&digest, &value.Status, &value.ExpiresAt, &value.Revision, &value.OperationID,
		&value.CreatedAt, &value.UpdatedAt,
	)
	if err != nil {
		return agentmanagement.PublicationPlan{}, mapNotFound(err)
	}
	value.Digest = "sha256:" + hex.EncodeToString(digest)
	return value, nil
}

func decodeJSON(document []byte, destination any) error {
	if err := json.Unmarshal(document, destination); err != nil {
		return fmt.Errorf("decode Agent PostgreSQL document: %w", err)
	}
	return nil
}
