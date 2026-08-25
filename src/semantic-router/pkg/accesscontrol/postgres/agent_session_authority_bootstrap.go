package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

type preparedAgentBootstrap struct {
	target          resolvedAgentTarget
	self            delegationmanagement.SelfContext
	key             delegationmanagement.EligibleKey
	delegation      delegationmanagement.Session
	authorityDigest string
	encrypted       agentmanagement.EncryptedSecret
	expiresAt       time.Time
}

func (authority *AgentSessionAuthority) bootstrapInTransaction(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
	now time.Time,
) (agentBootstrapResult, error) {
	stored, replayed, err := commandpostgres.Lock(ctx, tx, request.Command)
	if err != nil {
		return agentBootstrapResult{}, err
	}
	if replayed {
		return authority.replayAgentBootstrap(ctx, tx, request.NamespaceID, stored)
	}
	prepared, err := authority.prepareAgentBootstrap(ctx, tx, request, now)
	if err != nil {
		return agentBootstrapResult{}, err
	}
	return authority.persistAgentBootstrap(ctx, tx, request, prepared, now)
}

func (authority *AgentSessionAuthority) replayAgentBootstrap(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	stored managementcommand.StoredResult,
) (agentBootstrapResult, error) {
	if stored.Resource == nil || stored.Resource.ResourceType != "agent_session" {
		return agentBootstrapResult{}, agentmanagement.ErrConflict
	}
	session, delegation, desired, err := loadAgentSessionBootstrapReplay(
		ctx, tx, namespaceID, stored.Resource.ResourceID,
	)
	if err != nil {
		return agentBootstrapResult{}, err
	}
	if _, err := authority.authorizeAgentUse(
		ctx, tx, session.NamespaceID, session.OwnerPrincipalID,
		session.EffectiveUserID, session.EffectiveTeamID,
	); err != nil {
		return agentBootstrapResult{}, err
	}
	return agentBootstrapResult{
		session: session, delegation: delegation,
		desiredRevision: desired, replayed: true,
	}, nil
}

func (authority *AgentSessionAuthority) prepareAgentBootstrap(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
	now time.Time,
) (preparedAgentBootstrap, error) {
	target, err := resolveAgentTarget(ctx, tx, request.NamespaceID, request.Target)
	if err != nil {
		return preparedAgentBootstrap{}, err
	}
	if capabilityErr := verifyTargetCapabilities(
		ctx, tx, request.NamespaceID, target, request.Profile.MinimumTargetCapabilities,
	); capabilityErr != nil {
		return preparedAgentBootstrap{}, capabilityErr
	}
	if profileErr := lockActiveAgentProfile(ctx, tx, request); profileErr != nil {
		return preparedAgentBootstrap{}, profileErr
	}
	delegation := newAgentDelegation(request)
	self, err := lockDelegationSelf(ctx, tx, delegation)
	if err != nil {
		return preparedAgentBootstrap{}, agentDenied(err)
	}
	key, err := selectAgentInferenceKey(
		ctx, tx, request.NamespaceID, request.PrincipalID,
		request.KeyID, request.EffectiveTeamID, self.UserID, target,
	)
	if err != nil {
		return preparedAgentBootstrap{}, err
	}
	authorityDigest, err := authority.authorizeAgentUse(
		ctx, tx, request.NamespaceID, request.PrincipalID, self.UserID, key.TeamID,
	)
	if err != nil {
		return preparedAgentBootstrap{}, err
	}
	expiresAt, err := validateAgentDelegationCapacity(ctx, tx, request, self, key, now)
	if err != nil {
		return preparedAgentBootstrap{}, err
	}
	issued, err := authority.peppers.Issue(accesscredential.KindDelegation, delegation.ID)
	if err != nil {
		return preparedAgentBootstrap{}, agentmanagement.ErrToolUnavailable
	}
	plaintext := []byte(issued.Plaintext)
	defer clear(plaintext)
	encrypted, err := authority.secrets.Encrypt(ctx, plaintext)
	if err != nil {
		return preparedAgentBootstrap{}, agentmanagement.ErrToolUnavailable
	}
	completeAgentDelegation(&delegation, self, key, issued, authority.audience, expiresAt, now)
	return preparedAgentBootstrap{
		target: target, self: self, key: key, delegation: delegation,
		authorityDigest: authorityDigest, encrypted: encrypted, expiresAt: expiresAt,
	}, nil
}

func lockActiveAgentProfile(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
) error {
	var status string
	var revision int64
	if err := tx.QueryRowContext(ctx, `SELECT status,current_revision FROM agent_profiles
WHERE namespace_id=$1 AND id=$2 FOR KEY SHARE`, request.NamespaceID, request.Profile.ID).Scan(
		&status, &revision,
	); err != nil {
		return agentNotFound(err)
	}
	if status != string(agentmanagement.StatusActive) || revision != request.Profile.ContentRevision {
		return agentmanagement.ErrConflict
	}
	return nil
}

func newAgentDelegation(request agentmanagement.SessionBootstrapRequest) delegationmanagement.Session {
	id := uuid.NewString()
	return delegationmanagement.Session{
		ID: id, PublicID: id, NamespaceID: request.NamespaceID,
		ManagementSessionID: request.Mutation.ManagementSessionID,
		PrincipalID:         request.PrincipalID,
	}
}

func validateAgentDelegationCapacity(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
	self delegationmanagement.SelfContext,
	key delegationmanagement.EligibleKey,
	now time.Time,
) (time.Time, error) {
	expiresAt := now.Add(request.SessionTTL)
	for _, limit := range []time.Time{now.Add(self.Policy.DelegatedSessionTTL), self.ManagementSessionExpires} {
		if expiresAt.After(limit) {
			expiresAt = limit
		}
	}
	if key.ExpiresAt != nil && expiresAt.After(*key.ExpiresAt) {
		expiresAt = key.ExpiresAt.UTC()
	}
	if !expiresAt.After(now) {
		return time.Time{}, agentmanagement.ErrDenied
	}
	var activeCount int
	if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM delegated_inference_sessions
WHERE namespace_id=$1 AND user_id=$2 AND status='active' AND expires_at>clock_timestamp()`,
		request.NamespaceID, self.UserID,
	).Scan(&activeCount); err != nil {
		return time.Time{}, fmt.Errorf("count Agent delegations: %w", err)
	}
	if self.Policy.MaxDelegatedSessions <= 0 || activeCount >= self.Policy.MaxDelegatedSessions {
		return time.Time{}, agentmanagement.ErrDenied
	}
	return expiresAt, nil
}

func completeAgentDelegation(
	delegation *delegationmanagement.Session,
	self delegationmanagement.SelfContext,
	key delegationmanagement.EligibleKey,
	issued accesscredential.Issued,
	audience string,
	expiresAt time.Time,
	now time.Time,
) {
	delegation.QuotaPartition = self.QuotaPartition
	delegation.APIKeyID = key.KeyID
	delegation.DelegationEpoch = key.DelegationEpoch
	delegation.UserID = self.UserID
	delegation.TeamID = key.TeamID
	delegation.TokenHMAC = append([]byte(nil), issued.Digest.HMAC...)
	delegation.PepperVersion = issued.Digest.PepperVersion
	delegation.Audience = audience
	delegation.Status = delegationmanagement.SessionActive
	delegation.NotBefore = now
	delegation.ExpiresAt = expiresAt
	delegation.Revision = 1
	delegation.CreatedAt = now
}

func (authority *AgentSessionAuthority) persistAgentBootstrap(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
	prepared preparedAgentBootstrap,
	now time.Time,
) (agentBootstrapResult, error) {
	if err := insertAgentDelegation(ctx, tx, prepared.delegation); err != nil {
		return agentBootstrapResult{}, err
	}
	session := agentSessionFromBootstrap(request, prepared, now)
	if err := insertAgentSession(ctx, tx, session, prepared); err != nil {
		return agentBootstrapResult{}, err
	}
	desiredRevision, err := completeAgentBootstrapCommand(ctx, tx, request, prepared.delegation)
	if err != nil {
		return agentBootstrapResult{}, err
	}
	return agentBootstrapResult{
		session: session, delegation: prepared.delegation, desiredRevision: desiredRevision,
	}, nil
}

func insertAgentDelegation(ctx context.Context, tx *sql.Tx, delegation delegationmanagement.Session) error {
	if _, err := tx.ExecContext(ctx, `INSERT INTO delegated_inference_sessions
  (id,public_id,namespace_id,management_session_id,principal_id,api_key_id,delegation_epoch,
   user_id,team_id,token_hmac,pepper_version,audience,status,not_before,expires_at,revision,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'active',$13,$14,1,$15)`,
		delegation.ID, delegation.PublicID, delegation.NamespaceID, delegation.ManagementSessionID,
		delegation.PrincipalID, delegation.APIKeyID, delegation.DelegationEpoch, delegation.UserID,
		nullableString(delegation.TeamID), delegation.TokenHMAC, delegation.PepperVersion,
		delegation.Audience, delegation.NotBefore, delegation.ExpiresAt, delegation.CreatedAt,
	); err != nil {
		return mapAgentWrite(err)
	}
	return nil
}

func insertAgentSession(
	ctx context.Context,
	tx *sql.Tx,
	session agentmanagement.Session,
	prepared preparedAgentBootstrap,
) error {
	var targetModel, targetEntrypoint any
	if prepared.target.Kind == agentmanagement.TargetModel {
		targetModel = prepared.target.ResourceID
	} else {
		targetEntrypoint = prepared.target.ResourceID
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO agent_sessions
  (id,namespace_id,owner_principal_id,effective_user_id,effective_team_id,
   delegated_inference_session_id,profile_id,profile_revision,target_model_id,target_entrypoint_id,
   target_public_id,authority_digest,mode,title,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'active',1,$15,$15)`,
		session.ID, session.NamespaceID, session.OwnerPrincipalID, session.EffectiveUserID,
		nullableString(session.EffectiveTeamID), session.DelegatedInferenceSessionID,
		session.ProfileID, session.ProfileRevision, targetModel, targetEntrypoint,
		session.Target.ID, session.AuthorityDigest, session.Mode, session.Title, session.CreatedAt,
	); err != nil {
		return mapAgentWrite(err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO agent_session_inference_credentials
  (namespace_id,session_id,delegated_inference_session_id,secret_ciphertext,ciphertext_nonce,
   kek_version,expires_at,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$8)`, session.NamespaceID, session.ID,
		session.DelegatedInferenceSessionID, prepared.encrypted.Ciphertext, prepared.encrypted.Nonce,
		prepared.encrypted.KEKVersion, prepared.expiresAt, session.CreatedAt,
	); err != nil {
		return mapAgentWrite(err)
	}
	return nil
}

func agentSessionFromBootstrap(
	request agentmanagement.SessionBootstrapRequest,
	prepared preparedAgentBootstrap,
	now time.Time,
) agentmanagement.Session {
	return agentmanagement.Session{
		ID: request.SessionID, NamespaceID: request.NamespaceID, OwnerPrincipalID: request.PrincipalID,
		EffectiveUserID: prepared.self.UserID, EffectiveTeamID: prepared.key.TeamID,
		KeyID: prepared.key.KeyID, DelegatedInferenceSessionID: prepared.delegation.ID,
		ProfileID: request.Profile.ID, ProfileRevision: request.Profile.ContentRevision,
		Target:           agentmanagement.Target{Kind: prepared.target.Kind, ID: prepared.target.PublicID},
		TargetResourceID: prepared.target.ResourceID, AuthorityDigest: prepared.authorityDigest,
		Mode: request.Mode, Title: request.Title,
		Status: agentmanagement.SessionActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
}

func completeAgentBootstrapCommand(
	ctx context.Context,
	tx *sql.Tx,
	request agentmanagement.SessionBootstrapRequest,
	delegation delegationmanagement.Session,
) (uint64, error) {
	meta, err := agentMutationMeta(request.Mutation, "agent.session.create", "Create Agent session delegation.")
	if err != nil {
		return 0, err
	}
	receipt, err := appendMutationRecords(
		ctx, tx, accesscontrol.NamespaceID(request.NamespaceID),
		outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: 1, Operation: outboxCreated,
			References: map[string]string{"apiKeyId": delegation.APIKeyID},
		},
		meta,
	)
	if err != nil {
		return 0, err
	}
	if err := commandpostgres.CompleteResource(ctx, tx, request.Command, managementcommand.ResourceResult{
		ResourceType: "agent_session", ResourceID: request.SessionID,
		ResourceRevision: 1, ResponseStatus: 201,
	}); err != nil {
		return 0, err
	}
	return uint64(receipt.DesiredRevision), nil
}
