package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementpermission"
)

// AgentManagementAuthority loads the exact Management grant snapshot from a
// transaction owned by the session bootstrap or reauthorization operation.
type AgentManagementAuthority interface {
	LoadInTransaction(
		context.Context,
		*sql.Tx,
		accesscontrol.ManagementPrincipalID,
		accesscontrol.NamespaceID,
	) (managementauthorization.Snapshot, error)
}

type AgentSessionAuthorityOptions struct {
	Store       *Store
	Management  AgentManagementAuthority
	Peppers     accesscredential.PepperKeyring
	Secrets     agentmanagement.SecretCodec
	Waiter      delegationmanagement.PublicationWaiter
	Audience    string
	RenewalLead time.Duration
	Now         func() time.Time
}

// AgentSessionAuthority atomically binds one authenticated Management
// principal, an effective User/Team, an authorized public inference target,
// an internal delegated credential, and one durable Agent Session.
type AgentSessionAuthority struct {
	store       *Store
	management  AgentManagementAuthority
	peppers     accesscredential.PepperKeyring
	secrets     agentmanagement.SecretCodec
	waiter      delegationmanagement.PublicationWaiter
	audience    string
	renewalLead time.Duration
	now         func() time.Time
}

func NewAgentSessionAuthority(options AgentSessionAuthorityOptions) (*AgentSessionAuthority, error) {
	if options.Store == nil || options.Store.db == nil || options.Management == nil ||
		options.Secrets == nil || options.Waiter == nil ||
		strings.TrimSpace(options.Audience) == "" || options.Peppers.Validate() != nil {
		return nil, fmt.Errorf("agent session authority dependencies are incomplete")
	}
	lead := options.RenewalLead
	if lead == 0 {
		lead = 2 * time.Minute
	}
	if lead < 30*time.Second || lead > 15*time.Minute {
		return nil, fmt.Errorf("agent delegated credential renewal lead is invalid")
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	return &AgentSessionAuthority{
		store: options.Store, management: options.Management,
		// The factory erases its short-lived keyring after composition. The
		// authority must own the issuer it retains for future session creation.
		peppers: options.Peppers.Clone(), secrets: options.Secrets,
		waiter: options.Waiter, audience: options.Audience, renewalLead: lead, now: now,
	}, nil
}

type resolvedAgentTarget struct {
	Kind       agentmanagement.TargetKind
	ResourceID string
	PublicID   string
}

type agentBootstrapResult struct {
	session         agentmanagement.Session
	delegation      delegationmanagement.Session
	desiredRevision uint64
	replayed        bool
}

func (authority *AgentSessionAuthority) Prepare(
	ctx context.Context, request agentmanagement.SessionAuthorizationRequest,
) (agentmanagement.SessionAuthorization, error) {
	if authority == nil || uuid.Validate(request.NamespaceID) != nil ||
		uuid.Validate(request.PrincipalID) != nil ||
		uuid.Validate(request.KeyID) != nil ||
		(request.EffectiveTeamID != "" && uuid.Validate(request.EffectiveTeamID) != nil) {
		return agentmanagement.SessionAuthorization{}, agentmanagement.ErrInvalid
	}
	return inReadTransaction(ctx, authority.store, func(tx *sql.Tx) (agentmanagement.SessionAuthorization, error) {
		target, prepareErr := resolveAgentTarget(ctx, tx, request.NamespaceID, request.Target)
		if prepareErr != nil {
			return agentmanagement.SessionAuthorization{}, prepareErr
		}
		if err := verifyTargetCapabilities(
			ctx, tx, request.NamespaceID, target, request.Profile.MinimumTargetCapabilities,
		); err != nil {
			return agentmanagement.SessionAuthorization{}, err
		}
		var userID string
		if err := tx.QueryRowContext(ctx, `SELECT link.user_id
FROM management_principal_user_links link
JOIN management_principals principal ON principal.id=link.principal_id
JOIN access_users user_record ON user_record.namespace_id=link.namespace_id AND user_record.id=link.user_id
WHERE link.namespace_id=$1 AND link.principal_id=$2
  AND principal.status='active' AND user_record.status='active'`,
			request.NamespaceID, request.PrincipalID).Scan(&userID); err != nil {
			return agentmanagement.SessionAuthorization{}, agentNotFound(err)
		}
		key, prepareErr := selectAgentInferenceKeyRead(
			ctx, tx, request.NamespaceID, request.PrincipalID,
			request.KeyID, request.EffectiveTeamID, userID, target,
		)
		if prepareErr != nil {
			return agentmanagement.SessionAuthorization{}, prepareErr
		}
		return agentmanagement.SessionAuthorization{
			EffectiveUserID: userID, EffectiveTeamID: key.TeamID,
			KeyID:      key.KeyID,
			TargetKind: target.Kind, TargetResourceID: target.ResourceID,
		}, nil
	})
}

func (authority *AgentSessionAuthority) Bootstrap(
	ctx context.Context, request agentmanagement.SessionBootstrapRequest,
) (agentmanagement.Session, bool, error) {
	if authority == nil || uuid.Validate(request.SessionID) != nil || uuid.Validate(request.NamespaceID) != nil ||
		uuid.Validate(request.PrincipalID) != nil || uuid.Validate(request.Mutation.ManagementSessionID) != nil ||
		uuid.Validate(request.KeyID) != nil ||
		request.PrincipalID != request.Mutation.PrincipalID || request.SessionTTL <= 0 {
		return agentmanagement.Session{}, false, agentmanagement.ErrInvalid
	}
	now := authority.now().UTC()
	result, err := inTransaction(ctx, authority.store, func(tx *sql.Tx) (agentBootstrapResult, error) {
		return authority.bootstrapInTransaction(ctx, tx, request, now)
	})
	if err != nil {
		return agentmanagement.Session{}, false, err
	}
	clear(result.delegation.TokenHMAC)
	if err := authority.waiter.WaitActive(ctx, result.delegation, result.desiredRevision); err != nil {
		return agentmanagement.Session{}, false, fmt.Errorf("agent delegated inference publication: %w", err)
	}
	return result.session, result.replayed, nil
}

func (authority *AgentSessionAuthority) CanDiscover(
	ctx context.Context, namespaceID, principalID string, target agentmanagement.Target,
) (bool, error) {
	if authority == nil || uuid.Validate(namespaceID) != nil || uuid.Validate(principalID) != nil {
		return false, agentmanagement.ErrInvalid
	}
	return inReadTransaction(ctx, authority.store, func(tx *sql.Tx) (bool, error) {
		resolved, canDiscoverErr := resolveAgentTarget(ctx, tx, namespaceID, target)
		if canDiscoverErr != nil {
			if errors.Is(canDiscoverErr, agentmanagement.ErrNotFound) {
				return false, nil
			}
			return false, canDiscoverErr
		}
		var userID string
		if err := tx.QueryRowContext(ctx, `SELECT l.user_id FROM management_principal_user_links l
JOIN access_users u ON u.namespace_id=l.namespace_id AND u.id=l.user_id
JOIN management_principals p ON p.id=l.principal_id
WHERE l.namespace_id=$1 AND l.principal_id=$2 AND u.status='active' AND p.status='active'`,
			namespaceID, principalID).Scan(&userID); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return false, nil
			}
			return false, err
		}
		_, canDiscoverErr = queryAgentInferenceKey(
			ctx, tx, namespaceID, principalID, nil, false, nil, resolved,
			[]accesscontrol.GrantPermission{accesscontrol.GrantPermissionDiscover}, false,
		)
		if errors.Is(canDiscoverErr, agentmanagement.ErrNotFound) {
			return false, nil
		}
		return canDiscoverErr == nil, canDiscoverErr
	})
}

func (authority *AgentSessionAuthority) authorizeAgentUse(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID, principalID, userID, teamID string,
) (string, error) {
	if authority == nil || authority.management == nil || tx == nil ||
		uuid.Validate(namespaceID) != nil || uuid.Validate(principalID) != nil ||
		uuid.Validate(userID) != nil || (teamID != "" && uuid.Validate(teamID) != nil) {
		return "", agentmanagement.ErrInvalid
	}
	snapshot, err := authority.management.LoadInTransaction(
		ctx, tx,
		accesscontrol.ManagementPrincipalID(principalID),
		accesscontrol.NamespaceID(namespaceID),
	)
	if err != nil {
		return "", err
	}
	if snapshot.AuthorityDigest == "" ||
		snapshot.Principal.ID != accesscontrol.ManagementPrincipalID(principalID) {
		return "", agentmanagement.ErrDenied
	}
	target := accesscontrol.ScopedTarget{Scope: accesscontrol.UserScope(
		accesscontrol.NamespaceID(namespaceID), accesscontrol.UserID(userID),
	)}
	if teamID != "" {
		target.Scope = accesscontrol.TeamScope(
			accesscontrol.NamespaceID(namespaceID), accesscontrol.TeamID(teamID),
		)
	}
	evaluation := managementauthorization.EvaluationContext{
		Authenticated: true,
		RoleGrants:    snapshot.RoleGrants,
		TeamGrants:    snapshot.TeamGrants,
		Targets: map[string][]accesscontrol.ScopedTarget{
			"attributed_subject": {target},
		},
	}
	for _, permission := range []accesscontrol.Permission{
		accesscontrol.PermissionAgentUse,
		accesscontrol.PermissionDelegationUse,
	} {
		if err := managementauthorization.Evaluate(
			managementpermission.Require(string(permission), "attributed_subject"),
			evaluation,
		); err != nil {
			if errors.Is(err, managementauthorization.ErrDenied) {
				return "", agentmanagement.ErrDenied
			}
			return "", err
		}
	}
	return snapshot.AuthorityDigest, nil
}

func (authority *AgentSessionAuthority) Reauthorize(
	ctx context.Context, session agentmanagement.Session, capabilities []string,
) error {
	_, transactionErr := inTransaction(ctx, authority.store, func(tx *sql.Tx) (struct{}, error) {
		resolved := resolvedAgentTarget{Kind: session.Target.Kind, ResourceID: session.TargetResourceID}
		if resolved.ResourceID == "" {
			var err error
			resolved, err = resolveAgentTarget(ctx, tx, session.NamespaceID, session.Target)
			if err != nil {
				return struct{}{}, err
			}
		}
		if err := verifyTargetCapabilities(ctx, tx, session.NamespaceID, resolved, capabilities); err != nil {
			return struct{}{}, err
		}
		delegation, key, err := loadActiveAgentDelegation(ctx, tx, session)
		if err != nil {
			return struct{}{}, err
		}
		allowedDiscover, err := keyCanAccessTarget(ctx, tx, session.NamespaceID, key,
			delegation.UserID, resolved, accesscontrol.GrantPermissionDiscover)
		if err != nil {
			return struct{}{}, err
		}
		allowedInvoke, err := keyCanAccessTarget(ctx, tx, session.NamespaceID, key,
			delegation.UserID, resolved, accesscontrol.GrantPermissionInvoke)
		if err != nil {
			return struct{}{}, err
		}
		if !allowedDiscover || !allowedInvoke {
			return struct{}{}, agentmanagement.ErrDenied
		}
		if _, err := authority.authorizeAgentUse(
			ctx, tx, session.NamespaceID, session.OwnerPrincipalID,
			delegation.UserID, delegation.TeamID,
		); err != nil {
			return struct{}{}, err
		}
		return struct{}{}, nil
	})
	return transactionErr
}

func (authority *AgentSessionAuthority) RenewDelegation(
	ctx context.Context, session agentmanagement.Session, requestedTTL time.Duration,
) error {
	if requestedTTL <= 0 {
		return agentmanagement.ErrInvalid
	}
	now := authority.now().UTC()
	type renewalResult struct {
		delegation delegationmanagement.Session
		desired    uint64
		changed    bool
	}
	result, err := inTransaction(ctx, authority.store, func(tx *sql.Tx) (renewalResult, error) {
		delegation, key, renewDelegationErr := loadActiveAgentDelegation(ctx, tx, session)
		if renewDelegationErr != nil {
			return renewalResult{}, renewDelegationErr
		}
		if delegation.ExpiresAt.After(now.Add(authority.renewalLead)) {
			return renewalResult{delegation: delegation}, nil
		}
		self, renewDelegationErr := lockDelegationSelf(ctx, tx, delegation)
		if renewDelegationErr != nil {
			return renewalResult{}, agentDenied(renewDelegationErr)
		}
		expiresAt := now.Add(requestedTTL)
		policyExpiry := now.Add(self.Policy.DelegatedSessionTTL)
		if expiresAt.After(policyExpiry) {
			expiresAt = policyExpiry
		}
		if expiresAt.After(self.ManagementSessionExpires) {
			expiresAt = self.ManagementSessionExpires
		}
		if key.ExpiresAt != nil && expiresAt.After(*key.ExpiresAt) {
			expiresAt = key.ExpiresAt.UTC()
		}
		if !expiresAt.After(now.Add(authority.renewalLead / 2)) {
			return renewalResult{}, agentmanagement.ErrDenied
		}
		issued, renewDelegationErr := authority.peppers.Issue(accesscredential.KindDelegation, delegation.ID)
		if renewDelegationErr != nil {
			return renewalResult{}, agentmanagement.ErrToolUnavailable
		}
		plaintext := []byte(issued.Plaintext)
		defer clear(plaintext)
		encrypted, renewDelegationErr := authority.secrets.Encrypt(ctx, plaintext)
		if renewDelegationErr != nil {
			return renewalResult{}, agentmanagement.ErrToolUnavailable
		}
		var revision int64
		if err := tx.QueryRowContext(ctx, `UPDATE delegated_inference_sessions
SET token_hmac=$3,pepper_version=$4,not_before=$5,expires_at=$6,revision=revision+1
WHERE namespace_id=$1 AND id=$2 AND status='active' AND revision=$7
RETURNING revision`, session.NamespaceID, delegation.ID, issued.Digest.HMAC,
			issued.Digest.PepperVersion, now, expiresAt, delegation.Revision).Scan(&revision); err != nil {
			return renewalResult{}, mapAgentWrite(err)
		}
		if _, err := tx.ExecContext(ctx, `UPDATE agent_session_inference_credentials
SET secret_ciphertext=$3,ciphertext_nonce=$4,kek_version=$5,expires_at=$6,updated_at=$7
WHERE namespace_id=$1 AND session_id=$2`, session.NamespaceID, session.ID,
			encrypted.Ciphertext, encrypted.Nonce, encrypted.KEKVersion, expiresAt, now); err != nil {
			return renewalResult{}, mapAgentWrite(err)
		}
		revisionValue, revisionErr := positiveUint64(revision, "delegation revision")
		if revisionErr != nil {
			return renewalResult{}, revisionErr
		}
		meta := MutationMeta{
			ActorPrincipalID: nil, RequestID: "agent-worker-" + session.ID,
			Action: "agent.session.renew", Reason: "Renew Agent session delegation.", Details: AuditDetails{},
		}
		receipt, renewDelegationErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(session.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: accesscontrol.Revision(revisionValue), Operation: outboxUpdated,
			References: map[string]string{"apiKeyId": delegation.APIKeyID},
		}, meta)
		if renewDelegationErr != nil {
			return renewalResult{}, renewDelegationErr
		}
		delegation.TokenHMAC = append([]byte(nil), issued.Digest.HMAC...)
		delegation.PepperVersion = issued.Digest.PepperVersion
		delegation.NotBefore, delegation.ExpiresAt, delegation.Revision = now, expiresAt, revisionValue
		return renewalResult{delegation: delegation, desired: uint64(receipt.DesiredRevision), changed: true}, nil
	})
	if err != nil {
		return err
	}
	if !result.changed {
		return nil
	}
	clear(result.delegation.TokenHMAC)
	return authority.waiter.WaitActive(ctx, result.delegation, result.desired)
}

func (authority *AgentSessionAuthority) ResolveInferenceCredential(
	ctx context.Context, session agentmanagement.Session,
) ([]byte, error) {
	var encrypted agentmanagement.EncryptedSecret
	var expiresAt time.Time
	err := authority.store.db.QueryRowContext(ctx, `SELECT c.secret_ciphertext,c.ciphertext_nonce,c.kek_version,c.expires_at
FROM agent_session_inference_credentials c
JOIN agent_sessions s ON s.namespace_id=c.namespace_id AND s.id=c.session_id
JOIN delegated_inference_sessions d ON d.namespace_id=c.namespace_id AND d.id=c.delegated_inference_session_id
WHERE c.namespace_id=$1 AND c.session_id=$2 AND s.status='active' AND d.status='active'
  AND c.expires_at>clock_timestamp() AND d.expires_at>clock_timestamp()`,
		session.NamespaceID, session.ID).Scan(&encrypted.Ciphertext, &encrypted.Nonce, &encrypted.KEKVersion, &expiresAt)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, agentmanagement.ErrDenied
	}
	if err != nil {
		return nil, fmt.Errorf("resolve Agent inference credential: %w", err)
	}
	plaintext, err := authority.secrets.Decrypt(ctx, encrypted)
	if err != nil || len(plaintext) == 0 {
		clear(plaintext)
		return nil, agentmanagement.ErrDenied
	}
	return plaintext, nil
}

func (authority *AgentSessionAuthority) Close(
	ctx context.Context, session agentmanagement.Session, expected int64,
	patch agentmanagement.SessionPatch, mutation agentmanagement.MutationContext,
) (agentmanagement.Session, error) {
	if patch.Status == nil || *patch.Status != agentmanagement.SessionClosed {
		return agentmanagement.Session{}, agentmanagement.ErrInvalid
	}
	type closeResult struct {
		session   agentmanagement.Session
		partition string
		desired   uint64
	}
	result, err := inTransaction(ctx, authority.store, func(tx *sql.Tx) (closeResult, error) {
		delegation, _, closeErr := loadActiveAgentDelegation(ctx, tx, session)
		if closeErr != nil {
			return closeResult{}, closeErr
		}
		title := session.Title
		if patch.Title != nil {
			title = strings.TrimSpace(*patch.Title)
		}
		if len(title) > 256 {
			return closeResult{}, agentmanagement.ErrInvalid
		}
		var sessionRevision int64
		if err := tx.QueryRowContext(ctx, `UPDATE agent_sessions
SET title=$4,status='closed',closed_at=clock_timestamp(),revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND status='active'
RETURNING revision,updated_at`, session.NamespaceID, session.ID, expected, title).Scan(
			&sessionRevision, &session.UpdatedAt); err != nil {
			return closeResult{}, mapAgentWrite(err)
		}
		var delegationRevision int64
		var revokedAt time.Time
		if err := tx.QueryRowContext(ctx, `UPDATE delegated_inference_sessions
SET status='revoked',revoked_at=clock_timestamp(),revision=revision+1
WHERE namespace_id=$1 AND id=$2 AND status='active' RETURNING revision,revoked_at`,
			session.NamespaceID, delegation.ID).Scan(&delegationRevision, &revokedAt); err != nil {
			return closeResult{}, mapAgentWrite(err)
		}
		if _, err := tx.ExecContext(ctx, `DELETE FROM agent_session_inference_credentials
WHERE namespace_id=$1 AND session_id=$2`, session.NamespaceID, session.ID); err != nil {
			return closeResult{}, err
		}
		delegationRevisionValue, revisionErr := positiveUint64(delegationRevision, "delegation revision")
		if revisionErr != nil {
			return closeResult{}, revisionErr
		}
		meta, closeErr := agentMutationMeta(mutation, "agent.session.close", "Close Agent session delegation.")
		if closeErr != nil {
			return closeResult{}, closeErr
		}
		receipt, closeErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(session.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: accesscontrol.Revision(delegationRevisionValue), Operation: outboxDeleted,
			References: map[string]string{"apiKeyId": delegation.APIKeyID},
		}, meta)
		if closeErr != nil {
			return closeResult{}, closeErr
		}
		session.Title, session.Status, session.Revision = title, agentmanagement.SessionClosed, sessionRevision
		return closeResult{
			session: session, partition: delegation.QuotaPartition,
			desired: uint64(receipt.DesiredRevision),
		}, nil
	})
	if err != nil {
		return agentmanagement.Session{}, err
	}
	if err := authority.waiter.WaitApplied(ctx, session.NamespaceID, result.partition, result.desired); err != nil {
		return agentmanagement.Session{}, err
	}
	return result.session, nil
}

const eligibleAgentKeyForTargetSQL = `SELECT k.id,k.name,
  CASE WHEN k.owner_user_id IS NOT NULL THEN 'user' ELSE 'team' END,
  COALESCE(k.owner_user_id::text,k.owner_team_id::text),
  COALESCE(k.context_team_id::text,''),k.expires_at,k.delegation_epoch,
  COALESCE(k.context_team_id::text,k.owner_team_id::text,''),k.created_at
FROM access_api_keys k
JOIN management_principal_user_links l ON l.namespace_id=k.namespace_id AND l.principal_id=$2
JOIN access_users u ON u.namespace_id=l.namespace_id AND u.id=l.user_id
JOIN self_service_policies p ON p.namespace_id=k.namespace_id
LEFT JOIN access_teams t ON t.namespace_id=k.namespace_id AND t.id=COALESCE(k.context_team_id,k.owner_team_id)
LEFT JOIN access_team_memberships m ON m.namespace_id=k.namespace_id
  AND m.team_id=COALESCE(k.context_team_id,k.owner_team_id) AND m.user_id=l.user_id
JOIN LATERAL (
  SELECT candidate.subject_id
  FROM (VALUES
    (1,k.id),
    (2,l.user_id),
    (3,COALESCE(k.context_team_id,k.owner_team_id))
  ) AS candidate(priority,subject_id)
  WHERE candidate.subject_id IS NOT NULL AND EXISTS (
    SELECT 1 FROM access_policy_bindings inheritance_binding
    WHERE inheritance_binding.namespace_id=k.namespace_id
      AND inheritance_binding.subject_id=candidate.subject_id
      AND inheritance_binding.status='active'
  )
  ORDER BY candidate.priority LIMIT 1
) effective_access ON TRUE
WHERE k.namespace_id=$1 AND k.status='active' AND k.deleted_at IS NULL
  AND (k.expires_at IS NULL OR k.expires_at>clock_timestamp()) AND u.status='active'
  AND ((k.owner_user_id=l.user_id AND (k.context_team_id IS NULL OR (t.status='active' AND m.status='active')))
       OR (k.owner_team_id IS NOT NULL AND p.allow_team_key_delegation AND t.status='active' AND m.status='active'))
  AND (NOT $4::boolean
       OR ($3::uuid IS NULL AND COALESCE(k.context_team_id,k.owner_team_id) IS NULL)
       OR ($3::uuid IS NOT NULL AND COALESCE(k.context_team_id,k.owner_team_id)=$3))
	AND ($8::uuid IS NULL OR k.id=$8)
  AND NOT EXISTS (
    SELECT 1 FROM unnest($7::text[]) AS required_permission(value)
    WHERE NOT EXISTS (
      SELECT 1
      FROM access_policy_bindings binding
      JOIN access_policies policy ON policy.namespace_id=binding.namespace_id AND policy.id=binding.policy_id
      JOIN access_policy_grants grant_record ON grant_record.policy_id=policy.id
      WHERE binding.namespace_id=k.namespace_id AND binding.subject_id=effective_access.subject_id
        AND binding.status='active' AND policy.status='active'
        AND grant_record.resource_type=$5 AND grant_record.resource_id=$6
        AND grant_record.permission=required_permission.value AND grant_record.effect='allow'
    ) OR EXISTS (
      SELECT 1
      FROM access_policy_bindings binding
      JOIN access_policies policy ON policy.namespace_id=binding.namespace_id AND policy.id=binding.policy_id
      JOIN access_policy_grants grant_record ON grant_record.policy_id=policy.id
      WHERE binding.namespace_id=k.namespace_id AND binding.subject_id=effective_access.subject_id
        AND binding.status='active' AND policy.status='active'
        AND grant_record.resource_type=$5 AND grant_record.resource_id=$6
        AND grant_record.permission=required_permission.value AND grant_record.effect='deny'
    )
  )
ORDER BY (k.owner_user_id IS NOT NULL) DESC,k.created_at,k.id LIMIT 1`
