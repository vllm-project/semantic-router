package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauthorization"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
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
		peppers: options.Peppers, secrets: options.Secrets,
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
			request.EffectiveTeamID, userID, target,
		)
		if prepareErr != nil {
			return agentmanagement.SessionAuthorization{}, prepareErr
		}
		return agentmanagement.SessionAuthorization{
			EffectiveUserID: userID, EffectiveTeamID: key.TeamID,
			TargetKind: target.Kind, TargetResourceID: target.ResourceID,
		}, nil
	})
}

func (authority *AgentSessionAuthority) Bootstrap(
	ctx context.Context, request agentmanagement.SessionBootstrapRequest,
) (agentmanagement.Session, bool, error) {
	if authority == nil || uuid.Validate(request.SessionID) != nil || uuid.Validate(request.NamespaceID) != nil ||
		uuid.Validate(request.PrincipalID) != nil || uuid.Validate(request.Mutation.ManagementSessionID) != nil ||
		request.PrincipalID != request.Mutation.PrincipalID || request.SessionTTL <= 0 {
		return agentmanagement.Session{}, false, agentmanagement.ErrInvalid
	}
	now := authority.now().UTC()
	result, err := inTransaction(ctx, authority.store, func(tx *sql.Tx) (agentBootstrapResult, error) {
		stored, replayed, bootstrapErr := commandpostgres.Lock(ctx, tx, request.Command)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		if replayed {
			if stored.Resource == nil || stored.Resource.ResourceType != "agent_session" {
				return agentBootstrapResult{}, agentmanagement.ErrConflict
			}
			session, delegation, desired, err := loadAgentSessionBootstrapReplay(
				ctx, tx, request.NamespaceID, stored.Resource.ResourceID,
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
		target, bootstrapErr := resolveAgentTarget(ctx, tx, request.NamespaceID, request.Target)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		if err := verifyTargetCapabilities(ctx, tx, request.NamespaceID, target,
			request.Profile.MinimumTargetCapabilities); err != nil {
			return agentBootstrapResult{}, err
		}
		var profileStatus string
		var profileContentRevision int64
		if err := tx.QueryRowContext(ctx, `SELECT status,current_revision FROM agent_profiles
WHERE namespace_id=$1 AND id=$2 FOR KEY SHARE`, request.NamespaceID, request.Profile.ID).Scan(
			&profileStatus, &profileContentRevision); err != nil {
			return agentBootstrapResult{}, agentNotFound(err)
		}
		if profileStatus != string(agentmanagement.StatusActive) ||
			profileContentRevision != request.Profile.ContentRevision {
			return agentBootstrapResult{}, agentmanagement.ErrConflict
		}
		delegation := delegationmanagement.Session{
			ID: uuid.NewString(), NamespaceID: request.NamespaceID,
			ManagementSessionID: request.Mutation.ManagementSessionID,
			PrincipalID:         request.PrincipalID,
		}
		delegation.PublicID = delegation.ID
		self, bootstrapErr := lockDelegationSelf(ctx, tx, delegation)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, agentDenied(bootstrapErr)
		}
		key, bootstrapErr := selectAgentInferenceKey(ctx, tx, request.NamespaceID, request.PrincipalID,
			request.EffectiveTeamID, self.UserID, target)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		authorityDigest, bootstrapErr := authority.authorizeAgentUse(
			ctx, tx, request.NamespaceID, request.PrincipalID, self.UserID, key.TeamID,
		)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		expiresAt := now.Add(request.SessionTTL)
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
		if !expiresAt.After(now) {
			return agentBootstrapResult{}, agentmanagement.ErrDenied
		}
		var activeCount int
		if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM delegated_inference_sessions
WHERE namespace_id=$1 AND user_id=$2 AND status='active' AND expires_at>clock_timestamp()`,
			request.NamespaceID, self.UserID).Scan(&activeCount); err != nil {
			return agentBootstrapResult{}, fmt.Errorf("count Agent delegations: %w", err)
		}
		if self.Policy.MaxDelegatedSessions <= 0 || activeCount >= self.Policy.MaxDelegatedSessions {
			return agentBootstrapResult{}, agentmanagement.ErrDenied
		}
		issued, bootstrapErr := authority.peppers.Issue(accesscredential.KindDelegation, delegation.ID)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, agentmanagement.ErrToolUnavailable
		}
		plaintext := []byte(issued.Plaintext)
		defer clear(plaintext)
		encrypted, bootstrapErr := authority.secrets.Encrypt(ctx, plaintext)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, agentmanagement.ErrToolUnavailable
		}
		delegation.QuotaPartition = self.QuotaPartition
		delegation.APIKeyID = key.KeyID
		delegation.DelegationEpoch = key.DelegationEpoch
		delegation.UserID = self.UserID
		delegation.TeamID = key.TeamID
		delegation.TokenHMAC = append([]byte(nil), issued.Digest.HMAC...)
		delegation.PepperVersion = issued.Digest.PepperVersion
		delegation.Audience = authority.audience
		delegation.Status = delegationmanagement.SessionActive
		delegation.NotBefore = now
		delegation.ExpiresAt = expiresAt
		delegation.Revision = 1
		delegation.CreatedAt = now
		if _, err := tx.ExecContext(ctx, `INSERT INTO delegated_inference_sessions
  (id,public_id,namespace_id,management_session_id,principal_id,api_key_id,delegation_epoch,
   user_id,team_id,token_hmac,pepper_version,audience,status,not_before,expires_at,revision,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'active',$13,$14,1,$15)`,
			delegation.ID, delegation.PublicID, delegation.NamespaceID, delegation.ManagementSessionID,
			delegation.PrincipalID, delegation.APIKeyID, delegation.DelegationEpoch, delegation.UserID,
			nullableString(delegation.TeamID), delegation.TokenHMAC, delegation.PepperVersion,
			delegation.Audience, delegation.NotBefore, delegation.ExpiresAt, delegation.CreatedAt); err != nil {
			return agentBootstrapResult{}, mapAgentWrite(err)
		}
		var targetModel, targetEntrypoint any
		if target.Kind == agentmanagement.TargetModel {
			targetModel = target.ResourceID
		} else {
			targetEntrypoint = target.ResourceID
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_sessions
  (id,namespace_id,owner_principal_id,effective_user_id,effective_team_id,
   delegated_inference_session_id,profile_id,profile_revision,target_model_id,target_entrypoint_id,
   target_public_id,authority_digest,mode,title,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,'active',1,$15,$15)`,
			request.SessionID, request.NamespaceID, request.PrincipalID, self.UserID,
			nullableString(key.TeamID), delegation.ID, request.Profile.ID, request.Profile.ContentRevision,
			targetModel, targetEntrypoint, target.PublicID, authorityDigest,
			request.Mode, request.Title, now); err != nil {
			return agentBootstrapResult{}, mapAgentWrite(err)
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_session_inference_credentials
  (namespace_id,session_id,delegated_inference_session_id,secret_ciphertext,ciphertext_nonce,
   kek_version,expires_at,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$8)`, request.NamespaceID, request.SessionID, delegation.ID,
			encrypted.Ciphertext, encrypted.Nonce, encrypted.KEKVersion, expiresAt, now); err != nil {
			return agentBootstrapResult{}, mapAgentWrite(err)
		}
		meta, bootstrapErr := agentMutationMeta(request.Mutation, "agent.session.create", "Create Agent session delegation.")
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		receipt, bootstrapErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(request.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: 1, Operation: outboxCreated,
			References: map[string]string{"apiKeyId": delegation.APIKeyID},
		}, meta)
		if bootstrapErr != nil {
			return agentBootstrapResult{}, bootstrapErr
		}
		if err := commandpostgres.CompleteResource(ctx, tx, request.Command,
			managementcommand.ResourceResult{
				ResourceType: "agent_session", ResourceID: request.SessionID,
				ResourceRevision: 1, ResponseStatus: 201,
			}); err != nil {
			return agentBootstrapResult{}, err
		}
		session := agentmanagement.Session{
			ID: request.SessionID, NamespaceID: request.NamespaceID, OwnerPrincipalID: request.PrincipalID,
			EffectiveUserID: self.UserID, EffectiveTeamID: key.TeamID,
			DelegatedInferenceSessionID: delegation.ID, ProfileID: request.Profile.ID,
			ProfileRevision:  request.Profile.ContentRevision,
			Target:           agentmanagement.Target{Kind: target.Kind, ID: target.PublicID},
			TargetResourceID: target.ResourceID, AuthorityDigest: authorityDigest,
			Mode: request.Mode, Title: request.Title,
			Status: agentmanagement.SessionActive, Revision: 1, CreatedAt: now, UpdatedAt: now,
		}
		return agentBootstrapResult{
			session: session, delegation: delegation,
			desiredRevision: uint64(receipt.DesiredRevision),
		}, nil
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
			ctx, tx, namespaceID, principalID, nil, false, resolved,
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
		meta := MutationMeta{
			ActorPrincipalID: nil, RequestID: "agent-worker-" + session.ID,
			Action: "agent.session.renew", Reason: "Renew Agent session delegation.", Details: AuditDetails{},
		}
		receipt, renewDelegationErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(session.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: accesscontrol.Revision(revision), Operation: outboxUpdated,
			References: map[string]string{"apiKeyId": delegation.APIKeyID},
		}, meta)
		if renewDelegationErr != nil {
			return renewalResult{}, renewDelegationErr
		}
		delegation.TokenHMAC = append([]byte(nil), issued.Digest.HMAC...)
		delegation.PepperVersion = issued.Digest.PepperVersion
		delegation.NotBefore, delegation.ExpiresAt, delegation.Revision = now, expiresAt, uint64(revision)
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
		meta, closeErr := agentMutationMeta(mutation, "agent.session.close", "Close Agent session delegation.")
		if closeErr != nil {
			return closeResult{}, closeErr
		}
		receipt, closeErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(session.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: delegation.ID,
			AggregateRevision: accesscontrol.Revision(delegationRevision), Operation: outboxDeleted,
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

func selectAgentInferenceKey(
	ctx context.Context, tx *sql.Tx, namespaceID, principalID, effectiveTeamID string, _ string,
	target resolvedAgentTarget,
) (delegationmanagement.EligibleKey, error) {
	var team any
	if effectiveTeamID != "" {
		if uuid.Validate(effectiveTeamID) != nil {
			return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
		}
		team = effectiveTeamID
	}
	return queryAgentInferenceKey(
		ctx, tx, namespaceID, principalID, team, true, target,
		[]accesscontrol.GrantPermission{
			accesscontrol.GrantPermissionDiscover, accesscontrol.GrantPermissionInvoke,
		}, true,
	)
}

func selectAgentInferenceKeyRead(
	ctx context.Context, tx *sql.Tx, namespaceID, principalID, effectiveTeamID string, _ string,
	target resolvedAgentTarget,
) (delegationmanagement.EligibleKey, error) {
	var team any
	if effectiveTeamID != "" {
		if uuid.Validate(effectiveTeamID) != nil {
			return delegationmanagement.EligibleKey{}, agentmanagement.ErrInvalid
		}
		team = effectiveTeamID
	}
	return queryAgentInferenceKey(
		ctx, tx, namespaceID, principalID, team, true, target,
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
		target.Kind, target.ResourceID, pq.Array(values),
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
