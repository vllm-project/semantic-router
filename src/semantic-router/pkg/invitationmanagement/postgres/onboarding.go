package postgres

import (
	"context"
	"crypto/subtle"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

type lockedInvitation struct {
	invitation      invitationmanagement.Invitation
	tokenHMAC       []byte
	pepperVersion   string
	authSourceKind  string
	authSourceID    string
	evidenceKind    string
	response        accesscredential.Envelope
	resultExpiresAt time.Time
	deliveredAt     sql.NullTime
	erasedAt        sql.NullTime
}

type acceptanceSessionHooks struct {
	create func(context.Context, *sql.Tx, string, time.Time) (string, error)
	replay func(context.Context, *sql.Tx, string, string, time.Time) error
}

func acceptInTransaction(ctx context.Context, tx *sql.Tx, mutation invitationmanagement.AcceptMutation,
	hooks acceptanceSessionHooks,
) (invitationmanagement.AcceptanceEnvelope, error) {
	locked, acceptInTransactionErr := loadLockedInvitation(ctx, tx, mutation.InvitationID)
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, acceptInTransactionErr
	}
	now, acceptInTransactionErr := databaseNow(ctx, tx)
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, acceptInTransactionErr
	}
	if !validAcceptanceProof(locked, mutation) {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrIdentityMismatch
	}
	switch locked.invitation.Status {
	case invitationmanagement.StatusAccepted:
		return replayAcceptedInvitation(ctx, tx, mutation, locked, hooks, now)
	case invitationmanagement.StatusRevoked:
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrConflict
	case invitationmanagement.StatusExpired:
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrExpired
	case invitationmanagement.StatusPending:
		return acceptPendingInvitation(ctx, tx, mutation, locked, hooks, now)
	default:
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrUnavailable
	}
}

func validAcceptanceProof(locked lockedInvitation, mutation invitationmanagement.AcceptMutation) bool {
	return len(mutation.TokenHMAC) == 32 && len(locked.tokenHMAC) == 32 &&
		subtle.ConstantTimeCompare(mutation.TokenHMAC, locked.tokenHMAC) == 1 &&
		mutation.PepperVersion == locked.pepperVersion &&
		acceptanceIdentityMatches(locked.invitation.Expected, mutation.Identity)
}

func replayAcceptedInvitation(
	ctx context.Context,
	tx *sql.Tx,
	mutation invitationmanagement.AcceptMutation,
	locked lockedInvitation,
	hooks acceptanceSessionHooks,
	now time.Time,
) (invitationmanagement.AcceptanceEnvelope, error) {
	principalID, err := findPrincipal(ctx, tx, mutation.Identity.Issuer, mutation.Identity.Subject)
	if err != nil || principalID != locked.invitation.AcceptedPrincipalID ||
		locked.authSourceKind != mutation.AuthenticationSourceKind ||
		locked.authSourceID != mutation.AuthenticationSourceID || locked.evidenceKind != mutation.EvidenceKind {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrIdentityMismatch
	}
	if !now.Before(locked.resultExpiresAt) || locked.erasedAt.Valid || len(locked.response.Ciphertext) == 0 {
		if err := eraseAcceptanceResult(ctx, tx, mutation.InvitationID, now); err != nil {
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		return invitationmanagement.AcceptanceEnvelope{}, commitThen(invitationmanagement.ErrSecretExpired)
	}
	if hooks.replay == nil || locked.invitation.AcceptedManagementSessionID == "" {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrUnavailable
	}
	if err := hooks.replay(ctx, tx, locked.invitation.AcceptedManagementSessionID, principalID, now); err != nil {
		return invitationmanagement.AcceptanceEnvelope{}, err
	}
	return invitationmanagement.AcceptanceEnvelope{
		Invitation: locked.invitation,
		Envelope:   locked.response, ExpiresAt: locked.resultExpiresAt, Replayed: true,
	}, nil
}

func acceptPendingInvitation(
	ctx context.Context,
	tx *sql.Tx,
	mutation invitationmanagement.AcceptMutation,
	locked lockedInvitation,
	hooks acceptanceSessionHooks,
	now time.Time,
) (invitationmanagement.AcceptanceEnvelope, error) {
	if !now.Before(locked.invitation.ExpiresAt) {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrExpired
	}
	if mutation.SealResult == nil || hooks.create == nil ||
		len(mutation.RoleBindingIDs) != len(locked.invitation.Snapshot.RoleGrants) {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrInvalidRequest
	}
	if err := verifySnapshot(ctx, tx, locked.invitation.NamespaceID,
		locked.invitation.CreatedByPrincipalID, mutation.UserID, locked.invitation.Snapshot); err != nil {
		return invitationmanagement.AcceptanceEnvelope{}, err
	}
	principalID, acceptInTransactionErr := resolveInvitationPrincipal(ctx, tx, mutation, locked.invitation, now)
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, acceptInTransactionErr
	}
	actor := mutation.Actor
	actor.PrincipalID, actor.ActorChain = principalID, []string{principalID}
	if err := materializeOnboarding(ctx, tx, materialization{
		NamespaceID: locked.invitation.NamespaceID, PrincipalID: principalID, UserID: mutation.UserID,
		Email: mutation.Identity.VerifiedEmail, DisplayName: locked.invitation.DisplayName,
		Snapshot: locked.invitation.Snapshot, RoleBindingIDs: mutation.RoleBindingIDs,
		AccessBindingID: mutation.AccessBindingID, RateLimitBindingID: mutation.RateLimitBindingID,
		FirstKey: mutation.FirstKey, Now: now,
	}); err != nil {
		return invitationmanagement.AcceptanceEnvelope{}, err
	}
	sessionID, acceptInTransactionErr := hooks.create(ctx, tx, principalID, now)
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, acceptInTransactionErr
	}
	result := acceptanceResult(locked.invitation, principalID, mutation.UserID)
	envelope, resultExpiresAt, acceptInTransactionErr := mutation.SealResult(result)
	if acceptInTransactionErr != nil || !resultExpiresAt.After(now) {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrUnavailable
	}
	return completePendingInvitationAcceptance(
		ctx, tx, mutation, locked, actor, principalID, sessionID, envelope, resultExpiresAt, now,
	)
}

func acceptanceResult(
	invitation invitationmanagement.Invitation,
	principalID string,
	userID string,
) invitationmanagement.AcceptanceResult {
	result := invitationmanagement.AcceptanceResult{
		InvitationID: invitation.ID, PrincipalID: principalID, UserID: userID,
	}
	if invitation.Snapshot.Team != nil {
		result.TeamID = invitation.Snapshot.Team.TeamID
	}
	return result
}

func completePendingInvitationAcceptance(
	ctx context.Context,
	tx *sql.Tx,
	mutation invitationmanagement.AcceptMutation,
	locked lockedInvitation,
	actor invitationmanagement.Actor,
	principalID string,
	sessionID string,
	envelope accesscredential.Envelope,
	resultExpiresAt time.Time,
	now time.Time,
) (invitationmanagement.AcceptanceEnvelope, error) {
	updated, acceptInTransactionErr := scanInvitation(tx.QueryRowContext(ctx, `UPDATE management_invitations
SET status='accepted',accepted_principal_id=$2,accepted_user_id=$3,
    accepted_management_session_id=$4,
    accepted_auth_source_kind=$5,accepted_auth_source_id=$6,accepted_evidence_kind=$7,
    accepted_at=$8,acceptance_response_ciphertext=$9,acceptance_response_nonce=$10,
    acceptance_response_kek_version=$11,acceptance_result_expires_at=$12,
    revision=revision+1,updated_at=$8
WHERE id=$1 AND status='pending' AND revision=$13
RETURNING `+invitationColumns, mutation.InvitationID, principalID, mutation.UserID, sessionID,
		mutation.AuthenticationSourceKind, mutation.AuthenticationSourceID, mutation.EvidenceKind,
		now, envelope.Ciphertext, envelope.Nonce, envelope.KeyVersion, resultExpiresAt,
		locked.invitation.Revision))
	if errors.Is(acceptInTransactionErr, sql.ErrNoRows) {
		return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrConflict
	}
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, mapWriteError(acceptInTransactionErr, "accept invitation")
	}
	desiredRevision, acceptInTransactionErr := appendPublication(ctx, tx, updated.NamespaceID,
		"onboarding", mutation.UserID, 1, actor)
	if acceptInTransactionErr != nil {
		return invitationmanagement.AcceptanceEnvelope{}, acceptInTransactionErr
	}
	before := locked.invitation.Revision
	if err := appendAudit(ctx, tx, updated.NamespaceID, &desiredRevision, "invitation.accepted",
		"invitation", updated.ID, &before, updated.Revision, actor); err != nil {
		return invitationmanagement.AcceptanceEnvelope{}, err
	}
	return invitationmanagement.AcceptanceEnvelope{
		Invitation: updated, Envelope: envelope,
		ExpiresAt: resultExpiresAt,
	}, nil
}

func markAcceptanceDeliveredInTransaction(ctx context.Context, tx *sql.Tx, invitationID string,
	actor invitationmanagement.Actor,
) error {
	locked, markAcceptanceDeliveredInTransactionErr := loadLockedInvitation(ctx, tx, invitationID)
	if markAcceptanceDeliveredInTransactionErr != nil {
		return markAcceptanceDeliveredInTransactionErr
	}
	now, markAcceptanceDeliveredInTransactionErr := databaseNow(ctx, tx)
	if markAcceptanceDeliveredInTransactionErr != nil {
		return markAcceptanceDeliveredInTransactionErr
	}
	if locked.invitation.Status != invitationmanagement.StatusAccepted ||
		locked.invitation.AcceptedPrincipalID != actor.PrincipalID {
		return invitationmanagement.ErrIdentityMismatch
	}
	if !now.Before(locked.resultExpiresAt) || locked.erasedAt.Valid || len(locked.response.Ciphertext) == 0 {
		if err := eraseAcceptanceResult(ctx, tx, invitationID, now); err != nil {
			return err
		}
		return commitThen(invitationmanagement.ErrSecretExpired)
	}
	if locked.deliveredAt.Valid {
		return nil
	}
	result, markAcceptanceDeliveredInTransactionErr := tx.ExecContext(ctx, `UPDATE management_invitations
SET acceptance_result_delivered_at=$2,updated_at=updated_at
WHERE id=$1 AND acceptance_result_delivered_at IS NULL`, invitationID, now)
	if markAcceptanceDeliveredInTransactionErr != nil {
		return fmt.Errorf("mark invitation result delivered: %w", markAcceptanceDeliveredInTransactionErr)
	}
	rows, markAcceptanceDeliveredInTransactionErr := result.RowsAffected()
	if markAcceptanceDeliveredInTransactionErr != nil || rows != 1 {
		return invitationmanagement.ErrConflict
	}
	revision := locked.invitation.Revision
	return appendAudit(ctx, tx, locked.invitation.NamespaceID, nil,
		"invitation.result_delivered", "invitation", invitationID, &revision, revision, actor)
}

func (store *Store) Onboard(ctx context.Context, mutation invitationmanagement.PrivilegedOnboardingMutation) (invitationmanagement.AcceptanceEnvelope, error) {
	return inTransaction(ctx, store, sql.LevelSerializable, func(tx *sql.Tx) (invitationmanagement.AcceptanceEnvelope, error) {
		stored, replayed, onboardErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if onboardErr != nil {
			return invitationmanagement.AcceptanceEnvelope{}, mapCommandError(onboardErr)
		}
		if replayed {
			secret, err := storedSecret(stored, "onboarding")
			if err != nil {
				return invitationmanagement.AcceptanceEnvelope{}, err
			}
			fake := invitationmanagement.Invitation{
				NamespaceID:         mutation.NamespaceID,
				AcceptedPrincipalID: mutation.PrincipalID, AcceptedUserID: secret.Result.ResourceID,
				Status: invitationmanagement.StatusAccepted, Revision: secret.Result.ResourceRevision,
			}
			return invitationmanagement.AcceptanceEnvelope{
				Invitation: fake,
				Envelope: accesscredential.Envelope{
					KeyVersion: secret.Secret.KEKVersion,
					Nonce:      secret.Secret.Nonce, Ciphertext: secret.Secret.Ciphertext,
				},
				ExpiresAt: secret.Secret.ExpiresAt, Replayed: true,
			}, nil
		}
		now, onboardErr := databaseNow(ctx, tx)
		if onboardErr != nil {
			return invitationmanagement.AcceptanceEnvelope{}, onboardErr
		}
		if mutation.SealResult == nil || len(mutation.RoleBindingIDs) != len(mutation.Snapshot.RoleGrants) {
			return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrInvalidRequest
		}
		if err := verifySnapshot(ctx, tx, mutation.NamespaceID, mutation.Actor.PrincipalID,
			mutation.UserID, mutation.Snapshot); err != nil {
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		var active bool
		if err := tx.QueryRowContext(ctx, `SELECT status='active' FROM management_principals
WHERE id=$1 FOR UPDATE`, mutation.PrincipalID).Scan(&active); err != nil || !active {
			if errors.Is(err, sql.ErrNoRows) || err == nil {
				return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrNotFound
			}
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		if err := materializeOnboarding(ctx, tx, materialization{
			NamespaceID: mutation.NamespaceID, PrincipalID: mutation.PrincipalID, UserID: mutation.UserID,
			Email: mutation.Email, DisplayName: mutation.DisplayName, Snapshot: mutation.Snapshot,
			RoleBindingIDs: mutation.RoleBindingIDs, AccessBindingID: mutation.AccessBindingID,
			RateLimitBindingID: mutation.RateLimitBindingID, FirstKey: mutation.FirstKey, Now: now,
		}); err != nil {
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		result := invitationmanagement.AcceptanceResult{PrincipalID: mutation.PrincipalID, UserID: mutation.UserID}
		if mutation.Snapshot.Team != nil {
			result.TeamID = mutation.Snapshot.Team.TeamID
		}
		envelope, expiresAt, onboardErr := mutation.SealResult(result)
		if onboardErr != nil || !expiresAt.After(now) {
			return invitationmanagement.AcceptanceEnvelope{}, invitationmanagement.ErrUnavailable
		}
		desiredRevision, onboardErr := appendPublication(ctx, tx, mutation.NamespaceID,
			"onboarding", mutation.UserID, 1, mutation.Actor)
		if onboardErr != nil {
			return invitationmanagement.AcceptanceEnvelope{}, onboardErr
		}
		if err := appendAudit(ctx, tx, mutation.NamespaceID, &desiredRevision, "onboarding.created",
			"onboarding", mutation.UserID, nil, 1, mutation.Actor); err != nil {
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		resource := managementcommand.ResourceResult{
			ResourceType: "onboarding", ResourceID: mutation.UserID,
			ResourceRevision: 1, ResponseStatus: 201,
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command, resource,
			managementcommand.SecretResponse{
				Ciphertext: envelope.Ciphertext, Nonce: envelope.Nonce,
				KEKVersion: envelope.KeyVersion, ExpiresAt: expiresAt,
			}); err != nil {
			return invitationmanagement.AcceptanceEnvelope{}, err
		}
		fake := invitationmanagement.Invitation{
			NamespaceID:         mutation.NamespaceID,
			AcceptedPrincipalID: mutation.PrincipalID, AcceptedUserID: mutation.UserID,
			Status: invitationmanagement.StatusAccepted, Revision: 1,
		}
		return invitationmanagement.AcceptanceEnvelope{
			Invitation: fake, Envelope: envelope,
			ExpiresAt: expiresAt,
		}, nil
	})
}

type materialization struct {
	NamespaceID        string
	PrincipalID        string
	UserID             string
	Email              string
	DisplayName        string
	Snapshot           invitationmanagement.OnboardingSnapshot
	RoleBindingIDs     []string
	AccessBindingID    string
	RateLimitBindingID string
	FirstKey           *invitationmanagement.PreparedFirstKey
	Now                time.Time
}

func materializeOnboarding(ctx context.Context, tx *sql.Tx, value materialization) error {
	teamInheritance := value.Snapshot.Team != nil
	if (teamInheritance && (value.AccessBindingID != "" || value.RateLimitBindingID != "")) ||
		(!teamInheritance && (value.AccessBindingID == "" || value.RateLimitBindingID == "")) {
		return invitationmanagement.ErrInvalidRequest
	}
	var conflicts bool
	if err := tx.QueryRowContext(ctx, `SELECT
  EXISTS(SELECT 1 FROM management_principal_user_links WHERE principal_id=$1 AND namespace_id=$2)
  OR EXISTS(SELECT 1 FROM access_users WHERE namespace_id=$2 AND email=$3)`,
		value.PrincipalID, value.NamespaceID, value.Email).Scan(&conflicts); err != nil {
		return fmt.Errorf("check onboarding conflicts: %w", err)
	}
	if conflicts {
		return invitationmanagement.ErrAlreadyAccepted
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_subjects (namespace_id,id,kind,created_at)
VALUES ($1,$2,'user',$3)`, value.NamespaceID, value.UserID, value.Now); err != nil {
		return mapWriteError(err, "create onboarding User subject")
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_users
  (id,namespace_id,email,display_name,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)`, value.UserID, value.NamespaceID,
		value.Email, value.DisplayName, value.Now); err != nil {
		return mapWriteError(err, "create onboarding User")
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_principal_user_links
  (principal_id,namespace_id,user_id,revision,created_at,updated_at)
VALUES ($1,$2,$3,1,$4,$4)`, value.PrincipalID, value.NamespaceID, value.UserID, value.Now); err != nil {
		return mapWriteError(err, "link onboarding principal")
	}
	for index, grant := range value.Snapshot.RoleGrants {
		ceiling, _ := json.Marshal(grant.DelegationCeiling)
		scopeKind, resourceID := grant.ScopeKind, any(nil)
		if grant.ScopeKind == "user" {
			resourceID = value.UserID
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_role_bindings
  (id,principal_id,role_id,scope_kind,namespace_id,resource_type,resource_id,
   delegation_ceiling,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,NULL,$6,$7,'active',1,$8,$8)`, value.RoleBindingIDs[index],
			value.PrincipalID, grant.RoleID, scopeKind, value.NamespaceID, resourceID, ceiling, value.Now); err != nil {
			return mapWriteError(err, "create onboarding role binding")
		}
	}
	if value.Snapshot.Team != nil {
		if _, err := tx.ExecContext(ctx, `INSERT INTO access_team_memberships
  (namespace_id,team_id,user_id,role,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,'active',1,$5,$5)`, value.NamespaceID, value.Snapshot.Team.TeamID,
			value.UserID, value.Snapshot.Team.Role, value.Now); err != nil {
			return mapWriteError(err, "create onboarding Team membership")
		}
	}
	if !teamInheritance {
		if _, err := tx.ExecContext(ctx, `INSERT INTO access_policy_bindings
	  (id,namespace_id,policy_id,subject_id,status,revision,created_at,updated_at)
	VALUES ($1,$2,$3,$4,'active',1,$5,$5)`, value.AccessBindingID, value.NamespaceID,
			value.Snapshot.AccessPolicyID, value.UserID, value.Now); err != nil {
			return mapWriteError(err, "create onboarding AccessPolicy binding")
		}
		result, err := tx.ExecContext(ctx, `INSERT INTO rate_limit_bindings
	  (id,namespace_id,policy_id,subject_id,binding_mode,quota_partition_id,status,revision,created_at,updated_at)
	SELECT $1,$2,$3,$4,'allocation',quota_partition_id,'active',1,$5,$5
	FROM access_namespaces WHERE id=$2 AND status='active'`, value.RateLimitBindingID,
			value.NamespaceID, value.Snapshot.RateLimitPolicyID, value.UserID, value.Now)
		if err != nil {
			return mapWriteError(err, "create onboarding RateLimitPolicy binding")
		}
		rows, err := result.RowsAffected()
		if err != nil || rows != 1 {
			return invitationmanagement.ErrDefaultsChanged
		}
	}
	if value.Snapshot.AutomaticFirstKey {
		if value.FirstKey == nil {
			return invitationmanagement.ErrUnavailable
		}
		if err := insertFirstKey(ctx, tx, value); err != nil {
			return err
		}
	} else if value.FirstKey != nil {
		return invitationmanagement.ErrInvalidRequest
	}
	return nil
}

func insertFirstKey(ctx context.Context, tx *sql.Tx, value materialization) error {
	key, credential := value.FirstKey.Key, value.FirstKey.Credential
	expectedTeamID := ""
	if value.Snapshot.Team != nil {
		expectedTeamID = value.Snapshot.Team.TeamID
	}
	if key.Validate() != nil || credential.Validate() != nil || string(key.NamespaceID) != value.NamespaceID ||
		string(key.Owner.ID) != value.UserID || key.Owner.Kind != accesscontrol.SubjectKindUser ||
		string(key.ContextTeamID) != expectedTeamID || credential.APIKeyID != key.ID {
		return invitationmanagement.ErrInvalidRequest
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_subjects (namespace_id,id,kind,created_at)
VALUES ($1,$2,'api_key',$3)`, value.NamespaceID, key.ID, value.Now); err != nil {
		return mapWriteError(err, "create onboarding API-key subject")
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_api_keys
  (id,namespace_id,name,owner_user_id,owner_team_id,context_team_id,status,
   policy_epoch,delegation_epoch,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,NULL,NULLIF($5,'')::uuid,'active',1,1,1,$6,$6)`, key.ID,
		value.NamespaceID, key.Name, value.UserID, key.ContextTeamID, value.Now); err != nil {
		return mapWriteError(err, "create onboarding API key")
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_api_key_credentials
  (id,namespace_id,api_key_id,kid,secret_hmac,pepper_version,secret_ciphertext,
   ciphertext_nonce,kek_version,status,not_before,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,'active',$10,$10)`, credential.ID, value.NamespaceID,
		key.ID, credential.KID, credential.SecretHMAC, credential.PepperVersion,
		optionalFirstKeyBytes(credential.SecretCiphertext), optionalFirstKeyBytes(credential.CiphertextNonce),
		optionalFirstKeyString(credential.KEKVersion), value.Now); err != nil {
		return mapWriteError(err, "create onboarding API-key credential")
	}
	return nil
}

func optionalFirstKeyBytes(value []byte) any {
	if len(value) == 0 {
		return nil
	}
	return value
}

func optionalFirstKeyString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func resolveInvitationPrincipal(ctx context.Context, tx *sql.Tx, mutation invitationmanagement.AcceptMutation,
	invitation invitationmanagement.Invitation, now time.Time,
) (string, error) {
	principalID, findPrincipalErr := findPrincipal(ctx, tx, mutation.Identity.Issuer, mutation.Identity.Subject)
	if findPrincipalErr == nil {
		var active bool
		var verifiedEmail string
		if err := tx.QueryRowContext(ctx, `SELECT status='active',COALESCE(verified_email,'')
FROM management_principals WHERE id=$1 FOR UPDATE`, principalID).Scan(&active, &verifiedEmail); err != nil || !active {
			return "", invitationmanagement.ErrConflict
		}
		if verifiedEmail != "" && verifiedEmail != mutation.Identity.VerifiedEmail {
			return "", invitationmanagement.ErrIdentityMismatch
		}
		return principalID, nil
	}
	if !errors.Is(findPrincipalErr, invitationmanagement.ErrNotFound) {
		return "", findPrincipalErr
	}
	displayName := invitation.DisplayName
	if displayName == "" {
		displayName = mutation.Identity.DisplayName
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_principals
  (id,issuer,subject,display_name,verified_email,attributes,status,revision,created_at,updated_at)
VALUES ($1,$2,$3,$4,$5,'{}','active',1,$6,$6)`, mutation.PrincipalID,
		mutation.Identity.Issuer, mutation.Identity.Subject, displayName,
		mutation.Identity.VerifiedEmail, now); err != nil {
		return "", mapWriteError(err, "create invited Management principal")
	}
	return mutation.PrincipalID, nil
}

func findPrincipal(ctx context.Context, tx *sql.Tx, issuer, subject string) (string, error) {
	var id string
	err := tx.QueryRowContext(ctx, `SELECT id::text FROM management_principals
WHERE issuer=$1 AND subject=$2`, issuer, subject).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return "", invitationmanagement.ErrNotFound
	}
	if err != nil {
		return "", err
	}
	return id, nil
}

func loadLockedInvitation(ctx context.Context, tx *sql.Tx, invitationID string) (lockedInvitation, error) {
	var value lockedInvitation
	var deliveredAt, erasedAt, resultExpiresAt sql.NullTime
	destinations := []any{
		&value.invitation.ID, &value.invitation.NamespaceID, &value.invitation.CreatedByPrincipalID,
		&value.invitation.Expected.Issuer, &value.invitation.Expected.Subject, &value.invitation.Expected.Email,
		&value.invitation.DisplayName,
	}
	var snapshotJSON []byte
	destinations = append(destinations, &snapshotJSON, &value.invitation.ExpiresAt, &value.invitation.Status)
	var acceptedPrincipal, acceptedUser, acceptedSession sql.NullString
	var acceptedAt sql.NullTime
	destinations = append(destinations, &acceptedPrincipal, &acceptedUser, &acceptedSession, &acceptedAt,
		&value.invitation.Revision, &value.invitation.CreatedAt, &value.invitation.UpdatedAt,
		&value.tokenHMAC, &value.pepperVersion, &value.authSourceKind, &value.authSourceID,
		&value.evidenceKind, &value.response.Ciphertext, &value.response.Nonce,
		&value.response.KeyVersion, &resultExpiresAt, &deliveredAt, &erasedAt)
	err := tx.QueryRowContext(ctx, `SELECT `+invitationColumns+`,token_hmac,pepper_version,
       COALESCE(accepted_auth_source_kind,''),COALESCE(accepted_auth_source_id,''),
       COALESCE(accepted_evidence_kind,''),acceptance_response_ciphertext,
       acceptance_response_nonce,COALESCE(acceptance_response_kek_version,''),
       acceptance_result_expires_at,acceptance_result_delivered_at,acceptance_result_erased_at
FROM management_invitations WHERE id=$1 FOR UPDATE`, invitationID).Scan(destinations...)
	if errors.Is(err, sql.ErrNoRows) {
		return lockedInvitation{}, invitationmanagement.ErrNotFound
	}
	if err != nil {
		return lockedInvitation{}, fmt.Errorf("lock invitation acceptance: %w", err)
	}
	if json.Unmarshal(snapshotJSON, &value.invitation.Snapshot) != nil {
		return lockedInvitation{}, invitationmanagement.ErrUnavailable
	}
	value.invitation.AcceptedPrincipalID, value.invitation.AcceptedUserID = acceptedPrincipal.String, acceptedUser.String
	value.invitation.AcceptedManagementSessionID = acceptedSession.String
	if acceptedAt.Valid {
		timestamp := acceptedAt.Time.UTC()
		value.invitation.AcceptedAt = &timestamp
	}
	value.deliveredAt, value.erasedAt = deliveredAt, erasedAt
	value.invitation.ExpiresAt = value.invitation.ExpiresAt.UTC()
	if resultExpiresAt.Valid {
		value.resultExpiresAt = resultExpiresAt.Time.UTC()
	}
	return value, nil
}

func eraseAcceptanceResult(ctx context.Context, tx *sql.Tx, invitationID string, now time.Time) error {
	_, err := tx.ExecContext(ctx, `UPDATE management_invitations
SET acceptance_response_ciphertext=NULL,acceptance_response_nonce=NULL,
    acceptance_response_kek_version=NULL,acceptance_result_erased_at=COALESCE(acceptance_result_erased_at,$2)
WHERE id=$1 AND status='accepted' AND acceptance_result_expires_at<=$2`, invitationID, now)
	if err != nil {
		return fmt.Errorf("erase expired invitation result: %w", err)
	}
	return nil
}

func acceptanceIdentityMatches(expected invitationmanagement.ExpectedIdentity, actual invitationmanagement.AcceptanceIdentity) bool {
	return expected.Issuer == actual.Issuer && (expected.Subject == "" || expected.Subject == actual.Subject) &&
		(expected.Email == "" || expected.Email == actual.VerifiedEmail)
}
