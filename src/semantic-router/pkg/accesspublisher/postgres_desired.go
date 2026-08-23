package accesspublisher

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accessprojection"
	routingpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type PostgresDesiredStateReader struct {
	db *sql.DB
}

func NewPostgresDesiredStateReader(db *sql.DB) (*PostgresDesiredStateReader, error) {
	if db == nil {
		return nil, fmt.Errorf("PostgreSQL desired-state database is required")
	}
	return &PostgresDesiredStateReader{db: db}, nil
}

func (r *PostgresDesiredStateReader) LoadDesiredState(
	ctx context.Context,
	namespaceID string,
	desiredRevision uint64,
) (DesiredState, error) {
	if strings.TrimSpace(namespaceID) == "" || desiredRevision == 0 || desiredRevision > math.MaxInt64 {
		return DesiredState{}, fmt.Errorf("namespace and positive PostgreSQL revision are required")
	}
	tx, err := r.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return DesiredState{}, fmt.Errorf("begin desired-state snapshot: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	revisionTime, err := verifyDesiredRevision(ctx, tx, namespaceID, desiredRevision)
	if err != nil {
		return DesiredState{}, err
	}
	namespace, err := loadNamespace(ctx, tx, namespaceID)
	if err != nil {
		return DesiredState{}, err
	}
	state := DesiredState{
		Namespace: namespace, Revision: desiredRevision, RevisionTime: revisionTime,
		Routing: routingsnapshot.Bundle{
			NamespaceID: namespaceID, Revision: int64(desiredRevision), Currency: namespace.BillingCurrency,
		},
	}
	if namespace.Status == accesscontrol.NamespaceStatusDisabled {
		state.BarrierHints = append(state.BarrierHints, Barrier{
			Kind: "namespace", ResourceID: namespaceID, Reason: "namespace_disabled",
		})
	}

	accessData, err := loadAccessDesiredState(ctx, tx, namespace, desiredRevision)
	if err != nil {
		return DesiredState{}, err
	}
	state.Keys = accessData.keys
	state.Credentials = accessData.credentials
	state.BarrierHints = append(state.BarrierHints, accessData.barriers...)
	state.Routing, err = loadRoutingBundle(ctx, tx, namespace, desiredRevision)
	if err != nil {
		return DesiredState{}, err
	}
	state.ProviderCredentials, err = loadProviderCredentialCandidates(ctx, tx, namespace, state.Routing)
	if err != nil {
		return DesiredState{}, err
	}
	if err := tx.Commit(); err != nil {
		return DesiredState{}, fmt.Errorf("commit desired-state snapshot: %w", err)
	}
	return state, nil
}

func verifyDesiredRevision(ctx context.Context, tx *sql.Tx, namespaceID string, desired uint64) (time.Time, error) {
	var latest int64
	if err := tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(revision), 0) FROM policy_revisions WHERE namespace_id = $1`, namespaceID,
	).Scan(&latest); err != nil {
		return time.Time{}, fmt.Errorf("read latest desired revision: %w", err)
	}
	if latest > int64(desired) {
		return time.Time{}, ErrSuperseded
	}
	if latest != int64(desired) {
		return time.Time{}, fmt.Errorf("desired revision %d is absent; latest is %d", desired, latest)
	}
	var revisionTime time.Time
	if err := tx.QueryRowContext(ctx,
		`SELECT created_at FROM policy_revisions WHERE namespace_id = $1 AND revision = $2`, namespaceID, int64(desired),
	).Scan(&revisionTime); err != nil {
		return time.Time{}, fmt.Errorf("read desired revision timestamp: %w", err)
	}
	return revisionTime.UTC(), nil
}

func loadNamespace(ctx context.Context, tx *sql.Tx, namespaceID string) (accesscontrol.Namespace, error) {
	var namespace accesscontrol.Namespace
	var revision, epoch int64
	err := tx.QueryRowContext(ctx, `SELECT id, name, quota_partition_id, billing_currency, status,
revision, runtime_epoch, created_at, updated_at FROM access_namespaces WHERE id = $1`, namespaceID).Scan(
		&namespace.ID, &namespace.Name, &namespace.QuotaPartitionID, &namespace.BillingCurrency,
		&namespace.Status, &revision, &epoch, &namespace.CreatedAt, &namespace.UpdatedAt,
	)
	if err == sql.ErrNoRows {
		return accesscontrol.Namespace{}, fmt.Errorf("namespace %s does not exist", namespaceID)
	}
	if err != nil {
		return accesscontrol.Namespace{}, fmt.Errorf("read namespace: %w", err)
	}
	if revision <= 0 || epoch <= 0 {
		return accesscontrol.Namespace{}, fmt.Errorf("namespace revision or runtime epoch is invalid")
	}
	namespace.Revision, namespace.RuntimeEpoch = accesscontrol.Revision(revision), uint64(epoch)
	if err := namespace.Validate(); err != nil {
		return accesscontrol.Namespace{}, fmt.Errorf("validate namespace: %w", err)
	}
	return namespace, nil
}

type accessDesiredData struct {
	keys        []accessprojectionCandidate
	credentials []CredentialCandidate
	barriers    []Barrier
}

// Alias keeps the large assembler readable without weakening the public
// DesiredState contract.
type accessprojectionCandidate = accessprojection.Candidate

func loadAccessDesiredState(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	revision uint64,
) (accessDesiredData, error) {
	users, userBarriers, err := loadUsers(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	teams, teamBarriers, err := loadTeams(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	memberships, membershipBarriers, err := loadMemberships(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	subjects, err := loadSubjects(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	accessPolicies, err := loadAccessPolicies(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	ratePolicies, err := loadRatePolicies(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	accessBindings, err := loadAccessBindings(ctx, tx, namespace.ID, subjects)
	if err != nil {
		return accessDesiredData{}, err
	}
	rateBindings, err := loadRateBindings(ctx, tx, namespace, subjects)
	if err != nil {
		return accessDesiredData{}, err
	}
	claims, err := loadRoutingClaims(ctx, tx, namespace.ID)
	if err != nil {
		return accessDesiredData{}, err
	}
	keys, keyBarriers, included, err := loadKeyCandidates(
		ctx, tx, namespace, revision, users, teams, memberships,
		accessPolicies, ratePolicies, accessBindings, rateBindings, claims,
	)
	if err != nil {
		return accessDesiredData{}, err
	}
	credentials, credentialBarriers, err := loadCredentials(ctx, tx, namespace.ID, included)
	if err != nil {
		return accessDesiredData{}, err
	}
	delegated, delegatedBarriers, err := loadDelegatedCredentials(ctx, tx, namespace.ID, included)
	if err != nil {
		return accessDesiredData{}, err
	}
	credentials = append(credentials, delegated...)
	credentialBarriers = append(credentialBarriers, delegatedBarriers...)
	return accessDesiredData{
		keys: keys, credentials: credentials,
		barriers: append(append(append(append(userBarriers, teamBarriers...), membershipBarriers...), keyBarriers...), credentialBarriers...),
	}, nil
}

func loadDelegatedCredentials(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	included map[string]struct{},
) (_ []CredentialCandidate, _ []Barrier, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT
  d.id, d.public_id, d.api_key_id, d.token_hmac, d.pepper_version, d.status,
  d.not_before, d.expires_at, d.revoked_at, d.created_at,
  d.management_session_id, d.principal_id, d.delegation_epoch, d.user_id,
  d.team_id, d.audience,
  ms.principal_id, ms.status, ms.expires_at, mp.status,
  k.delegation_epoch, k.owner_user_id, k.owner_team_id, k.context_team_id,
  u.status, t.status, m.status,
  p.allow_team_key_delegation,
  EXISTS (SELECT 1 FROM management_principal_user_links l
          WHERE l.principal_id = d.principal_id AND l.namespace_id = d.namespace_id
            AND l.user_id = d.user_id),
  clock_timestamp()
FROM delegated_inference_sessions d
JOIN management_sessions ms ON ms.id = d.management_session_id
JOIN management_principals mp ON mp.id = d.principal_id
JOIN access_api_keys k ON k.namespace_id = d.namespace_id AND k.id = d.api_key_id
JOIN access_users u ON u.namespace_id = d.namespace_id AND u.id = d.user_id
JOIN self_service_policies p ON p.namespace_id = d.namespace_id
LEFT JOIN access_teams t ON t.namespace_id = d.namespace_id AND t.id = d.team_id
LEFT JOIN access_team_memberships m ON m.namespace_id = d.namespace_id
  AND m.team_id = d.team_id AND m.user_id = d.user_id
WHERE d.namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list delegated inference credentials: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]CredentialCandidate, 0)
	barriers := make([]Barrier, 0)
	for rows.Next() {
		var credential accesscontrol.CredentialVersion
		var status string
		var revoked sql.NullTime
		var managementSessionID, principalID, userID, audience string
		var teamID sql.NullString
		var delegationEpoch, currentDelegationEpoch int64
		var sessionPrincipalID, sessionStatus, principalStatus, userStatus string
		var sessionExpires time.Time
		var ownerUserID, ownerTeamID, contextTeamID sql.NullString
		var teamStatus, membershipStatus sql.NullString
		var linked, allowTeamKeyDelegation bool
		var databaseNow time.Time
		if err := rows.Scan(
			&credential.ID, &credential.KID, &credential.APIKeyID, &credential.SecretHMAC,
			&credential.PepperVersion, &status, &credential.NotBefore, &credential.ExpiresAt,
			&revoked, &credential.CreatedAt, &managementSessionID, &principalID,
			&delegationEpoch, &userID, &teamID, &audience,
			&sessionPrincipalID, &sessionStatus, &sessionExpires, &principalStatus,
			&currentDelegationEpoch, &ownerUserID, &ownerTeamID, &contextTeamID,
			&userStatus, &teamStatus, &membershipStatus, &allowTeamKeyDelegation, &linked,
			&databaseNow,
		); err != nil {
			return nil, nil, fmt.Errorf("scan delegated inference credential: %w", err)
		}
		identity := CredentialKindDelegation + ":" + credential.KID
		if revoked.Valid {
			value := revoked.Time.UTC()
			credential.RevokedAt = &value
		}
		credential.Status = accesscontrol.CredentialStatus(status)
		_, keyIncluded := included[string(credential.APIKeyID)]
		databaseNow = databaseNow.UTC()
		activeSession := sessionStatus == "active" && sessionExpires.After(databaseNow) && sessionPrincipalID == principalID
		activePrincipal := principalStatus == "active"
		validSubject := linked && userStatus == string(accesscontrol.UserStatusActive)
		validTeam := false
		switch {
		case ownerUserID.Valid && ownerUserID.String == userID:
			validTeam = (!contextTeamID.Valid && !teamID.Valid) ||
				(contextTeamID.Valid && teamID.Valid && contextTeamID.String == teamID.String &&
					teamStatus.String == string(accesscontrol.TeamStatusActive) &&
					membershipStatus.String == string(accesscontrol.MembershipStatusActive))
		case ownerTeamID.Valid && teamID.Valid && ownerTeamID.String == teamID.String:
			validTeam = allowTeamKeyDelegation && teamStatus.String == string(accesscontrol.TeamStatusActive) &&
				membershipStatus.String == string(accesscontrol.MembershipStatusActive)
		}
		active := status == "active" && !credential.NotBefore.After(databaseNow) &&
			credential.ExpiresAt != nil && credential.ExpiresAt.After(databaseNow) && keyIncluded &&
			delegationEpoch > 0 && delegationEpoch == currentDelegationEpoch && activeSession &&
			activePrincipal && validSubject && validTeam
		if !active {
			barriers = append(barriers, Barrier{Kind: "credential", ResourceID: identity, Reason: "delegation_inactive"})
			if !activeSession {
				barriers = append(barriers, Barrier{Kind: "management_session", ResourceID: managementSessionID, Reason: "management_session_inactive"})
			}
			if !activePrincipal {
				barriers = append(barriers, Barrier{Kind: "management_principal", ResourceID: principalID, Reason: "management_principal_inactive"})
			}
			continue
		}
		credential.Status = accesscontrol.CredentialStatusActive
		context := &accessprojection.DelegationContext{
			ManagementSessionID: managementSessionID, PrincipalID: principalID,
			DelegationEpoch: uint64(delegationEpoch), UserID: userID,
			Audience: audience,
		}
		if teamID.Valid {
			context.TeamID = teamID.String
		}
		result = append(result, CredentialCandidate{
			Kind: CredentialKindDelegation, Credential: credential, Delegation: context,
		})
	}
	return result, barriers, rows.Err()
}

func loadRoutingClaims(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (_ map[string]map[string]routingsnapshot.ClaimValue, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT subject_id, claim_name, value
FROM routing_subject_claims WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, fmt.Errorf("list routing claims: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make(map[string]map[string]routingsnapshot.ClaimValue)
	for rows.Next() {
		var subjectID, name string
		var raw []byte
		if err := rows.Scan(&subjectID, &name, &raw); err != nil {
			return nil, fmt.Errorf("scan routing claim: %w", err)
		}
		var value routingsnapshot.ClaimValue
		if err := strictJSON(raw, &value); err != nil {
			return nil, fmt.Errorf("decode routing claim %s/%s: %w", subjectID, name, err)
		}
		if result[subjectID] == nil {
			result[subjectID] = make(map[string]routingsnapshot.ClaimValue)
		}
		result[subjectID][name] = value
	}
	return result, rows.Err()
}

func loadKeyCandidates(
	ctx context.Context,
	tx *sql.Tx,
	namespace accesscontrol.Namespace,
	revision uint64,
	users map[string]accesscontrol.User,
	teams map[string]accesscontrol.Team,
	memberships map[string]accesscontrol.TeamMembership,
	accessPolicies map[accesscontrol.AccessPolicyID]accesscontrol.AccessPolicy,
	ratePolicies map[accesscontrol.RateLimitPolicyID]accesscontrol.RateLimitPolicy,
	accessBindings map[string][]accesscontrol.AccessPolicyBinding,
	rateBindings map[string][]accesscontrol.RateLimitBinding,
	claims map[string]map[string]routingsnapshot.ClaimValue,
) (_ []accessprojection.Candidate, _ []Barrier, _ map[string]struct{}, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, name, owner_user_id, owner_team_id, context_team_id,
status, expires_at, policy_epoch, delegation_epoch, revision, created_at, updated_at, deleted_at
FROM access_api_keys WHERE namespace_id = $1`, namespace.ID)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("list API keys: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]accessprojection.Candidate, 0)
	barriers := make([]Barrier, 0)
	included := make(map[string]struct{})
	for rows.Next() {
		key, relationships, err := scanKeyRelationships(rows, namespace.ID, users, teams, memberships)
		if err != nil {
			return nil, nil, nil, err
		}
		if namespace.Status != accesscontrol.NamespaceStatusActive || key.Status != accesscontrol.APIKeyStatusActive || key.DeletedAt != nil {
			barriers = append(barriers, Barrier{Kind: "api_key", ResourceID: string(key.ID), Reason: "key_inactive"})
			continue
		}
		if err := accesscontrol.ValidateAPIKeyRelationships(key, relationships); err != nil {
			barriers = append(barriers, Barrier{Kind: "api_key", ResourceID: string(key.ID), Reason: "owner_inactive"})
			continue
		}
		keySubject := string(key.ID)
		userSubject, teamSubject := keyInheritanceSubjects(key)
		candidate := accessprojection.Candidate{
			Revision: revision, Namespace: namespace, Key: key, Relationships: relationships,
			KeyAccessBindings: accessBindings[keySubject], UserAccessBindings: accessBindings[userSubject],
			TeamAccessBindings: accessBindings[teamSubject], AccessPolicies: accessPolicies,
			KeyRateBindings: rateBindings[keySubject], UserRateBindings: rateBindings[userSubject],
			TeamRateBindings: rateBindings[teamSubject], RatePolicies: ratePolicies,
			RoutingClaims: effectiveClaims(claims, keySubject, userSubject, teamSubject),
		}
		if selectedPolicyUnavailable(candidate) {
			barriers = append(barriers, Barrier{Kind: "api_key", ResourceID: string(key.ID), Reason: "policy_inactive"})
			continue
		}
		result = append(result, candidate)
		included[string(key.ID)] = struct{}{}
	}
	return result, barriers, included, rows.Err()
}

func scanKeyRelationships(
	scanner interface{ Scan(...any) error },
	namespaceID accesscontrol.NamespaceID,
	users map[string]accesscontrol.User,
	teams map[string]accesscontrol.Team,
	memberships map[string]accesscontrol.TeamMembership,
) (accesscontrol.APIKey, accesscontrol.APIKeyRelationships, error) {
	var key accesscontrol.APIKey
	var ownerUser, ownerTeam, contextTeam sql.NullString
	var expires, deleted sql.NullTime
	var policyEpoch, delegationEpoch, revision int64
	key.NamespaceID = namespaceID
	if err := scanner.Scan(&key.ID, &key.Name, &ownerUser, &ownerTeam, &contextTeam, &key.Status,
		&expires, &policyEpoch, &delegationEpoch, &revision, &key.CreatedAt, &key.UpdatedAt, &deleted); err != nil {
		return accesscontrol.APIKey{}, accesscontrol.APIKeyRelationships{}, fmt.Errorf("scan API key: %w", err)
	}
	if policyEpoch <= 0 || delegationEpoch <= 0 || revision <= 0 {
		return accesscontrol.APIKey{}, accesscontrol.APIKeyRelationships{}, fmt.Errorf("API key %s has invalid epoch or revision", key.ID)
	}
	key.PolicyEpoch, key.DelegationEpoch, key.Revision = uint64(policyEpoch), uint64(delegationEpoch), accesscontrol.Revision(revision)
	if expires.Valid {
		value := expires.Time.UTC()
		key.ExpiresAt = &value
	}
	if deleted.Valid {
		value := deleted.Time.UTC()
		key.DeletedAt = &value
	}
	relationships := accesscontrol.APIKeyRelationships{}
	if ownerUser.Valid {
		user, exists := users[ownerUser.String]
		if !exists {
			return key, relationships, fmt.Errorf("API key %s owner user is absent", key.ID)
		}
		key.Owner = user.SubjectRef()
		relationships.OwnerUser = &user
		if contextTeam.Valid {
			team, teamExists := teams[contextTeam.String]
			membership, membershipExists := memberships[membershipIdentity(team.ID, user.ID)]
			if !teamExists || !membershipExists {
				return key, relationships, fmt.Errorf("API key %s context relationship is absent", key.ID)
			}
			key.ContextTeamID = team.ID
			relationships.ContextTeam, relationships.ContextMembership = &team, &membership
		}
	} else if ownerTeam.Valid {
		team, exists := teams[ownerTeam.String]
		if !exists {
			return key, relationships, fmt.Errorf("API key %s owner team is absent", key.ID)
		}
		key.Owner = team.SubjectRef()
		relationships.OwnerTeam = &team
	} else {
		return key, relationships, fmt.Errorf("API key %s has no owner", key.ID)
	}
	return key, relationships, nil
}

func selectedPolicyUnavailable(candidate accessprojection.Candidate) bool {
	effectiveAccess, err := accesscontrol.ResolveAccessBindings(
		candidate.KeyAccessBindings, candidate.UserAccessBindings, candidate.TeamAccessBindings,
	)
	if err != nil {
		return true
	}
	for _, binding := range effectiveAccess.Bindings {
		policy, exists := candidate.AccessPolicies[binding.PolicyID]
		if !exists || policy.Status != accesscontrol.PolicyStatusActive {
			return true
		}
	}
	effectiveRate, err := accesscontrol.ResolveRateBindings(
		candidate.KeyRateBindings, candidate.UserRateBindings, candidate.TeamRateBindings,
	)
	if err != nil {
		return true
	}
	bindings := effectiveRate.HardCaps
	if effectiveRate.Allocation != nil {
		bindings = append(bindings, *effectiveRate.Allocation)
	}
	for _, resolved := range bindings {
		policy, exists := candidate.RatePolicies[resolved.Binding.PolicyID]
		if !exists || policy.Status != accesscontrol.PolicyStatusActive {
			return true
		}
	}
	return false
}

func keyInheritanceSubjects(key accesscontrol.APIKey) (string, string) {
	if key.Owner.Kind == accesscontrol.SubjectKindTeam {
		return "", string(key.Owner.ID)
	}
	return string(key.Owner.ID), string(key.ContextTeamID)
}

func effectiveClaims(all map[string]map[string]routingsnapshot.ClaimValue, keyID, userID, teamID string) map[string]routingsnapshot.ClaimValue {
	result := make(map[string]routingsnapshot.ClaimValue)
	for _, subjectID := range []string{teamID, userID, keyID} {
		for name, value := range all[subjectID] {
			result[name] = value
		}
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func loadCredentials(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	included map[string]struct{},
) (_ []CredentialCandidate, _ []Barrier, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT id, api_key_id, kid, secret_hmac, pepper_version, status,
not_before, expires_at, revoked_at, created_at FROM access_api_key_credentials WHERE namespace_id = $1`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list API-key credentials: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]CredentialCandidate, 0)
	barriers := make([]Barrier, 0)
	for rows.Next() {
		var credential accesscontrol.CredentialVersion
		var expires, revoked sql.NullTime
		if err := rows.Scan(&credential.ID, &credential.APIKeyID, &credential.KID, &credential.SecretHMAC,
			&credential.PepperVersion, &credential.Status, &credential.NotBefore, &expires, &revoked, &credential.CreatedAt); err != nil {
			return nil, nil, fmt.Errorf("scan API-key credential: %w", err)
		}
		if expires.Valid {
			value := expires.Time.UTC()
			credential.ExpiresAt = &value
		}
		if revoked.Valid {
			value := revoked.Time.UTC()
			credential.RevokedAt = &value
		}
		_, keyIncluded := included[string(credential.APIKeyID)]
		if !keyIncluded || (credential.Status != accesscontrol.CredentialStatusActive && credential.Status != accesscontrol.CredentialStatusRetiring) {
			barriers = append(barriers, Barrier{
				Kind: "credential", ResourceID: CredentialKindAPIKey + ":" + credential.KID, Reason: "credential_inactive",
			})
			continue
		}
		result = append(result, CredentialCandidate{Kind: CredentialKindAPIKey, Credential: credential})
	}
	return result, barriers, rows.Err()
}

func loadRoutingBundle(ctx context.Context, tx *sql.Tx, namespace accesscontrol.Namespace, revision uint64) (routingsnapshot.Bundle, error) {
	if revision > math.MaxInt64 {
		return routingsnapshot.Bundle{}, fmt.Errorf("routing revision exceeds PostgreSQL BIGINT")
	}
	return routingpostgres.LoadPublishedBundle(
		ctx, tx, string(namespace.ID), namespace.BillingCurrency, int64(revision),
	)
}

func strictJSON(payload []byte, target any) error {
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("JSON has trailing values")
	}
	return nil
}

func membershipIdentity(teamID accesscontrol.TeamID, userID accesscontrol.UserID) string {
	return string(teamID) + ":" + string(userID)
}
