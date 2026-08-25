package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/delegationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

type delegationRepositoryAdapter struct{ store *Store }

func NewDelegationManagementRepository(store *Store) (delegationmanagement.Repository, error) {
	if store == nil || store.db == nil {
		return nil, delegationmanagement.ErrUnavailable
	}
	return &delegationRepositoryAdapter{store: store}, nil
}

func (adapter *delegationRepositoryAdapter) Ready(ctx context.Context, codec *managementcommand.Codec) error {
	if adapter == nil || adapter.store == nil || codec == nil {
		return delegationmanagement.ErrUnavailable
	}
	if err := adapter.store.db.PingContext(ctx); err != nil {
		return err
	}
	return commandpostgres.ValidateReferencedHMACVersions(ctx, adapter.store.db, codec)
}

func (adapter *delegationRepositoryAdapter) ResolveSelf(
	ctx context.Context, namespaceID, principalID, managementSessionID string, _ bool,
) (delegationmanagement.SelfContext, error) {
	return scanDelegationSelf(adapter.store.db.QueryRowContext(ctx, `SELECT
  n.quota_partition_id, l.user_id, s.expires_at,
  p.max_delegated_sessions, p.delegated_session_ttl_seconds,
  p.allow_team_key_delegation, p.revision
FROM access_namespaces n
JOIN management_principal_user_links l ON l.namespace_id = n.id AND l.principal_id = $2
JOIN access_users u ON u.namespace_id = l.namespace_id AND u.id = l.user_id
JOIN management_sessions s ON s.id = $3 AND s.principal_id = $2
JOIN management_principals mp ON mp.id = $2
JOIN self_service_policies p ON p.namespace_id = n.id
WHERE n.id = $1 AND n.status = 'active' AND u.status = 'active'
  AND s.status = 'active' AND s.expires_at > clock_timestamp()
  AND mp.status = 'active'`, namespaceID, principalID, managementSessionID), namespaceID, principalID, managementSessionID)
}

func scanDelegationSelf(
	row *sql.Row, namespaceID, principalID, managementSessionID string,
) (delegationmanagement.SelfContext, error) {
	var result delegationmanagement.SelfContext
	var ttlSeconds, revision int64
	result.NamespaceID, result.PrincipalID, result.ManagementSessionID = namespaceID, principalID, managementSessionID
	err := row.Scan(&result.QuotaPartition, &result.UserID, &result.ManagementSessionExpires,
		&result.Policy.MaxDelegatedSessions, &ttlSeconds, &result.Policy.AllowTeamKeyDelegation, &revision)
	if errors.Is(err, sql.ErrNoRows) {
		return delegationmanagement.SelfContext{}, delegationmanagement.ErrNotEligible
	}
	if err != nil {
		return delegationmanagement.SelfContext{}, fmt.Errorf("resolve delegated inference self context: %w", err)
	}
	if ttlSeconds <= 0 || revision <= 0 {
		return delegationmanagement.SelfContext{}, delegationmanagement.ErrUnavailable
	}
	result.Policy.DelegatedSessionTTL = time.Duration(ttlSeconds) * time.Second
	result.Policy.Revision = uint64(revision)
	return result, nil
}

const eligibleKeySelect = `SELECT k.id, k.name,
  CASE WHEN k.owner_user_id IS NOT NULL THEN 'user' ELSE 'team' END,
  COALESCE(k.owner_user_id::text, k.owner_team_id::text),
  COALESCE(k.context_team_id::text, ''), k.expires_at, k.delegation_epoch,
  COALESCE(k.context_team_id::text, k.owner_team_id::text, ''), k.created_at
FROM access_api_keys k
JOIN management_principal_user_links l ON l.namespace_id = k.namespace_id AND l.principal_id = $2
JOIN access_users u ON u.namespace_id = l.namespace_id AND u.id = l.user_id
JOIN self_service_policies p ON p.namespace_id = k.namespace_id
LEFT JOIN access_teams t ON t.namespace_id = k.namespace_id
  AND t.id = COALESCE(k.context_team_id, k.owner_team_id)
LEFT JOIN access_team_memberships m ON m.namespace_id = k.namespace_id
  AND m.team_id = COALESCE(k.context_team_id, k.owner_team_id) AND m.user_id = l.user_id
WHERE k.namespace_id = $1 AND k.status = 'active' AND k.deleted_at IS NULL
  AND (k.expires_at IS NULL OR k.expires_at > clock_timestamp())
  AND u.status = 'active'
  AND ((k.owner_user_id = l.user_id
        AND (k.context_team_id IS NULL OR (t.status = 'active' AND m.status = 'active')))
       OR (k.owner_team_id IS NOT NULL AND p.allow_team_key_delegation
           AND t.status = 'active' AND m.status = 'active'))`

func (adapter *delegationRepositoryAdapter) ListEligibleKeys(
	ctx context.Context, query delegationmanagement.EligibleKeyQuery,
) (delegationmanagement.Page[delegationmanagement.EligibleKey], error) {
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	rows, err := adapter.store.db.QueryContext(ctx, eligibleKeySelect+`
  AND ($3::timestamptz IS NULL OR k.created_at < $3 OR (k.created_at = $3 AND k.id > $4::uuid))
ORDER BY k.created_at DESC, k.id ASC LIMIT $5`, query.NamespaceID, query.PrincipalID, afterTime, afterID, query.Limit+1)
	if err != nil {
		return delegationmanagement.Page[delegationmanagement.EligibleKey]{}, fmt.Errorf("list eligible inference keys: %w", err)
	}
	defer rows.Close()
	items := make([]delegationmanagement.EligibleKey, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanEligibleKey(rows)
		if err != nil {
			return delegationmanagement.Page[delegationmanagement.EligibleKey]{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return delegationmanagement.Page[delegationmanagement.EligibleKey]{}, err
	}
	hasMore := len(items) > query.Limit
	if hasMore {
		items = items[:query.Limit]
	}
	return delegationmanagement.Page[delegationmanagement.EligibleKey]{Items: items, HasMore: hasMore}, nil
}

func (adapter *delegationRepositoryAdapter) GetEligibleKey(
	ctx context.Context, namespaceID, principalID, _ string, keyID string,
) (delegationmanagement.EligibleKey, error) {
	item, err := scanEligibleKey(adapter.store.db.QueryRowContext(ctx, eligibleKeySelect+` AND k.id = $3`, namespaceID, principalID, keyID))
	if errors.Is(err, sql.ErrNoRows) {
		return delegationmanagement.EligibleKey{}, delegationmanagement.ErrNotEligible
	}
	return item, err
}

func scanEligibleKey(scanner rowScanner) (delegationmanagement.EligibleKey, error) {
	var item delegationmanagement.EligibleKey
	var ownerKind string
	var expires sql.NullTime
	var epoch int64
	if err := scanner.Scan(&item.KeyID, &item.Name, &ownerKind, &item.OwnerID, &item.ContextTeamID,
		&expires, &epoch, &item.TeamID, &item.CreatedAt); err != nil {
		return item, err
	}
	if epoch <= 0 {
		return item, delegationmanagement.ErrUnavailable
	}
	item.OwnerKind, item.DelegationEpoch = accesscontrol.SubjectKind(ownerKind), uint64(epoch)
	if expires.Valid {
		value := expires.Time.UTC()
		item.ExpiresAt = &value
	}
	return item, nil
}

func (adapter *delegationRepositoryAdapter) GetKey(ctx context.Context, namespaceID, keyID string) (accesscontrol.APIKey, error) {
	key, err := adapter.store.GetAPIKey(ctx, accesscontrol.NamespaceID(namespaceID), accesscontrol.APIKeyID(keyID))
	if errors.Is(err, ErrNotFound) {
		return accesscontrol.APIKey{}, delegationmanagement.ErrNotFound
	}
	return key, err
}

const delegatedSessionColumns = `d.id, d.public_id, d.namespace_id, n.quota_partition_id,
d.management_session_id, d.principal_id, d.api_key_id, d.delegation_epoch,
d.user_id, d.team_id, d.token_hmac, d.pepper_version, d.audience,
CASE WHEN d.status <> 'revoked' AND d.expires_at <= clock_timestamp() THEN 'expired' ELSE d.status END,
d.not_before, d.expires_at, d.revoked_at, d.revision, d.created_at`

func (adapter *delegationRepositoryAdapter) ListSessions(
	ctx context.Context, query delegationmanagement.SessionQuery,
) (delegationmanagement.Page[delegationmanagement.Session], error) {
	var afterTime any
	afterID := "00000000-0000-0000-0000-000000000000"
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	rows, err := adapter.store.db.QueryContext(ctx, `SELECT `+delegatedSessionColumns+`
FROM delegated_inference_sessions d JOIN access_namespaces n ON n.id = d.namespace_id
WHERE d.namespace_id = $1
  AND (NULLIF($2, '') IS NULL OR d.principal_id = NULLIF($2, '')::uuid)
  AND (NULLIF($3, '') IS NULL OR d.api_key_id = NULLIF($3, '')::uuid)
  AND ($4::timestamptz IS NULL OR d.created_at < $4 OR (d.created_at = $4 AND d.id > $5::uuid))
ORDER BY d.created_at DESC, d.id ASC LIMIT $6`, query.NamespaceID, query.PrincipalID, query.APIKeyID,
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return delegationmanagement.Page[delegationmanagement.Session]{}, fmt.Errorf("list delegated inference sessions: %w", err)
	}
	defer rows.Close()
	items := make([]delegationmanagement.Session, 0, query.Limit+1)
	for rows.Next() {
		item, err := scanDelegatedSession(rows)
		if err != nil {
			return delegationmanagement.Page[delegationmanagement.Session]{}, err
		}
		item.TokenHMAC = nil
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return delegationmanagement.Page[delegationmanagement.Session]{}, err
	}
	hasMore := len(items) > query.Limit
	if hasMore {
		items = items[:query.Limit]
	}
	return delegationmanagement.Page[delegationmanagement.Session]{Items: items, HasMore: hasMore}, nil
}

func (adapter *delegationRepositoryAdapter) GetSession(ctx context.Context, namespaceID, sessionID string) (delegationmanagement.Session, error) {
	item, err := scanDelegatedSession(adapter.store.db.QueryRowContext(ctx, `SELECT `+delegatedSessionColumns+`
FROM delegated_inference_sessions d JOIN access_namespaces n ON n.id = d.namespace_id
WHERE d.namespace_id = $1 AND d.id = $2`, namespaceID, sessionID))
	if errors.Is(err, sql.ErrNoRows) {
		return delegationmanagement.Session{}, delegationmanagement.ErrNotFound
	}
	if err != nil {
		return delegationmanagement.Session{}, fmt.Errorf("get delegated inference session: %w", err)
	}
	item.TokenHMAC = nil
	return item, nil
}

func scanDelegatedSession(scanner rowScanner) (delegationmanagement.Session, error) {
	var item delegationmanagement.Session
	var teamID sql.NullString
	var revoked sql.NullTime
	var epoch, revision int64
	if err := scanner.Scan(&item.ID, &item.PublicID, &item.NamespaceID, &item.QuotaPartition,
		&item.ManagementSessionID, &item.PrincipalID, &item.APIKeyID, &epoch,
		&item.UserID, &teamID, &item.TokenHMAC, &item.PepperVersion, &item.Audience,
		&item.Status, &item.NotBefore, &item.ExpiresAt, &revoked, &revision, &item.CreatedAt); err != nil {
		return item, err
	}
	if epoch <= 0 || revision <= 0 {
		return item, delegationmanagement.ErrUnavailable
	}
	item.DelegationEpoch, item.Revision = uint64(epoch), uint64(revision)
	if teamID.Valid {
		item.TeamID = teamID.String
	}
	if revoked.Valid {
		value := revoked.Time.UTC()
		item.RevokedAt = &value
	}
	return item, nil
}

func (adapter *delegationRepositoryAdapter) ReplaySecret(
	ctx context.Context, command managementcommand.Command,
) (delegationmanagement.StoredSecret, bool, error) {
	stored, found, err := commandpostgres.Lookup(ctx, adapter.store.db, command)
	if err != nil || !found {
		return delegationmanagement.StoredSecret{}, found, err
	}
	result, err := adapter.storedSecret(ctx, stored)
	return result, true, err
}

func (adapter *delegationRepositoryAdapter) storedSecret(ctx context.Context, stored managementcommand.StoredResult) (delegationmanagement.StoredSecret, error) {
	if stored.Resource == nil || stored.Secret == nil || stored.Resource.ResourceType != "delegated_inference_session" {
		return delegationmanagement.StoredSecret{}, delegationmanagement.ErrUnavailable
	}
	desired, err := latestAggregateDesiredRevision(ctx, adapter.store.db, stored.Resource.ResourceID)
	if err != nil {
		return delegationmanagement.StoredSecret{}, err
	}
	return delegationmanagement.StoredSecret{Result: *stored.Resource, Secret: *stored.Secret, DesiredRevision: desired}, nil
}

func (adapter *delegationRepositoryAdapter) Create(
	ctx context.Context, mutation delegationmanagement.CreateMutation,
) (delegationmanagement.MutationResult, error) {
	meta, err := delegationMutationMeta(mutation.Actor, "delegated_inference_session.create", "Create delegated inference session.")
	if err != nil {
		return delegationmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (delegationmanagement.MutationResult, error) {
		stored, replayed, createErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if createErr != nil {
			return delegationmanagement.MutationResult{}, createErr
		}
		if replayed {
			secret, err := storedDelegationSecretTx(ctx, tx, stored)
			if err != nil {
				return delegationmanagement.MutationResult{}, err
			}
			session, err := getDelegatedSessionTx(ctx, tx, mutation.Session.NamespaceID, secret.Result.ResourceID)
			if err != nil {
				return delegationmanagement.MutationResult{}, err
			}
			return delegationmanagement.MutationResult{
				Session: session, DesiredRevision: secret.DesiredRevision,
				Replayed: true, Stored: &secret,
			}, nil
		}
		self, createErr := lockDelegationSelf(ctx, tx, mutation.Session)
		if createErr != nil {
			return delegationmanagement.MutationResult{}, createErr
		}
		key, createErr := lockEligibleKey(ctx, tx, mutation.Session.NamespaceID, mutation.Session.PrincipalID, mutation.Session.APIKeyID)
		if createErr != nil {
			return delegationmanagement.MutationResult{}, createErr
		}
		if self.UserID != mutation.Session.UserID || self.QuotaPartition != mutation.Session.QuotaPartition ||
			key.DelegationEpoch != mutation.Session.DelegationEpoch || key.TeamID != mutation.Session.TeamID ||
			mutation.Session.Audience == "" || mutation.Session.ExpiresAt.After(self.ManagementSessionExpires) ||
			mutation.Session.ExpiresAt.After(mutation.Session.NotBefore.Add(self.Policy.DelegatedSessionTTL)) ||
			(key.ExpiresAt != nil && mutation.Session.ExpiresAt.After(*key.ExpiresAt)) {
			return delegationmanagement.MutationResult{}, delegationmanagement.ErrNotEligible
		}
		if mutation.Session.TeamID != "" {
			var teamStatus, membershipStatus string
			if err := tx.QueryRowContext(ctx, `SELECT t.status, m.status
FROM access_teams t JOIN access_team_memberships m
  ON m.namespace_id = t.namespace_id AND m.team_id = t.id AND m.user_id = $3
WHERE t.namespace_id = $1 AND t.id = $2
FOR KEY SHARE OF t, m`, mutation.Session.NamespaceID, mutation.Session.TeamID, mutation.Session.UserID).Scan(
				&teamStatus, &membershipStatus); err != nil || teamStatus != "active" || membershipStatus != "active" {
				return delegationmanagement.MutationResult{}, delegationmanagement.ErrNotEligible
			}
		}
		var activeCount int
		if err := tx.QueryRowContext(ctx, `SELECT count(*) FROM delegated_inference_sessions
WHERE namespace_id = $1 AND user_id = $2 AND status = 'active' AND expires_at > clock_timestamp()`,
			mutation.Session.NamespaceID, mutation.Session.UserID).Scan(&activeCount); err != nil {
			return delegationmanagement.MutationResult{}, err
		}
		if self.Policy.MaxDelegatedSessions <= 0 || activeCount >= self.Policy.MaxDelegatedSessions {
			return delegationmanagement.MutationResult{}, delegationmanagement.ErrSessionLimit
		}
		_, createErr = tx.ExecContext(ctx, `INSERT INTO delegated_inference_sessions
  (id, public_id, namespace_id, management_session_id, principal_id, api_key_id,
   delegation_epoch, user_id, team_id, token_hmac, pepper_version, audience,
   status, not_before, expires_at, revision, created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,'active',$13,$14,1,$15)`,
			mutation.Session.ID, mutation.Session.PublicID, mutation.Session.NamespaceID,
			mutation.Session.ManagementSessionID, mutation.Session.PrincipalID, mutation.Session.APIKeyID,
			mutation.Session.DelegationEpoch, mutation.Session.UserID, nullableString(mutation.Session.TeamID),
			mutation.Session.TokenHMAC, mutation.Session.PepperVersion, mutation.Session.Audience,
			mutation.Session.NotBefore, mutation.Session.ExpiresAt, mutation.Session.CreatedAt)
		if createErr != nil {
			return delegationmanagement.MutationResult{}, mapDelegationWriteError(createErr)
		}
		receipt, createErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(mutation.Session.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: mutation.Session.ID,
			AggregateRevision: 1, Operation: outboxCreated,
			References: map[string]string{"apiKeyId": mutation.Session.APIKeyID},
		}, meta)
		if createErr != nil {
			return delegationmanagement.MutationResult{}, createErr
		}
		if err := commandpostgres.CompleteSecretResource(ctx, tx, mutation.Command,
			managementcommand.ResourceResult{
				ResourceType: "delegated_inference_session", ResourceID: mutation.Session.ID,
				ResourceRevision: 1, ResponseStatus: 201,
			},
			managementcommand.SecretResponse{
				Ciphertext: mutation.Response.Ciphertext, Nonce: mutation.Response.Nonce,
				KEKVersion: mutation.Response.KeyVersion, ExpiresAt: mutation.ResponseExpiresAt,
			}); err != nil {
			return delegationmanagement.MutationResult{}, err
		}
		resultSession := mutation.Session
		resultSession.TokenHMAC = nil
		return delegationmanagement.MutationResult{Session: resultSession, DesiredRevision: uint64(receipt.DesiredRevision)}, nil
	})
}

func (adapter *delegationRepositoryAdapter) Revoke(
	ctx context.Context, request delegationmanagement.RevokeRequest,
) (delegationmanagement.MutationResult, error) {
	meta, err := delegationMutationMeta(request.Actor, "delegated_inference_session.revoke", "Revoke delegated inference session.")
	if err != nil {
		return delegationmanagement.MutationResult{}, err
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (delegationmanagement.MutationResult, error) {
		query := `SELECT ` + delegatedSessionColumns + ` FROM delegated_inference_sessions d
JOIN access_namespaces n ON n.id = d.namespace_id
WHERE d.namespace_id = $1 AND d.id = $2
  AND (NULLIF($3, '') IS NULL OR d.principal_id = NULLIF($3, '')::uuid)
  AND (NULLIF($4, '') IS NULL OR d.api_key_id = NULLIF($4, '')::uuid)
FOR UPDATE OF d`
		session, revokeErr := scanDelegatedSession(tx.QueryRowContext(ctx, query, request.NamespaceID, request.SessionID,
			request.PrincipalID, request.APIKeyID))
		if errors.Is(revokeErr, sql.ErrNoRows) {
			return delegationmanagement.MutationResult{}, delegationmanagement.ErrNotFound
		}
		if revokeErr != nil {
			return delegationmanagement.MutationResult{}, revokeErr
		}
		if session.Status == delegationmanagement.SessionRevoked {
			desired, err := latestNamespaceDesiredRevision(ctx, tx, request.NamespaceID)
			return delegationmanagement.MutationResult{Session: session, DesiredRevision: desired, Replayed: true}, err
		}
		var revision int64
		var revokedAt time.Time
		revokeErr = tx.QueryRowContext(ctx, `UPDATE delegated_inference_sessions
SET status = 'revoked', revoked_at = clock_timestamp(), revision = revision + 1
WHERE namespace_id = $1 AND id = $2 RETURNING revision, revoked_at`, request.NamespaceID, request.SessionID).Scan(&revision, &revokedAt)
		if revokeErr != nil {
			return delegationmanagement.MutationResult{}, revokeErr
		}
		revisionValue, revisionErr := positiveUint64(revision, "delegated session revision")
		if revisionErr != nil {
			return delegationmanagement.MutationResult{}, revisionErr
		}
		receipt, revokeErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(request.NamespaceID), outboxMutation{
			AggregateType: "delegated_inference_session", AggregateID: session.ID,
			AggregateRevision: accesscontrol.Revision(revisionValue), Operation: outboxDeleted,
			References: map[string]string{"apiKeyId": session.APIKeyID},
		}, meta)
		if revokeErr != nil {
			return delegationmanagement.MutationResult{}, revokeErr
		}
		revokedAt = revokedAt.UTC()
		session.Status, session.RevokedAt, session.Revision = delegationmanagement.SessionRevoked, &revokedAt, revisionValue
		session.TokenHMAC = nil
		return delegationmanagement.MutationResult{Session: session, DesiredRevision: uint64(receipt.DesiredRevision)}, nil
	})
}

func (adapter *delegationRepositoryAdapter) RevokeAll(
	ctx context.Context, mutation delegationmanagement.RevokeAllMutation,
) (delegationmanagement.RevokeAllResult, error) {
	meta, err := delegationMutationMeta(mutation.Actor, "delegated_inference_session.revoke_all", "Revoke all delegated inference sessions for API key.")
	if err != nil {
		return delegationmanagement.RevokeAllResult{}, err
	}
	return inTransaction(ctx, adapter.store, func(tx *sql.Tx) (delegationmanagement.RevokeAllResult, error) {
		stored, replayed, revokeAllErr := commandpostgres.Lock(ctx, tx, mutation.Command)
		if revokeAllErr != nil {
			return delegationmanagement.RevokeAllResult{}, revokeAllErr
		}
		if replayed {
			if stored.Resource == nil || stored.Secret != nil || stored.Resource.ResourceType != "api_key" || stored.Resource.ResourceID != mutation.KeyID {
				return delegationmanagement.RevokeAllResult{}, delegationmanagement.ErrUnavailable
			}
			return loadRevokeAllResult(ctx, tx, mutation.NamespaceID, mutation.KeyID, true)
		}
		var epoch, revision int64
		revokeAllErr = tx.QueryRowContext(ctx, `UPDATE access_api_keys
SET delegation_epoch = delegation_epoch + 1, revision = revision + 1, updated_at = clock_timestamp()
WHERE namespace_id = $1 AND id = $2 AND deleted_at IS NULL
RETURNING delegation_epoch, revision`, mutation.NamespaceID, mutation.KeyID).Scan(&epoch, &revision)
		if errors.Is(revokeAllErr, sql.ErrNoRows) {
			return delegationmanagement.RevokeAllResult{}, delegationmanagement.ErrNotFound
		}
		if revokeAllErr != nil {
			return delegationmanagement.RevokeAllResult{}, revokeAllErr
		}
		epochValue, conversionErr := positiveUint64(epoch, "delegation epoch")
		if conversionErr != nil {
			return delegationmanagement.RevokeAllResult{}, conversionErr
		}
		revisionValue, conversionErr := positiveUint64(revision, "API-key revision")
		if conversionErr != nil {
			return delegationmanagement.RevokeAllResult{}, conversionErr
		}
		if _, err := tx.ExecContext(ctx, `UPDATE delegated_inference_sessions
SET status = 'revoked', revoked_at = clock_timestamp(), revision = revision + 1
WHERE namespace_id = $1 AND api_key_id = $2 AND status = 'active'`, mutation.NamespaceID, mutation.KeyID); err != nil {
			return delegationmanagement.RevokeAllResult{}, err
		}
		receipt, revokeAllErr := appendMutationRecords(ctx, tx, accesscontrol.NamespaceID(mutation.NamespaceID), outboxMutation{
			AggregateType: "api_key", AggregateID: mutation.KeyID,
			AggregateRevision: accesscontrol.Revision(revisionValue), Operation: outboxUpdated,
			References: map[string]string{"delegationEpoch": fmt.Sprint(epoch)},
		}, meta)
		if revokeAllErr != nil {
			return delegationmanagement.RevokeAllResult{}, revokeAllErr
		}
		if err := commandpostgres.CompleteResource(ctx, tx, mutation.Command, managementcommand.ResourceResult{
			ResourceType: "api_key", ResourceID: mutation.KeyID, ResourceRevision: revisionValue, ResponseStatus: 204,
		}); err != nil {
			return delegationmanagement.RevokeAllResult{}, err
		}
		var partition string
		if err := tx.QueryRowContext(ctx, `SELECT quota_partition_id FROM access_namespaces WHERE id = $1`, mutation.NamespaceID).Scan(&partition); err != nil {
			return delegationmanagement.RevokeAllResult{}, err
		}
		return delegationmanagement.RevokeAllResult{
			KeyID: mutation.KeyID, DelegationEpoch: epochValue,
			DesiredRevision: uint64(receipt.DesiredRevision), QuotaPartition: partition,
		}, nil
	})
}

func lockDelegationSelf(ctx context.Context, tx *sql.Tx, session delegationmanagement.Session) (delegationmanagement.SelfContext, error) {
	return scanDelegationSelf(tx.QueryRowContext(ctx, `SELECT
  n.quota_partition_id, l.user_id, s.expires_at,
  p.max_delegated_sessions, p.delegated_session_ttl_seconds,
  p.allow_team_key_delegation, p.revision
FROM access_namespaces n
JOIN management_principal_user_links l ON l.namespace_id = n.id AND l.principal_id = $2
JOIN access_users u ON u.namespace_id = l.namespace_id AND u.id = l.user_id
JOIN management_sessions s ON s.id = $3 AND s.principal_id = $2
JOIN management_principals mp ON mp.id = $2
JOIN self_service_policies p ON p.namespace_id = n.id
WHERE n.id = $1 AND n.status = 'active' AND u.status = 'active'
  AND s.status = 'active' AND s.expires_at > clock_timestamp() AND mp.status = 'active'
FOR UPDATE OF l, u, s, mp, p`, session.NamespaceID, session.PrincipalID, session.ManagementSessionID),
		session.NamespaceID, session.PrincipalID, session.ManagementSessionID)
}

func lockEligibleKey(ctx context.Context, tx *sql.Tx, namespaceID, principalID, keyID string) (delegationmanagement.EligibleKey, error) {
	key, err := scanEligibleKey(tx.QueryRowContext(ctx, eligibleKeySelect+` AND k.id = $3 FOR UPDATE OF k, l, u, p`, namespaceID, principalID, keyID))
	if errors.Is(err, sql.ErrNoRows) {
		return delegationmanagement.EligibleKey{}, delegationmanagement.ErrNotEligible
	}
	return key, err
}

func getDelegatedSessionTx(ctx context.Context, tx *sql.Tx, namespaceID, sessionID string) (delegationmanagement.Session, error) {
	return scanDelegatedSession(tx.QueryRowContext(ctx, `SELECT `+delegatedSessionColumns+`
FROM delegated_inference_sessions d JOIN access_namespaces n ON n.id = d.namespace_id
WHERE d.namespace_id = $1 AND d.id = $2`, namespaceID, sessionID))
}

func storedDelegationSecretTx(ctx context.Context, tx *sql.Tx, stored managementcommand.StoredResult) (delegationmanagement.StoredSecret, error) {
	if stored.Resource == nil || stored.Secret == nil || stored.Resource.ResourceType != "delegated_inference_session" {
		return delegationmanagement.StoredSecret{}, delegationmanagement.ErrUnavailable
	}
	desired, err := latestAggregateDesiredRevision(ctx, tx, stored.Resource.ResourceID)
	if err != nil {
		return delegationmanagement.StoredSecret{}, err
	}
	return delegationmanagement.StoredSecret{Result: *stored.Resource, Secret: *stored.Secret, DesiredRevision: desired}, nil
}

type queryRower interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
}

func latestAggregateDesiredRevision(ctx context.Context, source queryRower, aggregateID string) (uint64, error) {
	var desired int64
	err := source.QueryRowContext(ctx, `SELECT desired_revision FROM policy_outbox
WHERE aggregate_id = $1 ORDER BY desired_revision DESC LIMIT 1`, aggregateID).Scan(&desired)
	if err != nil || desired <= 0 {
		return 0, delegationmanagement.ErrUnavailable
	}
	desiredValue, err := positiveUint64(desired, "desired policy revision")
	if err != nil {
		return 0, delegationmanagement.ErrUnavailable
	}
	return desiredValue, nil
}

func latestNamespaceDesiredRevision(ctx context.Context, source queryRower, namespaceID string) (uint64, error) {
	var desired int64
	err := source.QueryRowContext(ctx, `SELECT COALESCE(MAX(revision), 0)
FROM policy_revisions WHERE namespace_id = $1`, namespaceID).Scan(&desired)
	if err != nil || desired <= 0 {
		return 0, delegationmanagement.ErrUnavailable
	}
	return uint64(desired), nil
}

func loadRevokeAllResult(ctx context.Context, tx *sql.Tx, namespaceID, keyID string, replayed bool) (delegationmanagement.RevokeAllResult, error) {
	var epoch int64
	var partition string
	if err := tx.QueryRowContext(ctx, `SELECT k.delegation_epoch, n.quota_partition_id
FROM access_api_keys k JOIN access_namespaces n ON n.id = k.namespace_id
WHERE k.namespace_id = $1 AND k.id = $2`, namespaceID, keyID).Scan(&epoch, &partition); err != nil {
		return delegationmanagement.RevokeAllResult{}, err
	}
	epochValue, err := positiveUint64(epoch, "delegation epoch")
	if err != nil {
		return delegationmanagement.RevokeAllResult{}, err
	}
	desired, err := latestAggregateDesiredRevision(ctx, tx, keyID)
	return delegationmanagement.RevokeAllResult{
		KeyID: keyID, DelegationEpoch: epochValue,
		DesiredRevision: desired, QuotaPartition: partition, Replayed: replayed,
	}, err
}

func delegationMutationMeta(actor delegationmanagement.Actor, action, reason string) (MutationMeta, error) {
	principal := accesscontrol.ManagementPrincipalID(actor.PrincipalID)
	chain := make([]accesscontrol.ManagementPrincipalID, len(actor.ActorChain))
	for index, value := range actor.ActorChain {
		chain[index] = accesscontrol.ManagementPrincipalID(value)
	}
	meta := MutationMeta{
		ActorPrincipalID: &principal, ActorChain: chain, RequestID: actor.RequestID,
		SourceIP: actor.SourceIP, Action: action, Reason: reason, Details: AuditDetails{},
	}
	if err := validateMutationMeta(meta); err != nil {
		return MutationMeta{}, delegationmanagement.ErrInvalidRequest
	}
	return meta, nil
}

func mapDelegationWriteError(err error) error {
	var databaseError *pq.Error
	if errors.As(err, &databaseError) && databaseError.Code == "23505" {
		return delegationmanagement.ErrNotEligible
	}
	return fmt.Errorf("persist delegated inference session: %w", err)
}

var _ delegationmanagement.Repository = (*delegationRepositoryAdapter)(nil)
