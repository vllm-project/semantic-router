// Package postgres persists Router-native Management principals, roles,
// bindings, User links, and the cluster session policy. It owns no Dashboard
// account state and performs no positive authorization caching.
package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
	authpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

const (
	maximumPageSize = 200
	seedVersion     = 1
)

type Store struct {
	database      *sql.DB
	commands      *managementcommand.Codec
	sessionPolicy *authpostgres.Store
}

func New(database *sql.DB, commands *managementcommand.Codec) (*Store, error) {
	if database == nil || commands == nil {
		return nil, errors.New("management identity PostgreSQL database and command codec are required")
	}
	policy, err := authpostgres.New(database)
	if err != nil {
		return nil, err
	}
	return &Store{database: database, commands: commands, sessionPolicy: policy}, nil
}

func (store *Store) Ready(ctx context.Context) error {
	if store == nil || store.database == nil || store.commands == nil || store.sessionPolicy == nil {
		return errors.New("management identity PostgreSQL store is unavailable")
	}
	if err := store.database.PingContext(ctx); err != nil {
		return fmt.Errorf("ping Management identity PostgreSQL: %w", err)
	}
	if err := commandpostgres.ValidateReferencedHMACVersions(ctx, store.database, store.commands); err != nil {
		return err
	}
	policy, readyErr := store.sessionPolicy.LoadSessionPolicy(ctx)
	if readyErr != nil {
		return readyErr
	}
	if err := validateSeedPolicy(policy); err != nil {
		return err
	}
	if err := store.validateInstallationSeed(ctx); err != nil {
		return err
	}
	rows, readyErr := store.database.QueryContext(ctx, `SELECT
  replay.issuer_id,replay.token_id_digest,replay.claims_digest,replay.expires_at,
  tombstone.selector_kind,tombstone.selector_digest,
  tombstone.logout_issued_at,tombstone.logout_expires_at
FROM management_backchannel_logout_replays AS replay
FULL OUTER JOIN management_issuer_logout_tombstones AS tombstone ON FALSE
LIMIT 0`)
	if readyErr != nil {
		return fmt.Errorf("validate Management identity lifecycle schema: %w", readyErr)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close Management identity lifecycle schema validation: %w", err)
	}
	return store.validateBuiltInRoles(ctx)
}

func (store *Store) validateInstallationSeed(ctx context.Context) error {
	var count, version int64
	if err := store.database.QueryRowContext(ctx, `SELECT count(*), COALESCE(max(seed_version), 0)
FROM management_installation_state WHERE singleton = TRUE`).Scan(&count, &version); err != nil {
		return fmt.Errorf("validate Management installation seed: %w", err)
	}
	if count != 1 || version != seedVersion {
		return errors.New("management installation seed is missing or unsupported")
	}
	return nil
}

func (store *Store) validateBuiltInRoles(ctx context.Context) error {
	rows, err := store.database.QueryContext(ctx, `SELECT id::text, name, permissions, permissions_digest,
       namespace_id::text, builtin, status, revision
FROM management_roles WHERE builtin = TRUE ORDER BY name`)
	if err != nil {
		return fmt.Errorf("load built-in Management roles: %w", err)
	}
	defer rows.Close()
	found := make(map[accesscontrol.BuiltInRoleName]struct{}, 8)
	for rows.Next() {
		var (
			id, name, status string
			permissionJSON   []byte
			digest           []byte
			namespace        sql.NullString
			builtIn          bool
			revision         int64
		)
		if err := rows.Scan(&id, &name, &permissionJSON, &digest, &namespace, &builtIn, &status, &revision); err != nil {
			return fmt.Errorf("scan built-in Management role: %w", err)
		}
		roleName := accesscontrol.BuiltInRoleName(name)
		expected, ok := accesscontrol.BuiltInRole(roleName)
		if !ok || !builtIn || namespace.Valid || status != string(accesscontrol.RoleStatusActive) || revision != 1 ||
			!canonicalUUID(id) || id != builtInRoleID(roleName) {
			return errors.New("built-in Management role seed is invalid")
		}
		permissions, canonical, err := decodePermissionSet(permissionJSON)
		if err != nil || !permissions.Equal(expected.Permissions) || len(digest) != sha256.Size {
			return errors.New("built-in Management role permission seed is invalid")
		}
		expectedDigest := sha256.Sum256(canonical)
		if !slices.Equal(digest, expectedDigest[:]) {
			return errors.New("built-in Management role permission digest was modified")
		}
		if _, duplicate := found[roleName]; duplicate {
			return errors.New("built-in Management role seed is duplicated")
		}
		found[roleName] = struct{}{}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate built-in Management roles: %w", err)
	}
	if len(found) != 8 {
		return errors.New("built-in Management role seed is incomplete")
	}
	return nil
}

func validateSeedPolicy(policy managementauth.SessionPolicy) error {
	if policy.SeedVersion != seedVersion || len(policy.ActionRequirements) != 1 {
		return errors.New("management session policy seed is unsupported")
	}
	requirement, found := policy.ActionRequirements["cluster_sensitive"]
	if !found || len(requirement.AnyOf) != 2 {
		return errors.New("management session policy seed is incomplete")
	}
	return nil
}

func builtInRoleID(name accesscontrol.BuiltInRoleName) string {
	index := map[accesscontrol.BuiltInRoleName]int{
		accesscontrol.BuiltInRoleClusterAdmin: 1, accesscontrol.BuiltInRolePlatformAdmin: 2,
		accesscontrol.BuiltInRoleOperator: 3, accesscontrol.BuiltInRoleAccessAdmin: 4,
		accesscontrol.BuiltInRoleCredentialRevealer: 5, accesscontrol.BuiltInRoleAnalyst: 6,
		accesscontrol.BuiltInRoleViewer: 7, accesscontrol.BuiltInRoleConsumer: 8,
	}[name]
	if index == 0 {
		return ""
	}
	return fmt.Sprintf("10000000-0000-5000-8000-%012d", index)
}

func decodePermissionSet(document []byte) (accesscontrol.PermissionSet, []byte, error) {
	var values []string
	if err := json.Unmarshal(document, &values); err != nil || len(values) == 0 {
		return accesscontrol.PermissionSet{}, nil, errors.New("management permission set is invalid")
	}
	permissions := make([]accesscontrol.Permission, len(values))
	for index, value := range values {
		permissions[index] = accesscontrol.Permission(value)
	}
	set, err := accesscontrol.NewPermissionSet(permissions...)
	if err != nil || len(set.Permissions()) != len(values) {
		return accesscontrol.PermissionSet{}, nil, errors.New("management permission set contains unknown or duplicate permissions")
	}
	canonical, err := json.Marshal(set.Permissions())
	return set, canonical, err
}

func encodePermissionSet(set accesscontrol.PermissionSet) ([]byte, [sha256.Size]byte, error) {
	if err := set.ValidateDelegable(); err != nil || set.Empty() {
		return nil, [sha256.Size]byte{}, errors.New("management permission set is invalid")
	}
	document, err := json.Marshal(set.Permissions())
	if err != nil {
		return nil, [sha256.Size]byte{}, err
	}
	return document, sha256.Sum256(document), nil
}

func inTransaction[T any](ctx context.Context, store *Store, isolation sql.IsolationLevel, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.database.BeginTx(ctx, &sql.TxOptions{Isolation: isolation})
	if err != nil {
		return zero, fmt.Errorf("begin Management identity transaction: %w", err)
	}
	value, err := operation(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit Management identity transaction: %w", err)
	}
	return value, nil
}

func validateList(request managementidentity.ListRequest) error {
	if request.Limit < 1 || request.Limit > maximumPageSize {
		return errors.New("management identity page size must be between 1 and 200")
	}
	if request.AfterID != "" && !canonicalUUID(request.AfterID) {
		return errors.New("management identity page cursor is invalid")
	}
	return nil
}

func canonicalUUID(value string) bool {
	parsed, err := uuid.Parse(value)
	return err == nil && parsed.String() == value
}

func utc(value time.Time) time.Time { return value.UTC() }

var (
	_ managementidentity.Repository          = (*Store)(nil)
	_ managementidentity.LifecycleRepository = (*Store)(nil)
)
