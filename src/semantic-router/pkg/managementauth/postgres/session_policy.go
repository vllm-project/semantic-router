package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

const sessionPolicyDetailQuery = `SELECT
  access_token_ttl_seconds, session_ttl_seconds, max_active_sessions,
  action_requirements, seed_version, revision, updated_at
FROM management_session_policy
WHERE singleton = TRUE`

func (store *Store) LoadSessionPolicy(ctx context.Context) (managementauth.SessionPolicy, error) {
	if store == nil || store.db == nil {
		return managementauth.SessionPolicy{}, errors.New("management session PostgreSQL store is unavailable")
	}
	var (
		accessTTL, sessionTTL int64
		policy                managementauth.SessionPolicy
		actionJSON            []byte
		seedVersion, revision int64
	)
	err := store.db.QueryRowContext(ctx, sessionPolicyDetailQuery).Scan(
		&accessTTL, &sessionTTL, &policy.MaxActiveSessions, &actionJSON,
		&seedVersion, &revision, &policy.UpdatedAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return managementauth.SessionPolicy{}, errors.New("management session policy seed is missing")
	}
	if err != nil {
		return managementauth.SessionPolicy{}, fmt.Errorf("load Management session policy: %w", err)
	}
	if accessTTL <= 0 || sessionTTL <= 0 || seedVersion <= 0 || revision <= 0 {
		return managementauth.SessionPolicy{}, errors.New("management session policy seed is invalid")
	}
	policy.AccessTokenTTL = time.Duration(accessTTL) * time.Second
	policy.SessionTTL = time.Duration(sessionTTL) * time.Second
	policy.SeedVersion = uint64(seedVersion)
	policy.Revision = uint64(revision)
	policy.UpdatedAt = policy.UpdatedAt.UTC()
	if err := decodeStrictJSON(actionJSON, &policy.ActionRequirements); err != nil {
		return managementauth.SessionPolicy{}, fmt.Errorf("decode Management session policy requirements: %w", err)
	}
	if err := policy.Validate(); err != nil {
		return managementauth.SessionPolicy{}, fmt.Errorf("validate Management session policy: %w", err)
	}
	return policy, nil
}

func decodeStrictJSON(document []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(document))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New("JSON document contains trailing data")
	}
	return nil
}
