package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

func (store *Store) PutRegistryManifest(
	ctx context.Context, namespaceID string, manifest agentmanagement.RegistryManifest,
) error {
	document, err := json.Marshal(struct {
		Revision string                         `json:"revision"`
		Tools    []agentmanagement.ToolManifest `json:"tools"`
	}{Revision: manifest.Revision, Tools: manifest.Tools})
	if err != nil {
		return agentmanagement.ErrInvalid
	}
	result, err := store.db.ExecContext(ctx, `INSERT INTO agent_tool_registry_revisions
  (namespace_id,registry_revision,manifest,created_at,expires_at)
VALUES ($1,$2,$3,$4,$5)
ON CONFLICT (namespace_id,registry_revision) DO UPDATE
SET expires_at=GREATEST(agent_tool_registry_revisions.expires_at,EXCLUDED.expires_at)
WHERE agent_tool_registry_revisions.manifest=EXCLUDED.manifest`,
		namespaceID, manifest.Revision, document, manifest.CreatedAt.UTC(), manifest.ExpiresAt.UTC())
	if err != nil {
		return classifyWriteError(err)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return err
	}
	if rows != 1 {
		return agentmanagement.ErrConflict
	}
	return nil
}

func (store *Store) GetRegistryManifest(
	ctx context.Context, namespaceID, revision string,
) (agentmanagement.RegistryManifest, error) {
	var document []byte
	var createdAt, expiresAt time.Time
	err := store.db.QueryRowContext(ctx, `SELECT manifest,created_at,expires_at
FROM agent_tool_registry_revisions
WHERE namespace_id=$1 AND registry_revision=$2 AND expires_at>clock_timestamp()`,
		namespaceID, revision).Scan(&document, &createdAt, &expiresAt)
	if err == sql.ErrNoRows {
		return agentmanagement.RegistryManifest{}, agentmanagement.ErrNotFound
	}
	if err != nil {
		return agentmanagement.RegistryManifest{}, fmt.Errorf("get Agent Tool Registry revision: %w", err)
	}
	var persisted struct {
		Revision string                         `json:"revision"`
		Tools    []agentmanagement.ToolManifest `json:"tools"`
	}
	if err := json.Unmarshal(document, &persisted); err != nil || persisted.Revision != revision {
		return agentmanagement.RegistryManifest{}, agentmanagement.ErrConflict
	}
	return agentmanagement.RegistryManifest{
		Revision: persisted.Revision, Tools: persisted.Tools,
		CreatedAt: createdAt.UTC(), ExpiresAt: expiresAt.UTC(),
	}, nil
}
