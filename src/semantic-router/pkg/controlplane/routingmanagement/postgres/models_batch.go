package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// loadModelsTx resolves one list page with two fixed-shape queries. The
// caller-provided ID order is the stable keyset order and is restored after
// PostgreSQL returns the rows.
func loadModelsTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	ids []string,
) ([]routingmanagement.Model, error) {
	if len(ids) == 0 {
		return []routingmanagement.Model{}, nil
	}
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT m.namespace_id,m.id,r.name,m.status,m.revision,m.created_at,m.updated_at,
r.revision,r.provider_catalog_revision,r.aliases,r.param_size,r.context_window_size,r.description,
r.capabilities,r.reasoning,r.loras,r.quality_score,r.modality,r.tags,r.execution,r.pricing
FROM routing_models m
JOIN routing_model_revisions r ON r.model_id=m.id AND r.revision=m.current_revision
WHERE m.namespace_id=$1 AND m.id=ANY($2::text[]) AND m.deleted_at IS NULL`, namespaceID, pq.Array(ids))
	if queryContextErr != nil {
		return nil, fmt.Errorf("read routing Model page: %w", queryContextErr)
	}
	byID := make(map[string]routingmanagement.Model, len(ids))
	for rows.Next() {
		model, err := scanModelRow(rows)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("scan routing Model page: %w", err), rows.Close())
		}
		byID[model.ID] = model
	}
	if err := rows.Close(); err != nil {
		return nil, fmt.Errorf("close routing Model page: %w", err)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate routing Model page: %w", err)
	}

	backendRows, queryContextErr := tx.QueryContext(ctx, `SELECT b.model_id,b.id,b.provider_id,b.wire_format,b.normalized_origin,
b.provider_model_id,b.provider_credential_id,b.connection,b.weight::text
FROM routing_model_backends b
JOIN routing_models m ON m.id=b.model_id
WHERE m.namespace_id=$1 AND m.id=ANY($2::text[]) AND m.deleted_at IS NULL
	AND b.model_revision=m.current_revision
ORDER BY b.model_id,b.ordinal`, namespaceID, pq.Array(ids))
	if queryContextErr != nil {
		return nil, fmt.Errorf("read routing Model page backends: %w", queryContextErr)
	}
	for backendRows.Next() {
		var modelID string
		var backend routingsnapshot.Backend
		var credential sql.NullString
		var connection []byte
		if err := backendRows.Scan(
			&modelID, &backend.ID, &backend.ProviderID, &backend.WireFormat,
			&backend.Origin, &backend.ProviderModelID, &credential, &connection, &backend.Weight,
		); err != nil {
			return nil, errors.Join(fmt.Errorf("scan routing Model page backend: %w", err), backendRows.Close())
		}
		if credential.Valid {
			backend.ProviderCredentialID = credential.String
		}
		if err := strictJSON(connection, &backend.Connection); err != nil {
			return nil, errors.Join(fmt.Errorf("decode routing Model page backend: %w", err), backendRows.Close())
		}
		model, found := byID[modelID]
		if !found {
			return nil, errors.Join(
				fmt.Errorf("routing Model backend references missing page Model %q", modelID),
				backendRows.Close(),
			)
		}
		model.Current.Backends = append(model.Current.Backends, backend)
		byID[modelID] = model
	}
	if err := backendRows.Close(); err != nil {
		return nil, fmt.Errorf("close routing Model page backends: %w", err)
	}
	if err := backendRows.Err(); err != nil {
		return nil, fmt.Errorf("iterate routing Model page backends: %w", err)
	}

	result := make([]routingmanagement.Model, 0, len(ids))
	for _, id := range ids {
		model, found := byID[id]
		if !found {
			return nil, fmt.Errorf("routing Model %q disappeared while listing", id)
		}
		result = append(result, model)
	}
	return result, nil
}
