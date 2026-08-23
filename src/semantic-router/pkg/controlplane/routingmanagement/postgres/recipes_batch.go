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

// loadRecipesTx resolves one list page with two fixed-shape queries and then
// restores the keyset order selected by the page query.
func loadRecipesTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	ids []string,
) ([]routingmanagement.Recipe, error) {
	if len(ids) == 0 {
		return []routingmanagement.Recipe{}, nil
	}
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT p.namespace_id,p.id,r.name,p.description,p.status,p.revision,
p.created_at,p.updated_at,r.revision,r.description,r.document,
provenance.distribution_id,provenance.distribution_version,provenance.asset_digest,
provenance.source_recipe_id,provenance.source_recipe_revision,provenance.recipe_digest,
provenance.installed_at
FROM routing_recipes p
JOIN routing_recipe_revisions r ON r.recipe_id=p.id AND r.revision=p.current_revision
LEFT JOIN routing_recipe_provenance provenance
	ON provenance.namespace_id=p.namespace_id AND provenance.recipe_id=p.id
WHERE p.namespace_id=$1 AND p.id=ANY($2::text[]) AND p.deleted_at IS NULL`, namespaceID, pq.Array(ids))
	if queryContextErr != nil {
		return nil, fmt.Errorf("read routing Recipe page: %w", queryContextErr)
	}
	byID := make(map[string]routingmanagement.Recipe, len(ids))
	for rows.Next() {
		recipe, err := scanRecipeRow(rows)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("scan routing Recipe page: %w", err), rows.Close())
		}
		byID[recipe.ID] = recipe
	}
	if err := rows.Close(); err != nil {
		return nil, fmt.Errorf("close routing Recipe page: %w", err)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate routing Recipe page: %w", err)
	}

	decisionRows, queryContextErr := tx.QueryContext(ctx, `SELECT d.recipe_id,d.decision_id,d.name,d.dispatch_cardinality
FROM routing_recipe_decisions d
JOIN routing_recipes p ON p.id=d.recipe_id
WHERE p.namespace_id=$1 AND p.id=ANY($2::text[]) AND p.deleted_at IS NULL
	AND d.recipe_revision=p.current_revision
ORDER BY d.recipe_id,d.ordinal`, namespaceID, pq.Array(ids))
	if queryContextErr != nil {
		return nil, fmt.Errorf("read routing Recipe page decisions: %w", queryContextErr)
	}
	for decisionRows.Next() {
		var recipeID string
		var decision routingsnapshot.Decision
		if err := decisionRows.Scan(
			&recipeID, &decision.ID, &decision.Name, &decision.DispatchCardinality,
		); err != nil {
			return nil, errors.Join(fmt.Errorf("scan routing Recipe page decision: %w", err), decisionRows.Close())
		}
		recipe, found := byID[recipeID]
		if !found {
			return nil, errors.Join(
				fmt.Errorf("routing Recipe decision references missing page Recipe %q", recipeID),
				decisionRows.Close(),
			)
		}
		recipe.Current.Decisions = append(recipe.Current.Decisions, decision)
		byID[recipeID] = recipe
	}
	if err := decisionRows.Close(); err != nil {
		return nil, fmt.Errorf("close routing Recipe page decisions: %w", err)
	}
	if err := decisionRows.Err(); err != nil {
		return nil, fmt.Errorf("iterate routing Recipe page decisions: %w", err)
	}

	result := make([]routingmanagement.Recipe, 0, len(ids))
	for _, id := range ids {
		recipe, found := byID[id]
		if !found {
			return nil, fmt.Errorf("routing Recipe %q disappeared while listing", id)
		}
		result = append(result, recipe)
	}
	return result, nil
}
