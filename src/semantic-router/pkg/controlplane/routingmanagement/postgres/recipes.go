package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (store *Store) ListRecipes(
	ctx context.Context, namespaceID string, request routingmanagement.ListQuery,
) (routingmanagement.ListResult[routingmanagement.Recipe], error) {
	if err := validateListQuery(request); err != nil {
		return routingmanagement.ListResult[routingmanagement.Recipe]{}, err
	}
	scope, err := normalizeListScope(namespaceID, accesscontrol.ScopeResourceRecipe, request)
	if err != nil {
		return routingmanagement.ListResult[routingmanagement.Recipe]{}, err
	}
	if !scope.all && len(scope.ids) == 0 {
		return routingmanagement.ListResult[routingmanagement.Recipe]{Items: []routingmanagement.Recipe{}}, nil
	}
	return inReadTransaction(ctx, store, func(
		tx *sql.Tx,
	) (_ routingmanagement.ListResult[routingmanagement.Recipe], returnErr error) {
		var cursorTime, cursorID any
		if request.After != nil {
			cursorTime, cursorID = request.After.CreatedAt, request.After.ID
		}
		query := `SELECT id FROM routing_recipes
WHERE namespace_id=$1 AND deleted_at IS NULL
	AND ($2 OR id = ANY($3::text[]))
	AND ($4='' OR status=$4)
	AND ($5::timestamptz IS NULL OR (created_at,id)<($5,$6))
ORDER BY created_at DESC,id DESC LIMIT $7`
		arguments := []any{
			namespaceID, scope.all, pq.Array(scope.ids), request.Status,
			cursorTime, cursorID, request.Limit + 1,
		}
		if request.Search != "" {
			query = `SELECT id FROM routing_recipes
WHERE namespace_id=$1 AND deleted_at IS NULL
	AND ($2 OR id = ANY($3::text[]))
	AND ($4='' OR status=$4)
	AND (lower(name) LIKE $5 ESCAPE E'\\' OR id LIKE $5 ESCAPE E'\\')
	AND ($6::timestamptz IS NULL OR (created_at,id)<($6,$7))
ORDER BY created_at DESC,id DESC LIMIT $8`
			arguments = []any{
				namespaceID, scope.all, pq.Array(scope.ids), request.Status,
				managementsearch.PrefixPattern(request.Search), cursorTime, cursorID, request.Limit + 1,
			}
		}
		rows, listRecipesErr := tx.QueryContext(ctx, query, arguments...)
		if listRecipesErr != nil {
			return routingmanagement.ListResult[routingmanagement.Recipe]{}, fmt.Errorf("list routing Recipes: %w", listRecipesErr)
		}
		defer func() {
			returnErr = errors.Join(returnErr, rows.Close())
		}()
		ids := make([]string, 0, request.Limit+1)
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				return routingmanagement.ListResult[routingmanagement.Recipe]{}, err
			}
			ids = append(ids, id)
		}
		if err := rows.Err(); err != nil {
			return routingmanagement.ListResult[routingmanagement.Recipe]{}, err
		}
		result := routingmanagement.ListResult[routingmanagement.Recipe]{HasMore: len(ids) > request.Limit}
		if result.HasMore {
			ids = ids[:request.Limit]
		}
		result.Items, listRecipesErr = loadRecipesTx(ctx, tx, namespaceID, ids)
		if listRecipesErr != nil {
			return routingmanagement.ListResult[routingmanagement.Recipe]{}, listRecipesErr
		}
		return result, nil
	})
}

func (store *Store) GetRecipe(ctx context.Context, namespaceID, id string) (routingmanagement.Recipe, error) {
	return inReadTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.Recipe, error) {
		return loadRecipeTx(ctx, tx, namespaceID, id)
	})
}

func loadRecipeTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	id string,
) (_ routingmanagement.Recipe, returnErr error) {
	result, err := scanRecipeRow(tx.QueryRowContext(ctx, `SELECT p.namespace_id,p.id,r.name,p.description,p.status,p.revision,
p.created_at,p.updated_at,r.revision,r.description,r.document,
provenance.distribution_id,provenance.distribution_version,provenance.asset_digest,
provenance.source_recipe_id,provenance.source_recipe_revision,provenance.recipe_digest,
provenance.installed_at
FROM routing_recipes p JOIN routing_recipe_revisions r ON r.recipe_id=p.id AND r.revision=p.current_revision
LEFT JOIN routing_recipe_provenance provenance
  ON provenance.namespace_id=p.namespace_id AND provenance.recipe_id=p.id
WHERE p.namespace_id=$1 AND p.id=$2 AND p.deleted_at IS NULL`, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return routingmanagement.Recipe{}, routingmanagement.ErrNotFound
	}
	if err != nil {
		return routingmanagement.Recipe{}, fmt.Errorf("read routing Recipe: %w", err)
	}
	rows, err := tx.QueryContext(ctx, `SELECT decision_id,name,dispatch_cardinality FROM routing_recipe_decisions
WHERE recipe_id=$1 AND recipe_revision=$2 ORDER BY ordinal`, id, result.Current.Revision)
	if err != nil {
		return routingmanagement.Recipe{}, fmt.Errorf("read routing Recipe decisions: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	for rows.Next() {
		var decision routingsnapshot.Decision
		if err := rows.Scan(&decision.ID, &decision.Name, &decision.DispatchCardinality); err != nil {
			return routingmanagement.Recipe{}, err
		}
		result.Current.Decisions = append(result.Current.Decisions, decision)
	}
	return result, rows.Err()
}

func (store *Store) CreateRecipe(
	ctx context.Context, namespaceID, description string, recipe routingsnapshot.Recipe,
	meta routingmanagement.MutationContext,
) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (recipeMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return recipeMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayRecipe(ctx, tx, namespaceID, replay)
			}
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipes
  (id,namespace_id,name,description,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'draft',$5,1)`, recipe.ID, namespaceID, recipe.Name, description, recipe.Revision); err != nil {
			return recipeMutationResult{}, classifyWriteError(err)
		}
		if err := insertRecipeRevision(ctx, tx, recipe, meta.PrincipalID); err != nil {
			return recipeMutationResult{}, err
		}
		receipt, createRecipeErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_recipe", resourceID: recipe.ID, resourceRevision: 1,
			action: "routing.recipe.create", operation: "created",
		}, meta, false)
		if createRecipeErr != nil {
			return recipeMutationResult{}, createRecipeErr
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_recipe", ResourceID: recipe.ID, ResourceRevision: 1, ResponseStatus: 201,
			}); err != nil {
				return recipeMutationResult{}, err
			}
		}
		created, createRecipeErr := loadRecipeTx(ctx, tx, namespaceID, recipe.ID)
		return recipeMutationResult{recipe: created, receipt: receipt}, createRecipeErr
	})
	if err != nil {
		return routingmanagement.Recipe{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.recipe, value.receipt, nil
}

func insertRecipeRevision(ctx context.Context, tx *sql.Tx, recipe routingsnapshot.Recipe, principalID string) error {
	digestRecipe := recipe
	digestRecipe.Revision = 0
	digest := contentDigest(digestRecipe)
	if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipe_revisions
	(recipe_id,revision,name,description,document,content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6,$7)`, recipe.ID, recipe.Revision, recipe.Name, recipe.Description, []byte(recipe.Document), digest,
		actorValue(principalID)); err != nil {
		return classifyWriteError(err)
	}
	for index, decision := range recipe.Decisions {
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipe_decisions
  (recipe_id,recipe_revision,decision_id,name,dispatch_cardinality,ordinal,capabilities)
VALUES ($1,$2,$3,$4,$5,$6,'{}'::jsonb)`, recipe.ID, recipe.Revision, decision.ID,
			decision.Name, decision.DispatchCardinality, index); err != nil {
			return classifyWriteError(err)
		}
	}
	return nil
}

func (store *Store) UpdateRecipe(
	ctx context.Context, namespaceID, id string, expected int64, description string,
	recipe routingsnapshot.Recipe, meta routingmanagement.MutationContext,
) (routingmanagement.Recipe, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (recipeMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return recipeMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayRecipe(ctx, tx, namespaceID, replay)
			}
		}
		if immutable, err := immutableRecipeTx(ctx, tx, namespaceID, id); err != nil {
			return recipeMutationResult{}, err
		} else if immutable {
			return recipeMutationResult{}, routingmanagement.ErrImmutable
		}
		updated, updateRecipeErr := tx.ExecContext(ctx, `UPDATE routing_recipes SET name=$4,description=$5,
current_revision=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`, namespaceID, id,
			expected, recipe.Name, description, recipe.Revision)
		if updateRecipeErr != nil {
			return recipeMutationResult{}, classifyWriteError(updateRecipeErr)
		}
		rows, _ := updated.RowsAffected()
		if rows != 1 {
			return recipeMutationResult{}, routingmanagement.ErrConflict
		}
		if err := insertRecipeRevision(ctx, tx, recipe, meta.PrincipalID); err != nil {
			return recipeMutationResult{}, err
		}
		receipt, updateRecipeErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_recipe", resourceID: id, resourceRevision: expected + 1,
			action: "routing.recipe.update", operation: "updated",
		}, meta, false)
		if updateRecipeErr != nil {
			return recipeMutationResult{}, updateRecipeErr
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_recipe", ResourceID: id,
				ResourceRevision: uint64(expected + 1), ResponseStatus: 200,
			}); err != nil {
				return recipeMutationResult{}, err
			}
		}
		current, updateRecipeErr := loadRecipeTx(ctx, tx, namespaceID, id)
		return recipeMutationResult{recipe: current, receipt: receipt}, updateRecipeErr
	})
	if err != nil {
		return routingmanagement.Recipe{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.recipe, value.receipt, nil
}

func (store *Store) DeleteRecipe(
	ctx context.Context, namespaceID, id string, expected int64, meta routingmanagement.MutationContext,
) (routingmanagement.RevisionReceipt, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.RevisionReceipt, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return routingmanagement.RevisionReceipt{}, classifyCommandError(err)
			}
			if found {
				return replayResourceReceipt(replay, "routing_recipe", id)
			}
		}
		if immutable, err := immutableRecipeTx(ctx, tx, namespaceID, id); err != nil {
			return routingmanagement.RevisionReceipt{}, err
		} else if immutable {
			return routingmanagement.RevisionReceipt{}, routingmanagement.ErrImmutable
		}
		result, err := tx.ExecContext(ctx, `UPDATE routing_recipes SET status='deleted',deleted_at=clock_timestamp(),
revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`,
			namespaceID, id, expected)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, classifyWriteError(err)
		}
		rows, _ := result.RowsAffected()
		if rows != 1 {
			return routingmanagement.RevisionReceipt{}, routingmanagement.ErrConflict
		}
		receipt, err := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_recipe", resourceID: id, resourceRevision: expected + 1,
			action: "routing.recipe.delete", operation: "deleted",
		}, meta, false)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, err
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_recipe", ResourceID: id,
				ResourceRevision: uint64(expected + 1), ResponseStatus: 204,
			}); err != nil {
				return routingmanagement.RevisionReceipt{}, err
			}
		}
		return receipt, nil
	})
}

func replayRecipe(
	ctx context.Context, tx *sql.Tx, namespaceID string, stored managementcommand.StoredResult,
) (recipeMutationResult, error) {
	var result recipeMutationResult
	if stored.Resource == nil || stored.Resource.ResourceType != "routing_recipe" {
		return result, managementcommand.ErrConflict
	}
	recipe, err := loadRecipeTx(ctx, tx, namespaceID, stored.Resource.ResourceID)
	if err != nil {
		return result, managementcommand.ErrConflict
	}
	result.recipe = recipe
	result.receipt = routingmanagement.RevisionReceipt{
		ResourceRevision: int64(stored.Resource.ResourceRevision), Replayed: true,
	}
	return result, nil
}

type recipeMutationResult struct {
	recipe  routingmanagement.Recipe
	receipt routingmanagement.RevisionReceipt
}

func immutableRecipeTx(ctx context.Context, tx *sql.Tx, namespaceID, id string) (bool, error) {
	var immutable bool
	if err := tx.QueryRowContext(ctx, `SELECT EXISTS(
  SELECT 1 FROM routing_recipe_provenance WHERE namespace_id=$1 AND recipe_id=$2
)`, namespaceID, id).Scan(&immutable); err != nil {
		return false, fmt.Errorf("read routing Recipe provenance: %w", err)
	}
	return immutable, nil
}
