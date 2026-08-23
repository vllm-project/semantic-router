package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (store *Store) PendingBuiltInRecipeNamespaces(
	ctx context.Context,
	distribution routingmanagement.BuiltInRecipeDistribution,
	limit int,
) (_ []string, returnErr error) {
	if err := distribution.Validate(); err != nil {
		return nil, err
	}
	if limit < 1 || limit > 1024 {
		return nil, fmt.Errorf("%w: built-in Recipe namespace batch is invalid", routingmanagement.ErrInvalid)
	}
	rows, err := store.db.QueryContext(ctx, `SELECT namespace_row.id::text
FROM access_namespaces namespace_row
LEFT JOIN routing_recipe_distributions installed
  ON installed.namespace_id=namespace_row.id
 AND installed.distribution_id=$1 AND installed.distribution_version=$2
WHERE namespace_row.status='active' AND installed.namespace_id IS NULL
ORDER BY namespace_row.id LIMIT $3`, distribution.ID, distribution.Version, limit)
	if err != nil {
		return nil, fmt.Errorf("list pending built-in Recipe namespaces: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	var result []string
	for rows.Next() {
		var namespaceID string
		if err := rows.Scan(&namespaceID); err != nil {
			return nil, fmt.Errorf("scan pending built-in Recipe namespace: %w", err)
		}
		result = append(result, namespaceID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate pending built-in Recipe namespaces: %w", err)
	}
	return result, nil
}

func (store *Store) InstallBuiltInRecipes(
	ctx context.Context,
	namespaceID string,
	distribution routingmanagement.BuiltInRecipeDistribution,
	meta routingmanagement.MutationContext,
) ([]routingmanagement.Recipe, error) {
	const maximumSerializationAttempts = 4
	for attempt := 1; attempt <= maximumSerializationAttempts; attempt++ {
		installed, err := store.installBuiltInRecipesOnce(ctx, namespaceID, distribution, meta)
		if err == nil || !errors.Is(err, routingmanagement.ErrConflict) || attempt == maximumSerializationAttempts {
			return installed, err
		}
	}
	return nil, routingmanagement.ErrConflict
}

func (store *Store) installBuiltInRecipesOnce(
	ctx context.Context,
	namespaceID string,
	distribution routingmanagement.BuiltInRecipeDistribution,
	meta routingmanagement.MutationContext,
) ([]routingmanagement.Recipe, error) {
	recipes, installBuiltInRecipesOnceErr := distribution.RecipesForNamespace(namespaceID)
	if installBuiltInRecipesOnceErr != nil {
		return nil, installBuiltInRecipesOnceErr
	}
	assetDigest, installBuiltInRecipesOnceErr := decodedDigest(distribution.AssetDigest)
	if installBuiltInRecipesOnceErr != nil {
		return nil, installBuiltInRecipesOnceErr
	}
	value, installBuiltInRecipesOnceErr := inTransaction(ctx, store, func(tx *sql.Tx) ([]routingmanagement.Recipe, error) {
		if _, err := tx.ExecContext(ctx, `SELECT pg_advisory_xact_lock($1)`, builtInRecipeLockKey(
			namespaceID, distribution.ID, distribution.Version,
		)); err != nil {
			return nil, fmt.Errorf("lock built-in Recipe distribution: %w", err)
		}
		if err := ensureDistributionHeaderTx(ctx, tx, namespaceID, distribution, assetDigest); err != nil {
			return nil, err
		}

		installed := make([]routingmanagement.Recipe, 0, len(recipes))
		for index, recipe := range recipes {
			member := distribution.Recipes[index]
			recipeDocumentDigest, installBuiltInRecipesOnceErr2 := decodedDigest(member.RecipeDigest)
			if installBuiltInRecipesOnceErr2 != nil {
				return nil, installBuiltInRecipesOnceErr2
			}
			current, found, installBuiltInRecipesOnceErr2 := installedDistributionRecipeTx(
				ctx, tx, namespaceID, distribution, member, recipe, assetDigest, recipeDocumentDigest,
			)
			if installBuiltInRecipesOnceErr2 != nil {
				return nil, installBuiltInRecipesOnceErr2
			}
			if found {
				installed = append(installed, current)
				continue
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipes
  (id,namespace_id,name,description,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'draft',$5,1)`, recipe.ID, namespaceID, recipe.Name,
				member.Input.Description, recipe.Revision); err != nil {
				return nil, classifyWriteError(err)
			}
			if err := insertRecipeRevision(ctx, tx, recipe, ""); err != nil {
				return nil, err
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipe_provenance
  (namespace_id,recipe_id,recipe_revision,distribution_id,distribution_version,
   source_recipe_id,source_recipe_revision,asset_digest,recipe_digest)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)`, namespaceID, recipe.ID, recipe.Revision,
				distribution.ID, distribution.Version, member.SourceID, member.SourceRevision,
				assetDigest, recipeDocumentDigest); err != nil {
				return nil, classifyWriteError(err)
			}
			if _, err := appendMutation(ctx, tx, namespaceID, mutationRecord{
				resourceType: "routing_recipe", resourceID: recipe.ID, resourceRevision: 1,
				action: "routing.recipe.distribution.install", operation: "created",
				references: map[string]string{
					"distributionId": distribution.ID, "distributionVersion": distribution.Version,
					"sourceRecipeId": member.SourceID, "assetDigest": distribution.AssetDigest,
				},
			}, meta, false); err != nil {
				return nil, err
			}
			current, installBuiltInRecipesOnceErr2 = loadRecipeTx(ctx, tx, namespaceID, recipe.ID)
			if installBuiltInRecipesOnceErr2 != nil {
				return nil, installBuiltInRecipesOnceErr2
			}
			installed = append(installed, current)
		}
		return installed, nil
	})
	if installBuiltInRecipesOnceErr != nil {
		return nil, classifyWriteError(installBuiltInRecipesOnceErr)
	}
	return value, nil
}

func ensureDistributionHeaderTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	distribution routingmanagement.BuiltInRecipeDistribution,
	assetDigest []byte,
) error {
	if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipe_distributions
  (namespace_id,distribution_id,distribution_version,asset_digest,recipe_count)
VALUES ($1,$2,$3,$4,$5)
ON CONFLICT (namespace_id,distribution_id,distribution_version) DO NOTHING`,
		namespaceID, distribution.ID, distribution.Version, assetDigest, len(distribution.Recipes)); err != nil {
		return classifyWriteError(err)
	}
	var storedDigest []byte
	var storedCount int
	if err := tx.QueryRowContext(ctx, `SELECT asset_digest,recipe_count
FROM routing_recipe_distributions
WHERE namespace_id=$1 AND distribution_id=$2 AND distribution_version=$3
FOR UPDATE`, namespaceID, distribution.ID, distribution.Version).Scan(&storedDigest, &storedCount); err != nil {
		return fmt.Errorf("read built-in Recipe distribution: %w", err)
	}
	if !bytes.Equal(storedDigest, assetDigest) || storedCount != len(distribution.Recipes) {
		return fmt.Errorf("%w: built-in Recipe distribution %s@%s changed in place",
			routingmanagement.ErrConflict, distribution.ID, distribution.Version)
	}
	return nil
}

func installedDistributionRecipeTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	distribution routingmanagement.BuiltInRecipeDistribution,
	member routingmanagement.BuiltInRecipe,
	recipe routingsnapshot.Recipe,
	assetDigest, recipeDocumentDigest []byte,
) (routingmanagement.Recipe, bool, error) {
	var recipeID string
	var recipeRevision int64
	var storedAssetDigest, storedRecipeDigest []byte
	err := tx.QueryRowContext(ctx, `SELECT recipe_id,recipe_revision,asset_digest,recipe_digest
FROM routing_recipe_provenance
WHERE namespace_id=$1 AND distribution_id=$2 AND distribution_version=$3 AND source_recipe_id=$4
FOR UPDATE`, namespaceID, distribution.ID, distribution.Version, member.SourceID).Scan(
		&recipeID, &recipeRevision, &storedAssetDigest, &storedRecipeDigest,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return routingmanagement.Recipe{}, false, nil
	}
	if err != nil {
		return routingmanagement.Recipe{}, false, fmt.Errorf("read built-in Recipe provenance: %w", err)
	}
	if recipeID != recipe.ID || recipeRevision != recipe.Revision ||
		!bytes.Equal(storedAssetDigest, assetDigest) || !bytes.Equal(storedRecipeDigest, recipeDocumentDigest) {
		return routingmanagement.Recipe{}, false, fmt.Errorf("%w: built-in Recipe provenance changed in place", routingmanagement.ErrConflict)
	}
	current, err := loadRecipeTx(ctx, tx, namespaceID, recipeID)
	if err != nil {
		return routingmanagement.Recipe{}, false, err
	}
	expected := recipe
	expected.Revision = 0
	var storedContentDigest []byte
	if err := tx.QueryRowContext(ctx, `SELECT content_digest FROM routing_recipe_revisions
WHERE recipe_id=$1 AND revision=$2`, recipeID, recipeRevision).Scan(&storedContentDigest); err != nil {
		return routingmanagement.Recipe{}, false, fmt.Errorf("read built-in Recipe revision digest: %w", err)
	}
	if current.Revision != 1 || current.Current.Revision != recipeRevision || current.Status != routingmanagement.StatusDraft ||
		!bytes.Equal(storedContentDigest, contentDigest(expected)) {
		return routingmanagement.Recipe{}, false, fmt.Errorf("%w: immutable built-in Recipe was modified", routingmanagement.ErrConflict)
	}
	return current, true, nil
}

func (store *Store) VerifyBuiltInRecipes(
	ctx context.Context,
	distribution routingmanagement.BuiltInRecipeDistribution,
) error {
	if err := distribution.Validate(); err != nil {
		return err
	}
	assetDigest, err := decodedDigest(distribution.AssetDigest)
	if err != nil {
		return err
	}
	var pending int
	if err := store.db.QueryRowContext(ctx, `SELECT count(*)
FROM access_namespaces namespace_row
LEFT JOIN routing_recipe_distributions installed
  ON installed.namespace_id=namespace_row.id
 AND installed.distribution_id=$1 AND installed.distribution_version=$2
WHERE namespace_row.status='active' AND installed.namespace_id IS NULL`,
		distribution.ID, distribution.Version).Scan(&pending); err != nil {
		return fmt.Errorf("verify built-in Recipe namespace coverage: %w", err)
	}
	if pending != 0 {
		return fmt.Errorf("built-in Recipe distribution has %d pending namespaces", pending)
	}
	var invalid int
	if err := store.db.QueryRowContext(ctx, `SELECT count(*)
FROM routing_recipe_distributions installed
WHERE installed.distribution_id=$1 AND installed.distribution_version=$2
  AND (installed.asset_digest<>$3 OR installed.recipe_count<>$4 OR
    (SELECT count(*) FROM routing_recipe_provenance provenance
     WHERE provenance.namespace_id=installed.namespace_id
       AND provenance.distribution_id=installed.distribution_id
       AND provenance.distribution_version=installed.distribution_version)<>installed.recipe_count)`,
		distribution.ID, distribution.Version, assetDigest, len(distribution.Recipes)).Scan(&invalid); err != nil {
		return fmt.Errorf("verify built-in Recipe distribution: %w", err)
	}
	if invalid != 0 {
		return fmt.Errorf("%w: built-in Recipe distribution provenance is inconsistent", routingmanagement.ErrConflict)
	}
	return store.verifyBuiltInRecipeMembers(ctx, distribution, assetDigest)
}

func (store *Store) verifyBuiltInRecipeMembers(
	ctx context.Context,
	distribution routingmanagement.BuiltInRecipeDistribution,
	assetDigest []byte,
) (returnErr error) {
	members := make(map[string]routingmanagement.BuiltInRecipe, len(distribution.Recipes))
	for _, member := range distribution.Recipes {
		members[member.SourceID] = member
	}
	rows, err := store.db.QueryContext(ctx, `SELECT provenance.namespace_id::text,
  provenance.source_recipe_id,provenance.source_recipe_revision,
  provenance.recipe_id,provenance.recipe_revision,provenance.asset_digest,provenance.recipe_digest,
  recipe.name,recipe.description,recipe.status,recipe.revision,recipe.current_revision,
  revision.name,revision.content_digest
FROM routing_recipe_provenance provenance
JOIN routing_recipes recipe
  ON recipe.namespace_id=provenance.namespace_id AND recipe.id=provenance.recipe_id
JOIN routing_recipe_revisions revision
  ON revision.recipe_id=provenance.recipe_id AND revision.revision=provenance.recipe_revision
WHERE provenance.distribution_id=$1 AND provenance.distribution_version=$2
ORDER BY provenance.namespace_id,provenance.source_recipe_id`, distribution.ID, distribution.Version)
	if err != nil {
		return fmt.Errorf("verify immutable built-in Recipes: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()

	expectedByNamespace := make(map[string]map[string]routingsnapshot.Recipe)
	counts := make(map[string]int)
	for rows.Next() {
		var namespaceID, sourceID, recipeID, recipeName, description, status, revisionName string
		var sourceRevision, recipeRevision, resourceRevision, currentRevision int64
		var storedAssetDigest, storedRecipeDigest, storedContentDigest []byte
		if err := rows.Scan(
			&namespaceID, &sourceID, &sourceRevision, &recipeID, &recipeRevision,
			&storedAssetDigest, &storedRecipeDigest, &recipeName, &description, &status,
			&resourceRevision, &currentRevision, &revisionName, &storedContentDigest,
		); err != nil {
			return fmt.Errorf("scan immutable built-in Recipe: %w", err)
		}
		member, exists := members[sourceID]
		if !exists {
			return fmt.Errorf("%w: built-in Recipe provenance contains unknown source %q", routingmanagement.ErrConflict, sourceID)
		}
		expectedRecipes, exists := expectedByNamespace[namespaceID]
		if !exists {
			compiled, err := distribution.RecipesForNamespace(namespaceID)
			if err != nil {
				return err
			}
			expectedRecipes = make(map[string]routingsnapshot.Recipe, len(compiled))
			for index, expected := range compiled {
				expectedRecipes[distribution.Recipes[index].SourceID] = expected
			}
			expectedByNamespace[namespaceID] = expectedRecipes
		}
		expected := expectedRecipes[sourceID]
		expectedDigest, err := decodedDigest(member.RecipeDigest)
		if err != nil {
			return err
		}
		digestRecipe := expected
		digestRecipe.Revision = 0
		if sourceRevision != member.SourceRevision || recipeID != expected.ID || recipeRevision != expected.Revision ||
			!bytes.Equal(storedAssetDigest, assetDigest) || !bytes.Equal(storedRecipeDigest, expectedDigest) ||
			recipeName != expected.Name || revisionName != expected.Name || description != member.Input.Description ||
			status != string(routingmanagement.StatusDraft) || resourceRevision != 1 || currentRevision != expected.Revision ||
			!bytes.Equal(storedContentDigest, contentDigest(digestRecipe)) {
			return fmt.Errorf("%w: immutable built-in Recipe %q is inconsistent", routingmanagement.ErrConflict, recipeID)
		}
		counts[namespaceID]++
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate immutable built-in Recipes: %w", err)
	}
	for namespaceID, count := range counts {
		if count != len(distribution.Recipes) {
			return fmt.Errorf("%w: Namespace %s has %d built-in Recipes, want %d",
				routingmanagement.ErrConflict, namespaceID, count, len(distribution.Recipes))
		}
	}
	return nil
}

func builtInRecipeLockKey(namespaceID, distributionID, version string) int64 {
	digest := sha256.Sum256([]byte(namespaceID + "\x00" + distributionID + "\x00" + version))
	return int64(binary.BigEndian.Uint64(digest[:8]))
}

func decodedDigest(value string) ([]byte, error) {
	const prefix = "sha256:"
	if len(value) != len(prefix)+sha256.Size*2 || value[:len(prefix)] != prefix {
		return nil, fmt.Errorf("%w: content digest is invalid", routingmanagement.ErrInvalid)
	}
	result, err := hex.DecodeString(value[len(prefix):])
	if err != nil || len(result) != sha256.Size {
		return nil, fmt.Errorf("%w: content digest is invalid", routingmanagement.ErrInvalid)
	}
	return result, nil
}

func encodedDigest(value []byte) string {
	if len(value) != sha256.Size {
		return ""
	}
	return "sha256:" + hex.EncodeToString(value)
}

var _ routingmanagement.BuiltInRecipeStore = (*Store)(nil)
