package postgres

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

type manifestState struct {
	models      []routingmanagement.Model
	recipes     []routingmanagement.Recipe
	entrypoints []routingmanagement.Entrypoint
	currency    string
	revision    int64
}

type manifestPlan struct {
	snapshot       *routingsnapshot.Snapshot
	diff           routingmanagement.ManifestDiff
	models         map[string]routingmanagement.Model
	recipes        map[string]routingmanagement.Recipe
	entries        map[string]routingmanagement.Entrypoint
	writeModels    map[string]bool
	writeRecipes   map[string]bool
	writeEntries   map[string]bool
	targetIDs      []string
	disableModels  []string
	disableRecipes []string
	disableEntries []string
}

func (store *Store) PreviewManifest(ctx context.Context, namespaceID string, expected int64, source *routingsnapshot.Snapshot) (routingmanagement.ManifestDiff, error) {
	return inReadTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.ManifestDiff, error) {
		state, err := loadManifestState(ctx, tx, namespaceID, false)
		if err != nil {
			return routingmanagement.ManifestDiff{}, err
		}
		if state.revision != expected {
			return routingmanagement.ManifestDiff{}, routingmanagement.ErrConflict
		}
		plan, err := buildManifestPlan(namespaceID, source, state)
		if err != nil {
			return routingmanagement.ManifestDiff{}, err
		}
		if err := validateManifestCredentials(ctx, tx, namespaceID, plan.snapshot); err != nil {
			return routingmanagement.ManifestDiff{}, err
		}
		return plan.diff, nil
	})
}

func (store *Store) CurrentManifest(ctx context.Context, namespaceID string) (*routingsnapshot.Snapshot, int64, error) {
	type result struct {
		snapshot *routingsnapshot.Snapshot
		revision int64
	}
	value, err := inReadTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		state, err := loadManifestState(ctx, tx, namespaceID, false)
		if err != nil {
			return result{}, err
		}
		bundle := routingsnapshot.Bundle{NamespaceID: namespaceID, Revision: max64(1, state.revision), Currency: state.currency}
		for _, model := range state.models {
			if model.Status == routingmanagement.StatusActive {
				bundle.Models = append(bundle.Models, model.Current)
			}
		}
		for _, recipe := range state.recipes {
			if recipe.Status == routingmanagement.StatusActive {
				bundle.Recipes = append(bundle.Recipes, recipe.Current)
			}
		}
		for _, entrypoint := range state.entrypoints {
			if entrypoint.Status == routingmanagement.StatusActive {
				bundle.Entrypoints = append(bundle.Entrypoints, entrypoint.Current)
			}
		}
		snapshot, err := routingsnapshot.Compile(bundle)
		return result{snapshot: snapshot, revision: state.revision}, err
	})
	return value.snapshot, value.revision, err
}

func (store *Store) ImportManifest(ctx context.Context, namespaceID string, expected int64, source *routingsnapshot.Snapshot, meta routingmanagement.MutationContext) (routingmanagement.ManifestDiff, routingmanagement.RevisionReceipt, error) {
	type result struct {
		diff    routingmanagement.ManifestDiff
		receipt routingmanagement.RevisionReceipt
	}
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (result, error) {
		if meta.Command == nil {
			return result{}, fmt.Errorf("%w: manifest import requires an idempotent command", routingmanagement.ErrInvalid)
		}
		stored, replayed, err := commandpostgres.Lock(ctx, tx, *meta.Command)
		if err != nil {
			return result{}, classifyCommandError(err)
		}
		if replayed {
			if stored.Operation == nil || stored.Operation.DesiredRevision == nil {
				return result{}, managementcommand.ErrConflict
			}
			desiredRevision, revisionErr := postgresRevision(*stored.Operation.DesiredRevision, "stored manifest desired revision")
			if revisionErr != nil {
				return result{}, managementcommand.ErrConflict
			}
			diff, diffErr := manifestOperationDiff(ctx, tx, namespaceID, stored.Operation.OperationID)
			if diffErr != nil {
				return result{}, diffErr
			}
			return result{diff: diff, receipt: routingmanagement.RevisionReceipt{
				OperationID: stored.Operation.OperationID, DesiredRevision: desiredRevision, Replayed: true,
			}}, nil
		}
		state, err := loadManifestState(ctx, tx, namespaceID, true)
		if err != nil {
			return result{}, err
		}
		if state.revision != expected {
			return result{}, routingmanagement.ErrConflict
		}
		plan, err := buildManifestPlan(namespaceID, source, state)
		if err != nil {
			return result{}, err
		}
		if credentialErr := validateManifestCredentials(ctx, tx, namespaceID, plan.snapshot); credentialErr != nil {
			return result{}, credentialErr
		}
		if applyErr := applyManifestPlan(ctx, tx, namespaceID, plan, meta); applyErr != nil {
			return result{}, applyErr
		}
		receipt, err := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_manifest", resourceID: "routing_manifest", resourceRevision: expected + 1,
			action: "routing.manifest.import", operation: "updated",
			references: map[string]string{"semanticDigest": plan.snapshot.SemanticDigest},
		}, meta, true)
		if err != nil {
			return result{}, err
		}
		bundle, err := LoadPublishedBundle(ctx, tx, namespaceID, state.currency, receipt.DesiredRevision)
		if err != nil {
			return result{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, err)
		}
		published, err := routingsnapshot.Compile(bundle)
		if err != nil {
			return result{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, err)
		}
		if err := store.validatePublication(published); err != nil {
			return result{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, err)
		}
		operationID := uuid.NewString()
		active := meta.Command.ActiveDigest()
		actorChain, _ := json.Marshal(meta.ActorChain)
		targetScope, _ := json.Marshal(map[string]any{"diff": plan.diff})
		targetIDs, _ := json.Marshal(manifestTargetIDs(plan))
		if _, err := tx.ExecContext(ctx, `INSERT INTO management_operations
  (id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
   progress_total,target_scope,target_ids,desired_revision)
VALUES ($1,$2,'routing.manifest.import',$3,$4,$5,'pending',1,$6,$7,$8)`,
			operationID, namespaceID, meta.PrincipalID, actorChain, active.RequestDigest[:],
			targetScope, targetIDs, receipt.DesiredRevision); err != nil {
			return result{}, fmt.Errorf("insert routing manifest Operation: %w", err)
		}
		desired, revisionErr := publicRevision(receipt.DesiredRevision, "manifest desired revision")
		if revisionErr != nil {
			return result{}, revisionErr
		}
		if err := commandpostgres.CompleteOperation(ctx, tx, *meta.Command, managementcommand.OperationResult{
			OperationID: operationID, DesiredRevision: &desired, ResponseStatus: 202,
		}); err != nil {
			return result{}, err
		}
		receipt.OperationID = operationID
		return result{diff: plan.diff, receipt: receipt}, nil
	})
	if err != nil {
		return routingmanagement.ManifestDiff{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.diff, value.receipt, nil
}

func manifestOperationDiff(
	ctx context.Context, tx *sql.Tx, namespaceID, operationID string,
) (routingmanagement.ManifestDiff, error) {
	var targetScope []byte
	if err := tx.QueryRowContext(ctx, `SELECT target_scope FROM management_operations
WHERE namespace_id=$1 AND id=$2`, namespaceID, operationID).Scan(&targetScope); err != nil {
		return routingmanagement.ManifestDiff{}, err
	}
	var value struct {
		Diff routingmanagement.ManifestDiff `json:"diff"`
	}
	if err := json.Unmarshal(targetScope, &value); err != nil {
		return routingmanagement.ManifestDiff{}, fmt.Errorf("decode routing manifest Operation result: %w", err)
	}
	return value.Diff, nil
}

func validateManifestCredentials(
	ctx context.Context, tx *sql.Tx, namespaceID string, snapshot *routingsnapshot.Snapshot,
) error {
	seen := make(map[string]struct{})
	for _, model := range snapshot.Models {
		for _, backend := range model.Backends {
			if backend.ProviderCredentialID != "" {
				seen[backend.ProviderCredentialID] = struct{}{}
			}
		}
	}
	for credentialID := range seen {
		var active bool
		err := tx.QueryRowContext(ctx, `SELECT status='active' FROM provider_credentials
WHERE namespace_id=$1 AND id=$2`, namespaceID, credentialID).Scan(&active)
		if errors.Is(err, sql.ErrNoRows) || (err == nil && !active) {
			return fmt.Errorf("%w: referenced Provider Credential is unavailable", routingmanagement.ErrPublication)
		}
		if err != nil {
			return err
		}
	}
	return nil
}

func loadManifestState(ctx context.Context, tx *sql.Tx, namespaceID string, lock bool) (manifestState, error) {
	var state manifestState
	lockClause := ""
	if lock {
		lockClause = " FOR UPDATE"
	}
	if err := tx.QueryRowContext(ctx, `SELECT billing_currency FROM access_namespaces WHERE id=$1 AND status='active'`+lockClause, namespaceID).Scan(&state.currency); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return state, routingmanagement.ErrNotFound
		}
		return state, err
	}
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(revision),0) FROM policy_revisions WHERE namespace_id=$1`, namespaceID).Scan(&state.revision); err != nil {
		return state, err
	}
	modelIDs, err := manifestIDs(ctx, tx, `SELECT id FROM routing_models WHERE namespace_id=$1 AND deleted_at IS NULL ORDER BY id`, namespaceID)
	if err != nil {
		return state, err
	}
	state.models, err = loadModelsTx(ctx, tx, namespaceID, modelIDs)
	if err != nil {
		return state, err
	}
	recipeIDs, err := manifestIDs(ctx, tx, `SELECT id FROM routing_recipes WHERE namespace_id=$1 AND deleted_at IS NULL ORDER BY id`, namespaceID)
	if err != nil {
		return state, err
	}
	state.recipes, err = loadRecipesTx(ctx, tx, namespaceID, recipeIDs)
	if err != nil {
		return state, err
	}
	entryIDs, err := manifestIDs(ctx, tx, `SELECT id FROM routing_entrypoints WHERE namespace_id=$1 AND deleted_at IS NULL ORDER BY id`, namespaceID)
	if err != nil {
		return state, err
	}
	for _, id := range entryIDs {
		entrypoint, loadErr := loadEntrypointTx(ctx, tx, namespaceID, id)
		if loadErr != nil {
			return state, loadErr
		}
		state.entrypoints = append(state.entrypoints, entrypoint)
	}
	return state, nil
}

func manifestIDs(ctx context.Context, tx *sql.Tx, query, namespaceID string) (_ []string, returnErr error) {
	rows, err := tx.QueryContext(ctx, query, namespaceID)
	if err != nil {
		return nil, err
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, err
		}
		ids = append(ids, id)
	}
	return ids, rows.Err()
}

func buildManifestPlan(namespaceID string, source *routingsnapshot.Snapshot, state manifestState) (manifestPlan, error) {
	return buildManifestPlanPhases(namespaceID, source, state)
}

func applyManifestPlan(ctx context.Context, tx *sql.Tx, namespaceID string, plan manifestPlan, meta routingmanagement.MutationContext) error {
	for _, id := range plan.disableEntries {
		if _, err := tx.ExecContext(ctx, `UPDATE routing_entrypoints SET status='disabled',published_revision=NULL,revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, id); err != nil {
			return err
		}
	}
	for _, id := range plan.disableModels {
		if _, err := tx.ExecContext(ctx, `UPDATE routing_models SET status='disabled',revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, id); err != nil {
			return err
		}
	}
	for _, id := range plan.disableRecipes {
		if _, err := tx.ExecContext(ctx, `UPDATE routing_recipes SET status='disabled',revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, id); err != nil {
			return err
		}
	}
	for _, model := range plan.snapshot.Models {
		_, exists := plan.models[model.ID]
		if plan.writeModels[model.ID] {
			if insertErr := insertModelRevision(ctx, tx, namespaceID, model, meta.PrincipalID, !exists); insertErr != nil {
				return insertErr
			}
		}
		if exists {
			if _, err := tx.ExecContext(ctx, `UPDATE routing_models SET name=$3,aliases=$4,status='active',current_revision=$5,revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, model.ID, model.Name, mustJSON(model.Aliases), model.Revision); err != nil {
				return err
			}
		} else if _, err := tx.ExecContext(ctx, `UPDATE routing_models SET status='active' WHERE namespace_id=$1 AND id=$2`, namespaceID, model.ID); err != nil {
			return err
		}
	}
	for _, recipe := range plan.snapshot.Recipes {
		_, exists := plan.recipes[recipe.ID]
		if !exists {
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_recipes (id,namespace_id,name,description,status,current_revision,revision) VALUES ($1,$2,$3,$4,'active',$5,1)`, recipe.ID, namespaceID, recipe.Name, recipe.Description, recipe.Revision); err != nil {
				return classifyWriteError(err)
			}
		}
		if plan.writeRecipes[recipe.ID] {
			if err := insertRecipeRevision(ctx, tx, recipe, meta.PrincipalID); err != nil {
				return err
			}
		}
		if exists {
			if _, err := tx.ExecContext(ctx, `UPDATE routing_recipes SET name=$3,description=$4,status='active',current_revision=$5,revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, recipe.ID, recipe.Name, recipe.Description, recipe.Revision); err != nil {
				return err
			}
		}
	}
	for _, entrypoint := range plan.snapshot.Entrypoints {
		_, exists := plan.entries[entrypoint.ID]
		if !exists {
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_entrypoints (id,namespace_id,name,aliases,status,current_revision,published_revision,revision) VALUES ($1,$2,$3,$4,'active',$5,$5,1)`, entrypoint.ID, namespaceID, entrypoint.Name, mustJSON(entrypoint.Aliases), entrypoint.Revision); err != nil {
				return classifyWriteError(err)
			}
		}
		if plan.writeEntries[entrypoint.ID] {
			if err := insertEntrypointRevision(ctx, tx, entrypoint, meta.PrincipalID); err != nil {
				return err
			}
		}
		if exists {
			if _, err := tx.ExecContext(ctx, `UPDATE routing_entrypoints SET name=$3,aliases=$4,status='active',current_revision=$5,published_revision=$5,revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2`, namespaceID, entrypoint.ID, entrypoint.Name, mustJSON(entrypoint.Aliases), entrypoint.Revision); err != nil {
				return err
			}
		}
	}
	return nil
}

func manifestTargetIDs(plan manifestPlan) []string {
	ids := append([]string{}, plan.targetIDs...)
	sort.Strings(ids)
	return slices.Compact(ids)
}

func sortManifestDiff(diff *routingmanagement.ManifestDiff) {
	for _, values := range []*[]string{
		&diff.Models.Create, &diff.Models.Update, &diff.Models.Disable,
		&diff.Recipes.Create, &diff.Recipes.Update, &diff.Recipes.Disable,
		&diff.Entrypoints.Create, &diff.Entrypoints.Update, &diff.Entrypoints.Disable,
	} {
		sort.Strings(*values)
	}
}

func generatedManifestID(prefix string) string {
	return prefix + "_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:20]
}

func max64(left, right int64) int64 {
	if left > right {
		return left
	}
	return right
}

func equalManifestValue(left, right any) bool {
	a, _ := json.Marshal(left)
	b, _ := json.Marshal(right)
	return bytes.Equal(a, b)
}
