package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (store *Store) ListModels(
	ctx context.Context, namespaceID string, request routingmanagement.ListQuery,
) (routingmanagement.ListResult[routingmanagement.Model], error) {
	if err := validateListQuery(request); err != nil {
		return routingmanagement.ListResult[routingmanagement.Model]{}, err
	}
	scope, err := normalizeListScope(namespaceID, accesscontrol.ScopeResourceModel, request)
	if err != nil {
		return routingmanagement.ListResult[routingmanagement.Model]{}, err
	}
	if !scope.all && len(scope.ids) == 0 {
		return routingmanagement.ListResult[routingmanagement.Model]{Items: []routingmanagement.Model{}}, nil
	}
	return inReadTransaction(ctx, store, func(
		tx *sql.Tx,
	) (_ routingmanagement.ListResult[routingmanagement.Model], returnErr error) {
		var cursorTime, cursorID any
		if request.After != nil {
			cursorTime, cursorID = request.After.CreatedAt, request.After.ID
		}
		query := `SELECT id FROM routing_models
WHERE namespace_id=$1 AND deleted_at IS NULL
	AND ($2 OR id = ANY($3::text[]))
	AND ($4 = '' OR status = $4)
	AND ($5::timestamptz IS NULL OR (created_at, id) < ($5, $6))
ORDER BY created_at DESC, id DESC LIMIT $7`
		arguments := []any{
			namespaceID, scope.all, pq.Array(scope.ids), request.Status,
			cursorTime, cursorID, request.Limit + 1,
		}
		if request.Search != "" {
			query = `SELECT id FROM routing_models
WHERE namespace_id=$1 AND deleted_at IS NULL
	AND ($2 OR id = ANY($3::text[]))
	AND ($4 = '' OR status = $4)
	AND (lower(name) LIKE $5 ESCAPE E'\\' OR id LIKE $5 ESCAPE E'\\')
	AND ($6::timestamptz IS NULL OR (created_at, id) < ($6, $7))
ORDER BY created_at DESC, id DESC LIMIT $8`
			arguments = []any{
				namespaceID, scope.all, pq.Array(scope.ids), request.Status,
				managementsearch.PrefixPattern(request.Search), cursorTime, cursorID, request.Limit + 1,
			}
		}
		rows, listModelsErr := tx.QueryContext(ctx, query, arguments...)
		if listModelsErr != nil {
			return routingmanagement.ListResult[routingmanagement.Model]{}, fmt.Errorf("list routing Models: %w", listModelsErr)
		}
		defer func() {
			returnErr = errors.Join(returnErr, rows.Close())
		}()
		ids := make([]string, 0, request.Limit+1)
		for rows.Next() {
			var id string
			if err := rows.Scan(&id); err != nil {
				return routingmanagement.ListResult[routingmanagement.Model]{}, err
			}
			ids = append(ids, id)
		}
		if err := rows.Err(); err != nil {
			return routingmanagement.ListResult[routingmanagement.Model]{}, err
		}
		result := routingmanagement.ListResult[routingmanagement.Model]{HasMore: len(ids) > request.Limit}
		if result.HasMore {
			ids = ids[:request.Limit]
		}
		result.Items, listModelsErr = loadModelsTx(ctx, tx, namespaceID, ids)
		if listModelsErr != nil {
			return routingmanagement.ListResult[routingmanagement.Model]{}, listModelsErr
		}
		return result, nil
	})
}

func (store *Store) GetModel(ctx context.Context, namespaceID, id string) (routingmanagement.Model, error) {
	return inReadTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.Model, error) {
		return loadModelTx(ctx, tx, namespaceID, id)
	})
}

func loadModelTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	id string,
) (_ routingmanagement.Model, returnErr error) {
	result, err := scanModelRow(tx.QueryRowContext(ctx, `SELECT m.namespace_id,m.id,r.name,m.status,m.revision,m.created_at,m.updated_at,
r.revision,r.provider_catalog_revision,r.aliases,r.param_size,r.context_window_size,r.description,
r.capabilities,r.reasoning,r.loras,r.quality_score,r.modality,r.tags,r.execution,r.pricing
FROM routing_models m
JOIN routing_model_revisions r ON r.model_id=m.id AND r.revision=m.current_revision
WHERE m.namespace_id=$1 AND m.id=$2 AND m.deleted_at IS NULL`, namespaceID, id))
	if errors.Is(err, sql.ErrNoRows) {
		return routingmanagement.Model{}, routingmanagement.ErrNotFound
	}
	if err != nil {
		return routingmanagement.Model{}, fmt.Errorf("read routing Model: %w", err)
	}
	rows, err := tx.QueryContext(ctx, `SELECT id,provider_id,wire_format,normalized_origin,
provider_model_id,provider_credential_id,connection,weight::text
FROM routing_model_backends WHERE model_id=$1 AND model_revision=$2 ORDER BY ordinal`, id, result.Current.Revision)
	if err != nil {
		return routingmanagement.Model{}, fmt.Errorf("read routing Model backends: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	for rows.Next() {
		var backend routingsnapshot.Backend
		var credential sql.NullString
		var connection []byte
		if err := rows.Scan(&backend.ID, &backend.ProviderID, &backend.WireFormat, &backend.Origin,
			&backend.ProviderModelID, &credential, &connection, &backend.Weight); err != nil {
			return routingmanagement.Model{}, fmt.Errorf("scan routing Model backend: %w", err)
		}
		if credential.Valid {
			backend.ProviderCredentialID = credential.String
		}
		if err := strictJSON(connection, &backend.Connection); err != nil {
			return routingmanagement.Model{}, fmt.Errorf("decode routing Model backend: %w", err)
		}
		result.Current.Backends = append(result.Current.Backends, backend)
	}
	return result, rows.Err()
}

type modelsMutationResult struct {
	models  []routingmanagement.Model
	receipt routingmanagement.RevisionReceipt
}

func (store *Store) CreateModels(
	ctx context.Context,
	namespaceID string,
	models []routingsnapshot.Model,
	meta routingmanagement.MutationContext,
) ([]routingmanagement.Model, routingmanagement.RevisionReceipt, error) {
	if len(models) == 0 || len(models) > 200 {
		return nil, routingmanagement.RevisionReceipt{}, fmt.Errorf("%w: Model batch size is invalid", routingmanagement.ErrInvalid)
	}
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (modelsMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return modelsMutationResult{}, classifyCommandError(err)
			}
			if found {
				if len(models) == 1 {
					return replayModel(ctx, tx, namespaceID, replay)
				}
				return replayModelBatch(replay)
			}
		}
		for _, model := range models {
			if err := insertModelRevision(ctx, tx, namespaceID, model, meta.PrincipalID, true); err != nil {
				return modelsMutationResult{}, err
			}
		}
		aggregateID := models[0].ID
		if len(models) > 1 {
			aggregateID = "model_batch_" + strings.ReplaceAll(uuid.NewString(), "-", "")[:20]
		}
		receipt, err := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_model", resourceID: aggregateID, resourceRevision: 1,
			action: "routing.model.create", operation: "created",
		}, meta, false)
		if err != nil {
			return modelsMutationResult{}, err
		}
		created := make([]routingmanagement.Model, len(models))
		for index := range models {
			created[index], err = loadModelTx(ctx, tx, namespaceID, models[index].ID)
			if err != nil {
				return modelsMutationResult{}, err
			}
		}
		if len(models) == 1 && meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_model", ResourceID: models[0].ID,
				ResourceRevision: 1, ResponseStatus: 201,
			}); err != nil {
				return modelsMutationResult{}, err
			}
		} else if meta.Command != nil {
			activeDigest := meta.Command.ActiveDigest()
			operationID := uuid.NewString()
			actorChain, _ := json.Marshal(meta.ActorChain)
			targetIDs := make([]string, len(models))
			for index := range models {
				targetIDs[index] = models[index].ID
			}
			targets, _ := json.Marshal(targetIDs)
			scope, _ := json.Marshal(map[string]string{"resourceType": "routing_model"})
			if _, err := tx.ExecContext(ctx, `INSERT INTO management_operations
  (id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
   progress_completed,progress_total,target_scope,target_ids)
VALUES ($1,$2,'routing.model.bulk_import',$3,$4,$5,'succeeded',$6,$6,$7,$8)`,
				operationID, namespaceID, meta.PrincipalID, actorChain, activeDigest.RequestDigest[:],
				len(models), scope, targets); err != nil {
				return modelsMutationResult{}, fmt.Errorf("insert Model bulk-import operation: %w", err)
			}
			if err := commandpostgres.CompleteOperation(ctx, tx, *meta.Command, managementcommand.OperationResult{
				OperationID: operationID, ResponseStatus: 202,
			}); err != nil {
				return modelsMutationResult{}, err
			}
			receipt.OperationID = operationID
		}
		return modelsMutationResult{models: created, receipt: receipt}, nil
	})
	if err != nil {
		return nil, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.models, value.receipt, nil
}

func replayModelBatch(stored managementcommand.StoredResult) (modelsMutationResult, error) {
	if stored.Operation == nil || stored.Operation.DesiredRevision != nil {
		return modelsMutationResult{}, managementcommand.ErrConflict
	}
	return modelsMutationResult{receipt: routingmanagement.RevisionReceipt{
		OperationID: stored.Operation.OperationID, Replayed: true,
	}}, nil
}

func insertModelRevision(
	ctx context.Context, tx *sql.Tx, namespaceID string, model routingsnapshot.Model, principalID string, create bool,
) error {
	aliases, _ := json.Marshal(model.Aliases)
	capabilities, _ := json.Marshal(model.Capabilities)
	reasoning, _ := json.Marshal(model.Reasoning)
	loras, _ := json.Marshal(model.LoRAs)
	tags, _ := json.Marshal(model.Tags)
	execution, _ := json.Marshal(model.Execution)
	pricing, _ := json.Marshal(model.Pricing)
	digestModel := model
	digestModel.Revision = 0
	digestPayload, _ := json.Marshal(digestModel)
	digest := sha256.Sum256(digestPayload)
	if create {
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_models
  (id,namespace_id,name,aliases,status,current_revision,revision)
VALUES ($1,$2,$3,$4,'draft',$5,1)`, model.ID, namespaceID, model.Name, aliases, model.Revision); err != nil {
			return fmt.Errorf("insert routing Model: %w", classifyWriteError(err))
		}
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO routing_model_revisions
  (model_id,revision,provider_catalog_revision,name,aliases,param_size,context_window_size,
   description,capabilities,reasoning,loras,quality_score,modality,tags,execution,pricing,
   content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18)`, model.ID, model.Revision,
		model.CatalogRevision, model.Name, aliases, model.ParamSize, model.ContextWindowSize,
		model.Description, capabilities, reasoning, loras, model.QualityScore, model.Modality, tags,
		execution, pricing, digest[:], actorValue(principalID)); err != nil {
		return fmt.Errorf("insert routing Model revision: %w", classifyWriteError(err))
	}
	for index, backend := range model.Backends {
		connection, _ := json.Marshal(backend.Connection)
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_model_backends
  (id,namespace_id,model_id,model_revision,ordinal,provider_id,wire_format,
   normalized_origin,provider_model_id,provider_credential_id,connection,weight)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)`, backend.ID, namespaceID, model.ID,
			model.Revision, index, backend.ProviderID, backend.WireFormat, backend.Origin,
			backend.ProviderModelID, nullableString(backend.ProviderCredentialID), connection, backend.Weight); err != nil {
			return fmt.Errorf("insert routing Model backend: %w", classifyWriteError(err))
		}
	}
	return nil
}

func (store *Store) UpdateModel(
	ctx context.Context, namespaceID, id string, expected int64, model routingsnapshot.Model,
	meta routingmanagement.MutationContext,
) (routingmanagement.Model, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (modelsMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return modelsMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayModel(ctx, tx, namespaceID, replay)
			}
		}
		result, updateModelErr := tx.ExecContext(ctx, `UPDATE routing_models SET name=$4,aliases=$5,current_revision=$6,
revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`, namespaceID, id, expected,
			model.Name, mustJSON(model.Aliases), model.Revision)
		if updateModelErr != nil {
			return modelsMutationResult{}, classifyWriteError(updateModelErr)
		}
		rows, _ := result.RowsAffected()
		if rows != 1 {
			return modelsMutationResult{}, routingmanagement.ErrConflict
		}
		if err := insertModelRevision(ctx, tx, namespaceID, model, meta.PrincipalID, false); err != nil {
			return modelsMutationResult{}, err
		}
		receipt, updateModelErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_model", resourceID: id, resourceRevision: expected + 1,
			action: "routing.model.update", operation: "updated",
		}, meta, false)
		if updateModelErr != nil {
			return modelsMutationResult{}, updateModelErr
		}
		updated, updateModelErr := loadModelTx(ctx, tx, namespaceID, id)
		if updateModelErr != nil {
			return modelsMutationResult{}, updateModelErr
		}
		if meta.Command != nil {
			resourceRevision, revisionErr := publicRevision(expected+1, "updated Model revision")
			if revisionErr != nil {
				return modelsMutationResult{}, revisionErr
			}
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_model", ResourceID: id, ResourceRevision: resourceRevision,
				ResponseStatus: 200,
			}); err != nil {
				return modelsMutationResult{}, err
			}
		}
		return modelsMutationResult{models: []routingmanagement.Model{updated}, receipt: receipt}, nil
	})
	if err != nil {
		return routingmanagement.Model{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.models[0], value.receipt, nil
}

func (store *Store) DeleteModel(
	ctx context.Context, namespaceID, id string, expected int64, meta routingmanagement.MutationContext,
) (routingmanagement.RevisionReceipt, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.RevisionReceipt, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return routingmanagement.RevisionReceipt{}, classifyCommandError(err)
			}
			if found {
				return replayResourceReceipt(replay, "routing_model", id)
			}
		}
		result, err := tx.ExecContext(ctx, `UPDATE routing_models SET status='deleted',deleted_at=clock_timestamp(),
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
			resourceType: "routing_model", resourceID: id, resourceRevision: expected + 1,
			action: "routing.model.delete", operation: "deleted",
		}, meta, false)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, err
		}
		if meta.Command != nil {
			resourceRevision, revisionErr := publicRevision(expected+1, "deleted Model revision")
			if revisionErr != nil {
				return routingmanagement.RevisionReceipt{}, revisionErr
			}
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_model", ResourceID: id,
				ResourceRevision: resourceRevision, ResponseStatus: 204,
			}); err != nil {
				return routingmanagement.RevisionReceipt{}, err
			}
		}
		return receipt, nil
	})
}

func replayModel(
	ctx context.Context, tx *sql.Tx, namespaceID string, stored managementcommand.StoredResult,
) (modelsMutationResult, error) {
	if stored.Resource == nil || stored.Resource.ResourceType != "routing_model" {
		return modelsMutationResult{}, managementcommand.ErrConflict
	}
	model, err := loadModelTx(ctx, tx, namespaceID, stored.Resource.ResourceID)
	if err != nil {
		return modelsMutationResult{}, managementcommand.ErrConflict
	}
	resourceRevision, revisionErr := postgresRevision(stored.Resource.ResourceRevision, "stored Model revision")
	if revisionErr != nil {
		return modelsMutationResult{}, managementcommand.ErrConflict
	}
	return modelsMutationResult{
		models: []routingmanagement.Model{model},
		receipt: routingmanagement.RevisionReceipt{
			ResourceRevision: resourceRevision,
			Replayed:         true,
		},
	}, nil
}

func classifyCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return fmt.Errorf("%w: %w", routingmanagement.ErrConflict, managementcommand.ErrConflict)
	}
	return err
}

func nullableString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func mustJSON(value any) []byte {
	result, _ := json.Marshal(value)
	return result
}

func contentDigest(value any) []byte {
	payload, _ := json.Marshal(value)
	digest := sha256.Sum256(payload)
	return digest[:]
}
