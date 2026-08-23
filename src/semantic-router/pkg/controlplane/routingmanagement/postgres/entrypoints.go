package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/google/uuid"
	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

func (store *Store) ListEntrypoints(
	ctx context.Context, namespaceID string, request routingmanagement.ListQuery,
) (routingmanagement.ListResult[routingmanagement.Entrypoint], error) {
	if err := validateListQuery(request); err != nil {
		return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, err
	}
	scope, err := normalizeListScope(namespaceID, accesscontrol.ScopeResourceEntrypoint, request)
	if err != nil {
		return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, err
	}
	if !scope.all && len(scope.ids) == 0 {
		return routingmanagement.ListResult[routingmanagement.Entrypoint]{Items: []routingmanagement.Entrypoint{}}, nil
	}
	return inReadTransaction(ctx, store, func(
		tx *sql.Tx,
	) (_ routingmanagement.ListResult[routingmanagement.Entrypoint], returnErr error) {
		var cursorTime, cursorID any
		if request.After != nil {
			cursorTime, cursorID = request.After.CreatedAt, request.After.ID
		}
		query := `WITH page AS (
  SELECT namespace_id,id,status,revision,created_at,updated_at,current_revision
  FROM routing_entrypoints
  WHERE namespace_id=$1 AND deleted_at IS NULL
    AND ($2 OR id = ANY($3::text[]))
    AND ($4='' OR status=$4)
    AND ($5::timestamptz IS NULL OR (created_at,id)<($5,$6))
  ORDER BY created_at DESC,id DESC LIMIT $7
)
SELECT page.namespace_id,page.id,revision.name,page.status,page.revision,
  page.created_at,page.updated_at,revision.revision,revision.aliases,
  (SELECT count(*) FROM routing_entrypoint_rules rule
   WHERE rule.entrypoint_id=page.id AND rule.entrypoint_revision=page.current_revision),
  (SELECT count(DISTINCT assignment.model_id) FROM routing_assignment_models assignment
   WHERE assignment.entrypoint_id=page.id AND assignment.entrypoint_revision=page.current_revision)
FROM page
JOIN routing_entrypoint_revisions revision
  ON revision.entrypoint_id=page.id AND revision.revision=page.current_revision
ORDER BY page.created_at DESC,page.id DESC`
		arguments := []any{
			namespaceID, scope.all, pq.Array(scope.ids), request.Status,
			cursorTime, cursorID, request.Limit + 1,
		}
		if request.Search != "" {
			query = `WITH page AS (
  SELECT namespace_id,id,status,revision,created_at,updated_at,current_revision
  FROM routing_entrypoints
  WHERE namespace_id=$1 AND deleted_at IS NULL
    AND ($2 OR id = ANY($3::text[]))
    AND ($4='' OR status=$4)
    AND (lower(name) LIKE $5 ESCAPE E'\\' OR id LIKE $5 ESCAPE E'\\')
    AND ($6::timestamptz IS NULL OR (created_at,id)<($6,$7))
  ORDER BY created_at DESC,id DESC LIMIT $8
)
SELECT page.namespace_id,page.id,revision.name,page.status,page.revision,
  page.created_at,page.updated_at,revision.revision,revision.aliases,
  (SELECT count(*) FROM routing_entrypoint_rules rule
   WHERE rule.entrypoint_id=page.id AND rule.entrypoint_revision=page.current_revision),
  (SELECT count(DISTINCT assignment.model_id) FROM routing_assignment_models assignment
   WHERE assignment.entrypoint_id=page.id AND assignment.entrypoint_revision=page.current_revision)
FROM page
JOIN routing_entrypoint_revisions revision
  ON revision.entrypoint_id=page.id AND revision.revision=page.current_revision
ORDER BY page.created_at DESC,page.id DESC`
			arguments = []any{
				namespaceID, scope.all, pq.Array(scope.ids), request.Status,
				managementsearch.PrefixPattern(request.Search), cursorTime, cursorID, request.Limit + 1,
			}
		}
		rows, err := tx.QueryContext(ctx, query, arguments...)
		if err != nil {
			return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, fmt.Errorf("list routing Entrypoints: %w", err)
		}
		defer func() {
			returnErr = errors.Join(returnErr, rows.Close())
		}()
		items := make([]routingmanagement.Entrypoint, 0, request.Limit+1)
		for rows.Next() {
			var entrypoint routingmanagement.Entrypoint
			var aliases []byte
			if err := rows.Scan(
				&entrypoint.NamespaceID, &entrypoint.ID, &entrypoint.Name, &entrypoint.Status,
				&entrypoint.Revision, &entrypoint.CreatedAt, &entrypoint.UpdatedAt,
				&entrypoint.Current.Revision, &aliases, &entrypoint.RuleCount,
				&entrypoint.AssignedModelCount,
			); err != nil {
				return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, err
			}
			entrypoint.Current.ID, entrypoint.Current.Name = entrypoint.ID, entrypoint.Name
			if err := strictJSON(aliases, &entrypoint.Current.Aliases); err != nil {
				return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, fmt.Errorf("decode routing Entrypoint aliases: %w", err)
			}
			items = append(items, entrypoint)
		}
		if err := rows.Err(); err != nil {
			return routingmanagement.ListResult[routingmanagement.Entrypoint]{}, err
		}
		result := routingmanagement.ListResult[routingmanagement.Entrypoint]{
			Items: items, HasMore: len(items) > request.Limit,
		}
		if result.HasMore {
			result.Items = result.Items[:request.Limit]
		}
		return result, nil
	})
}

func (store *Store) GetEntrypoint(ctx context.Context, namespaceID, id string) (routingmanagement.Entrypoint, error) {
	return inReadTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.Entrypoint, error) {
		return loadEntrypointTx(ctx, tx, namespaceID, id)
	})
}

func loadEntrypointTx(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	id string,
) (_ routingmanagement.Entrypoint, returnErr error) {
	var result routingmanagement.Entrypoint
	var aliases []byte
	loadEntrypointTxErr := tx.QueryRowContext(ctx, `SELECT e.namespace_id,e.id,r.name,e.status,e.revision,e.created_at,e.updated_at,
r.revision,r.aliases FROM routing_entrypoints e
JOIN routing_entrypoint_revisions r ON r.entrypoint_id=e.id AND r.revision=e.current_revision
WHERE e.namespace_id=$1 AND e.id=$2 AND e.deleted_at IS NULL`, namespaceID, id).Scan(
		&result.NamespaceID, &result.ID, &result.Name, &result.Status, &result.Revision,
		&result.CreatedAt, &result.UpdatedAt, &result.Current.Revision, &aliases,
	)
	if errors.Is(loadEntrypointTxErr, sql.ErrNoRows) {
		return routingmanagement.Entrypoint{}, routingmanagement.ErrNotFound
	}
	if loadEntrypointTxErr != nil {
		return routingmanagement.Entrypoint{}, fmt.Errorf("read routing Entrypoint: %w", loadEntrypointTxErr)
	}
	result.Current.ID, result.Current.Name = result.ID, result.Name
	if err := strictJSON(aliases, &result.Current.Aliases); err != nil {
		return routingmanagement.Entrypoint{}, fmt.Errorf("decode routing Entrypoint aliases: %w", err)
	}
	rows, loadEntrypointTxErr := tx.QueryContext(ctx, `SELECT id,name,matchers,recipe_id,recipe_revision
FROM routing_entrypoint_rules WHERE entrypoint_id=$1 AND entrypoint_revision=$2 ORDER BY ordinal`,
		id, result.Current.Revision)
	if loadEntrypointTxErr != nil {
		return routingmanagement.Entrypoint{}, fmt.Errorf("read routing Entrypoint rules: %w", loadEntrypointTxErr)
	}
	type ruleKey struct{ id string }
	rules := make(map[ruleKey]int)
	for rows.Next() {
		var rule routingsnapshot.EntrypointRule
		var matchers []byte
		if err := rows.Scan(&rule.ID, &rule.Name, &matchers, &rule.RecipeID, &rule.RecipeRevision); err != nil {
			return routingmanagement.Entrypoint{}, errors.Join(err, rows.Close())
		}
		if err := strictJSON(matchers, &rule.Matchers); err != nil {
			return routingmanagement.Entrypoint{}, errors.Join(err, rows.Close())
		}
		rule.Assignments = make(map[string]routingsnapshot.AssignmentSet)
		result.Current.Rules = append(result.Current.Rules, rule)
		rules[ruleKey{rule.ID}] = len(result.Current.Rules) - 1
	}
	if err := rows.Err(); err != nil {
		return routingmanagement.Entrypoint{}, errors.Join(err, rows.Close())
	}
	if err := rows.Close(); err != nil {
		return routingmanagement.Entrypoint{}, err
	}
	decisionRows, loadEntrypointTxErr := tx.QueryContext(ctx, `SELECT rule_id,decision_id,fallback_strategy,fallback_on
FROM routing_decision_assignments
WHERE entrypoint_id=$1 AND entrypoint_revision=$2 ORDER BY rule_id,decision_id`, id, result.Current.Revision)
	if loadEntrypointTxErr != nil {
		return routingmanagement.Entrypoint{}, fmt.Errorf("read routing Entrypoint decision assignments: %w", loadEntrypointTxErr)
	}
	for decisionRows.Next() {
		var ruleID, decisionID string
		var strategy sql.NullString
		var on []byte
		if err := decisionRows.Scan(&ruleID, &decisionID, &strategy, &on); err != nil {
			return routingmanagement.Entrypoint{}, errors.Join(err, decisionRows.Close())
		}
		ruleIndex, exists := rules[ruleKey{ruleID}]
		if !exists {
			return routingmanagement.Entrypoint{}, errors.Join(
				fmt.Errorf("stored decision assignment references absent rule"),
				decisionRows.Close(),
			)
		}
		rule := &result.Current.Rules[ruleIndex]
		assignmentSet := routingsnapshot.AssignmentSet{}
		if strategy.Valid {
			assignmentSet.Fallback = &routingsnapshot.FallbackPolicy{Strategy: strategy.String}
			if err := strictJSON(on, &assignmentSet.Fallback.On); err != nil {
				return routingmanagement.Entrypoint{}, errors.Join(err, decisionRows.Close())
			}
		}
		rule.Assignments[decisionID] = assignmentSet
	}
	if err := decisionRows.Err(); err != nil {
		return routingmanagement.Entrypoint{}, errors.Join(err, decisionRows.Close())
	}
	if err := decisionRows.Close(); err != nil {
		return routingmanagement.Entrypoint{}, err
	}
	assignmentRows, loadEntrypointTxErr := tx.QueryContext(ctx, `SELECT rule_id,decision_id,model_id,model_revision,
priority,weight::text,lora_name,reasoning FROM routing_assignment_models
WHERE entrypoint_id=$1 AND entrypoint_revision=$2 ORDER BY rule_id,decision_id,priority,ordinal`, id, result.Current.Revision)
	if loadEntrypointTxErr != nil {
		return routingmanagement.Entrypoint{}, fmt.Errorf("read routing Entrypoint assignments: %w", loadEntrypointTxErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, assignmentRows.Close())
	}()
	assignedModels := make(map[string]struct{})
	for assignmentRows.Next() {
		var ruleID, decisionID string
		var assignment routingsnapshot.Assignment
		var lora sql.NullString
		var reasoning []byte
		if err := assignmentRows.Scan(&ruleID, &decisionID, &assignment.ModelID, &assignment.ModelRevision,
			&assignment.Priority, &assignment.Weight, &lora, &reasoning); err != nil {
			return routingmanagement.Entrypoint{}, err
		}
		if lora.Valid {
			assignment.LoRAName = lora.String
		}
		if len(reasoning) != 0 && string(reasoning) != "null" {
			var value routingsnapshot.AssignmentReasoning
			if err := strictJSON(reasoning, &value); err != nil {
				return routingmanagement.Entrypoint{}, err
			}
			assignment.Reasoning = &value
		}
		ruleIndex, exists := rules[ruleKey{ruleID}]
		if !exists {
			return routingmanagement.Entrypoint{}, fmt.Errorf("stored assignment references absent rule")
		}
		rule := &result.Current.Rules[ruleIndex]
		assignmentSet, exists := rule.Assignments[decisionID]
		if !exists {
			return routingmanagement.Entrypoint{}, fmt.Errorf("stored Model assignment references absent decision assignment")
		}
		assignmentSet.Models = append(assignmentSet.Models, assignment)
		rule.Assignments[decisionID] = assignmentSet
		assignedModels[assignment.ModelID] = struct{}{}
	}
	if err := assignmentRows.Err(); err != nil {
		return routingmanagement.Entrypoint{}, err
	}
	result.RuleCount = len(result.Current.Rules)
	result.AssignedModelCount = len(assignedModels)
	return result, nil
}

func (store *Store) CreateEntrypoint(
	ctx context.Context, namespaceID string, entrypoint routingsnapshot.Entrypoint,
	meta routingmanagement.MutationContext,
) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (entrypointMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return entrypointMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayEntrypoint(ctx, tx, namespaceID, replay)
			}
		}
		aliases := mustJSON(entrypoint.Aliases)
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_entrypoints
  (id,namespace_id,name,aliases,status,current_revision,published_revision,revision)
VALUES ($1,$2,$3,$4,'draft',$5,NULL,1)`, entrypoint.ID, namespaceID, entrypoint.Name,
			aliases, entrypoint.Revision); err != nil {
			return entrypointMutationResult{}, classifyWriteError(err)
		}
		if err := insertEntrypointRevision(ctx, tx, entrypoint, meta.PrincipalID); err != nil {
			return entrypointMutationResult{}, err
		}
		receipt, createEntrypointErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_entrypoint", resourceID: entrypoint.ID, resourceRevision: 1,
			action: "routing.entrypoint.create", operation: "created",
		}, meta, false)
		if createEntrypointErr != nil {
			return entrypointMutationResult{}, createEntrypointErr
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_entrypoint", ResourceID: entrypoint.ID,
				ResourceRevision: 1, ResponseStatus: 201,
			}); err != nil {
				return entrypointMutationResult{}, err
			}
		}
		created, createEntrypointErr := loadEntrypointTx(ctx, tx, namespaceID, entrypoint.ID)
		return entrypointMutationResult{entrypoint: created, receipt: receipt}, createEntrypointErr
	})
	if err != nil {
		return routingmanagement.Entrypoint{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.entrypoint, value.receipt, nil
}

func insertEntrypointRevision(
	ctx context.Context, tx *sql.Tx, entrypoint routingsnapshot.Entrypoint, principalID string,
) error {
	digestValue := entrypoint
	digestValue.Revision = 0
	if _, err := tx.ExecContext(ctx, `INSERT INTO routing_entrypoint_revisions
  (entrypoint_id,revision,name,aliases,content_digest,created_by)
VALUES ($1,$2,$3,$4,$5,$6)`, entrypoint.ID, entrypoint.Revision, entrypoint.Name,
		mustJSON(entrypoint.Aliases), contentDigest(digestValue), actorValue(principalID)); err != nil {
		return classifyWriteError(err)
	}
	for ruleIndex, rule := range entrypoint.Rules {
		if _, err := tx.ExecContext(ctx, `INSERT INTO routing_entrypoint_rules
  (id,entrypoint_id,entrypoint_revision,name,ordinal,matchers,recipe_id,recipe_revision)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, rule.ID, entrypoint.ID, entrypoint.Revision, rule.Name,
			ruleIndex, mustJSON(rule.Matchers), rule.RecipeID, rule.RecipeRevision); err != nil {
			return classifyWriteError(err)
		}
		for decisionID, assignmentSet := range rule.Assignments {
			var fallbackStrategy, fallbackOn any
			if assignmentSet.Fallback != nil {
				fallbackStrategy = assignmentSet.Fallback.Strategy
				fallbackOn = mustJSON(assignmentSet.Fallback.On)
			}
			if _, err := tx.ExecContext(ctx, `INSERT INTO routing_decision_assignments
  (entrypoint_id,entrypoint_revision,rule_id,recipe_id,recipe_revision,decision_id,
   fallback_strategy,fallback_on)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8)`, entrypoint.ID, entrypoint.Revision, rule.ID,
				rule.RecipeID, rule.RecipeRevision, decisionID, fallbackStrategy, fallbackOn); err != nil {
				return classifyWriteError(err)
			}
			for assignmentIndex, assignment := range assignmentSet.Models {
				var reasoning any
				if assignment.Reasoning != nil {
					reasoning = mustJSON(assignment.Reasoning)
				}
				if _, err := tx.ExecContext(ctx, `INSERT INTO routing_assignment_models
  (entrypoint_id,entrypoint_revision,rule_id,decision_id,ordinal,
   model_id,model_revision,priority,weight,lora_name,reasoning)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)`, entrypoint.ID, entrypoint.Revision,
					rule.ID, decisionID, assignmentIndex, assignment.ModelID, assignment.ModelRevision,
					assignment.Priority, assignment.Weight,
					nullableString(assignment.LoRAName), reasoning); err != nil {
					return classifyWriteError(err)
				}
			}
		}
	}
	return nil
}

func (store *Store) UpdateEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, entrypoint routingsnapshot.Entrypoint,
	meta routingmanagement.MutationContext,
) (routingmanagement.Entrypoint, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (entrypointMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return entrypointMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayEntrypoint(ctx, tx, namespaceID, replay)
			}
		}
		updated, updateEntrypointErr := tx.ExecContext(ctx, `UPDATE routing_entrypoints SET name=$4,aliases=$5,
current_revision=$6,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`, namespaceID, id,
			expected, entrypoint.Name, mustJSON(entrypoint.Aliases), entrypoint.Revision)
		if updateEntrypointErr != nil {
			return entrypointMutationResult{}, classifyWriteError(updateEntrypointErr)
		}
		rows, _ := updated.RowsAffected()
		if rows != 1 {
			return entrypointMutationResult{}, routingmanagement.ErrConflict
		}
		if err := insertEntrypointRevision(ctx, tx, entrypoint, meta.PrincipalID); err != nil {
			return entrypointMutationResult{}, err
		}
		receipt, updateEntrypointErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_entrypoint", resourceID: id, resourceRevision: expected + 1,
			action: "routing.entrypoint.update", operation: "updated",
		}, meta, false)
		if updateEntrypointErr != nil {
			return entrypointMutationResult{}, updateEntrypointErr
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_entrypoint", ResourceID: id,
				ResourceRevision: uint64(expected + 1), ResponseStatus: 200,
			}); err != nil {
				return entrypointMutationResult{}, err
			}
		}
		current, updateEntrypointErr := loadEntrypointTx(ctx, tx, namespaceID, id)
		return entrypointMutationResult{entrypoint: current, receipt: receipt}, updateEntrypointErr
	})
	if err != nil {
		return routingmanagement.Entrypoint{}, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.entrypoint, value.receipt, nil
}

func (store *Store) DeleteEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, meta routingmanagement.MutationContext,
) (routingmanagement.RevisionReceipt, error) {
	return inTransaction(ctx, store, func(tx *sql.Tx) (routingmanagement.RevisionReceipt, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return routingmanagement.RevisionReceipt{}, classifyCommandError(err)
			}
			if found {
				return replayResourceReceipt(replay, "routing_entrypoint", id)
			}
		}
		result, err := tx.ExecContext(ctx, `UPDATE routing_entrypoints SET status='deleted',deleted_at=clock_timestamp(),
revision=revision+1,updated_at=clock_timestamp() WHERE namespace_id=$1 AND id=$2 AND revision=$3
AND status<>'active' AND deleted_at IS NULL`, namespaceID, id, expected)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, classifyWriteError(err)
		}
		rows, _ := result.RowsAffected()
		if rows != 1 {
			return routingmanagement.RevisionReceipt{}, routingmanagement.ErrConflict
		}
		receipt, err := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_entrypoint", resourceID: id, resourceRevision: expected + 1,
			action: "routing.entrypoint.delete", operation: "deleted",
		}, meta, false)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, err
		}
		if meta.Command != nil {
			if err := commandpostgres.CompleteResource(ctx, tx, *meta.Command, managementcommand.ResourceResult{
				ResourceType: "routing_entrypoint", ResourceID: id,
				ResourceRevision: uint64(expected + 1), ResponseStatus: 204,
			}); err != nil {
				return routingmanagement.RevisionReceipt{}, err
			}
		}
		return receipt, nil
	})
}

func (store *Store) PublishEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, meta routingmanagement.MutationContext,
) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	return store.changePublication(ctx, namespaceID, id, expected, true, meta)
}

func (store *Store) UnpublishEntrypoint(
	ctx context.Context, namespaceID, id string, expected int64, meta routingmanagement.MutationContext,
) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	return store.changePublication(ctx, namespaceID, id, expected, false, meta)
}

func (store *Store) changePublication(
	ctx context.Context, namespaceID, id string, expected int64, publish bool,
	meta routingmanagement.MutationContext,
) (*routingsnapshot.Snapshot, routingmanagement.RevisionReceipt, error) {
	value, err := inTransaction(ctx, store, func(tx *sql.Tx) (publicationMutationResult, error) {
		if meta.Command != nil {
			replay, found, err := commandpostgres.Lock(ctx, tx, *meta.Command)
			if err != nil {
				return publicationMutationResult{}, classifyCommandError(err)
			}
			if found {
				return replayPublication(replay)
			}
		}
		status, action := "draft", "routing.entrypoint.unpublish"
		if publish {
			status, action = "active", "routing.entrypoint.publish"
			if err := validatePublishableClosure(ctx, tx, namespaceID, id); err != nil {
				return publicationMutationResult{}, err
			}
		}
		query := `UPDATE routing_entrypoints SET status=$4,published_revision=NULL,
revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`
		if publish {
			query = `UPDATE routing_entrypoints SET status=$4,published_revision=current_revision,
revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND revision=$3 AND deleted_at IS NULL`
		}
		updated, changePublicationErr := tx.ExecContext(ctx, query, namespaceID, id, expected, status)
		if changePublicationErr != nil {
			return publicationMutationResult{}, classifyWriteError(changePublicationErr)
		}
		rows, _ := updated.RowsAffected()
		if rows != 1 {
			return publicationMutationResult{}, routingmanagement.ErrConflict
		}
		receipt, changePublicationErr := appendMutation(ctx, tx, namespaceID, mutationRecord{
			resourceType: "routing_entrypoint", resourceID: id, resourceRevision: expected + 1,
			action: action, operation: "updated",
		}, meta, true)
		if changePublicationErr != nil {
			return publicationMutationResult{}, changePublicationErr
		}
		var currency string
		if err := tx.QueryRowContext(ctx, `SELECT billing_currency FROM access_namespaces WHERE id=$1`, namespaceID).Scan(&currency); err != nil {
			return publicationMutationResult{}, err
		}
		bundle, changePublicationErr := LoadPublishedBundle(ctx, tx, namespaceID, currency, receipt.DesiredRevision)
		if changePublicationErr != nil {
			return publicationMutationResult{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, changePublicationErr)
		}
		snapshot, changePublicationErr := routingsnapshot.Compile(bundle)
		if changePublicationErr != nil {
			return publicationMutationResult{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, changePublicationErr)
		}
		if err := store.validatePublication(snapshot); err != nil {
			return publicationMutationResult{}, fmt.Errorf("%w: %w", routingmanagement.ErrPublication, err)
		}
		if meta.Command != nil {
			activeDigest := meta.Command.ActiveDigest()
			operationID := uuid.NewString()
			actorChain, _ := json.Marshal(meta.ActorChain)
			targetIDs, _ := json.Marshal([]string{id})
			targetScope, _ := json.Marshal(map[string]string{"entrypointId": id})
			if _, err := tx.ExecContext(ctx, `INSERT INTO management_operations
  (id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
   progress_total,target_scope,target_ids,desired_revision)
VALUES ($1,$2,$3,$4,$5,$6,'pending',1,$7,$8,$9)`, operationID, namespaceID,
				action, meta.PrincipalID, actorChain, activeDigest.RequestDigest[:], targetScope,
				targetIDs, receipt.DesiredRevision); err != nil {
				return publicationMutationResult{}, fmt.Errorf("insert routing publication operation: %w", err)
			}
			desired := uint64(receipt.DesiredRevision)
			if err := commandpostgres.CompleteOperation(ctx, tx, *meta.Command, managementcommand.OperationResult{
				OperationID: operationID, DesiredRevision: &desired, ResponseStatus: 202,
			}); err != nil {
				return publicationMutationResult{}, err
			}
			receipt.OperationID = operationID
		}
		return publicationMutationResult{snapshot: snapshot, receipt: receipt}, nil
	})
	if err != nil {
		return nil, routingmanagement.RevisionReceipt{}, classifyWriteError(err)
	}
	return value.snapshot, value.receipt, nil
}

type entrypointMutationResult struct {
	entrypoint routingmanagement.Entrypoint
	receipt    routingmanagement.RevisionReceipt
}

func replayEntrypoint(
	ctx context.Context, tx *sql.Tx, namespaceID string, stored managementcommand.StoredResult,
) (entrypointMutationResult, error) {
	if stored.Resource == nil || stored.Resource.ResourceType != "routing_entrypoint" {
		return entrypointMutationResult{}, managementcommand.ErrConflict
	}
	entrypoint, err := loadEntrypointTx(ctx, tx, namespaceID, stored.Resource.ResourceID)
	if err != nil {
		return entrypointMutationResult{}, managementcommand.ErrConflict
	}
	return entrypointMutationResult{
		entrypoint: entrypoint,
		receipt: routingmanagement.RevisionReceipt{
			ResourceRevision: int64(stored.Resource.ResourceRevision), Replayed: true,
		},
	}, nil
}

type publicationMutationResult struct {
	snapshot *routingsnapshot.Snapshot
	receipt  routingmanagement.RevisionReceipt
}

func replayPublication(stored managementcommand.StoredResult) (publicationMutationResult, error) {
	if stored.Operation == nil || stored.Operation.DesiredRevision == nil {
		return publicationMutationResult{}, managementcommand.ErrConflict
	}
	return publicationMutationResult{receipt: routingmanagement.RevisionReceipt{
		DesiredRevision: int64(*stored.Operation.DesiredRevision), OperationID: stored.Operation.OperationID,
		Replayed: true,
	}}, nil
}

func validatePublishableClosure(ctx context.Context, tx *sql.Tx, namespaceID, entrypointID string) error {
	var invalidRecipes, invalidModels, invalidCredentials int64
	if err := tx.QueryRowContext(ctx, `SELECT
  count(DISTINCT r.recipe_id) FILTER (WHERE p.id IS NULL OR p.status IN ('disabled','deleted')),
  count(DISTINCT a.model_id) FILTER (WHERE m.id IS NULL OR m.status IN ('disabled','deleted')),
  count(DISTINCT b.provider_credential_id) FILTER (
    WHERE b.provider_credential_id IS NOT NULL AND (c.id IS NULL OR c.status <> 'active')
  )
FROM routing_entrypoints e
LEFT JOIN routing_entrypoint_rules r
  ON r.entrypoint_id=e.id AND r.entrypoint_revision=e.current_revision
LEFT JOIN routing_recipes p ON p.namespace_id=e.namespace_id AND p.id=r.recipe_id
LEFT JOIN routing_decision_assignments da
  ON da.entrypoint_id=r.entrypoint_id AND da.entrypoint_revision=r.entrypoint_revision AND da.rule_id=r.id
LEFT JOIN routing_assignment_models a
  ON a.entrypoint_id=da.entrypoint_id AND a.entrypoint_revision=da.entrypoint_revision
  AND a.rule_id=da.rule_id AND a.decision_id=da.decision_id
LEFT JOIN routing_models m ON m.namespace_id=e.namespace_id AND m.id=a.model_id
LEFT JOIN routing_model_backends b ON b.model_id=a.model_id AND b.model_revision=a.model_revision
LEFT JOIN provider_credentials c
  ON c.namespace_id=e.namespace_id AND c.id=b.provider_credential_id
WHERE e.namespace_id=$1 AND e.id=$2 AND e.deleted_at IS NULL`, namespaceID, entrypointID).Scan(
		&invalidRecipes, &invalidModels, &invalidCredentials,
	); err != nil {
		return fmt.Errorf("validate published routing closure: %w", err)
	}
	if invalidRecipes != 0 || invalidModels != 0 || invalidCredentials != 0 {
		return fmt.Errorf("%w: Entrypoint references inactive Recipe, Model, or ProviderCredential", routingmanagement.ErrPublication)
	}
	return nil
}

func (store *Store) ActiveSnapshot(ctx context.Context, namespaceID string) (*routingsnapshot.Snapshot, error) {
	var payload []byte
	activeSnapshotErr := store.db.QueryRowContext(ctx, `SELECT compiled_blob FROM routing_snapshots
WHERE namespace_id=$1 AND status='active' ORDER BY routing_revision DESC LIMIT 1`, namespaceID).Scan(&payload)
	if errors.Is(activeSnapshotErr, sql.ErrNoRows) {
		return nil, routingmanagement.ErrNotFound
	}
	if activeSnapshotErr != nil {
		return nil, fmt.Errorf("read active routing snapshot: %w", activeSnapshotErr)
	}
	var stored routingsnapshot.Snapshot
	if err := json.Unmarshal(payload, &stored); err != nil {
		return nil, fmt.Errorf("%w: active routing snapshot is corrupt", routingmanagement.ErrPublication)
	}
	restored, activeSnapshotErr := routingsnapshot.Compile(stored.Bundle)
	if activeSnapshotErr != nil || restored.Digest != stored.Digest {
		return nil, fmt.Errorf("%w: active routing snapshot failed validation", routingmanagement.ErrPublication)
	}
	return restored, nil
}
