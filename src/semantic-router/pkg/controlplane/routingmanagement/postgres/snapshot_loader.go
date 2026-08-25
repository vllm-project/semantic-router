package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// LoadPublishedBundle materializes only the exact dependency closure rooted at
// active Entrypoint published revisions. Draft/current dependency revisions can
// change without altering this value.
func LoadPublishedBundle(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	currency string,
	desiredRevision int64,
) (routingsnapshot.Bundle, error) {
	if tx == nil || desiredRevision <= 0 {
		return routingsnapshot.Bundle{}, fmt.Errorf("published routing transaction and revision are required")
	}
	bundle := routingsnapshot.Bundle{NamespaceID: namespaceID, Revision: desiredRevision, Currency: currency}
	var err error
	bundle.Entrypoints, err = loadPublishedEntrypoints(ctx, tx, namespaceID)
	if err != nil {
		return routingsnapshot.Bundle{}, err
	}
	bundle.Recipes, err = loadPublishedRecipes(ctx, tx, namespaceID)
	if err != nil {
		return routingsnapshot.Bundle{}, err
	}
	bundle.Models, err = loadPublishedModels(ctx, tx, namespaceID)
	if err != nil {
		return routingsnapshot.Bundle{}, err
	}
	return bundle, nil
}

func loadPublishedEntrypoints(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
) ([]routingsnapshot.Entrypoint, error) {
	result, byID, err := loadPublishedEntrypointHeaders(ctx, tx, namespaceID)
	if err != nil {
		return nil, err
	}
	rules, err := loadPublishedEntrypointRules(ctx, tx, namespaceID, result, byID)
	if err != nil {
		return nil, err
	}
	if err := loadPublishedDecisionAssignments(ctx, tx, namespaceID, result, rules); err != nil {
		return nil, err
	}
	if err := loadPublishedAssignmentModels(ctx, tx, namespaceID, result, rules); err != nil {
		return nil, err
	}
	return result, nil
}

func loadPublishedEntrypointHeaders(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
) (_ []routingsnapshot.Entrypoint, _ map[string]int, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT e.id, e.published_revision, r.name, r.aliases
FROM routing_entrypoints e
JOIN routing_entrypoint_revisions r ON r.entrypoint_id = e.id AND r.revision = e.published_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY e.id`, namespaceID)
	if err != nil {
		return nil, nil, fmt.Errorf("list published Entrypoints: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, rows.Close()) }()
	result := make([]routingsnapshot.Entrypoint, 0)
	byID := make(map[string]int)
	for rows.Next() {
		var value routingsnapshot.Entrypoint
		var aliases []byte
		if err := rows.Scan(&value.ID, &value.Revision, &value.Name, &aliases); err != nil {
			return nil, nil, fmt.Errorf("scan published Entrypoint: %w", err)
		}
		if err := strictJSON(aliases, &value.Aliases); err != nil {
			return nil, nil, fmt.Errorf("decode published Entrypoint %s aliases: %w", value.ID, err)
		}
		result = append(result, value)
		byID[value.ID] = len(result) - 1
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}
	return result, byID, nil
}

type publishedRuleKey struct {
	entrypointID string
	ruleID       string
}

type publishedRuleLocation struct {
	entrypointIndex int
	ruleIndex       int
}

func loadPublishedEntrypointRules(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	entrypoints []routingsnapshot.Entrypoint,
	byID map[string]int,
) (_ map[publishedRuleKey]publishedRuleLocation, returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT r.entrypoint_id, r.id, r.name, r.matchers, r.recipe_id, r.recipe_revision
FROM routing_entrypoint_rules r
JOIN routing_entrypoints e ON e.id = r.entrypoint_id AND e.published_revision = r.entrypoint_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY r.entrypoint_id, r.ordinal`, namespaceID)
	if err != nil {
		return nil, fmt.Errorf("list published Entrypoint rules: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, rows.Close()) }()
	rules := make(map[publishedRuleKey]publishedRuleLocation)
	for rows.Next() {
		var entrypointID string
		var rule routingsnapshot.EntrypointRule
		var matchers []byte
		if err := rows.Scan(
			&entrypointID, &rule.ID, &rule.Name, &matchers, &rule.RecipeID, &rule.RecipeRevision,
		); err != nil {
			return nil, fmt.Errorf("scan published Entrypoint rule: %w", err)
		}
		if err := strictJSON(matchers, &rule.Matchers); err != nil {
			return nil, fmt.Errorf("decode published rule %s matchers: %w", rule.ID, err)
		}
		rule.Assignments = make(map[string]routingsnapshot.AssignmentSet)
		entrypointIndex, exists := byID[entrypointID]
		if !exists {
			return nil, fmt.Errorf("published rule references absent Entrypoint %s", entrypointID)
		}
		entrypoint := &entrypoints[entrypointIndex]
		entrypoint.Rules = append(entrypoint.Rules, rule)
		rules[publishedRuleKey{entrypointID: entrypointID, ruleID: rule.ID}] = publishedRuleLocation{
			entrypointIndex: entrypointIndex, ruleIndex: len(entrypoint.Rules) - 1,
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return rules, nil
}

func loadPublishedDecisionAssignments(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	entrypoints []routingsnapshot.Entrypoint,
	rules map[publishedRuleKey]publishedRuleLocation,
) (returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT a.entrypoint_id, a.rule_id, a.decision_id,
a.fallback_strategy, a.fallback_on
FROM routing_decision_assignments a
JOIN routing_entrypoints e ON e.id = a.entrypoint_id AND e.published_revision = a.entrypoint_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY a.entrypoint_id, a.rule_id, a.decision_id`, namespaceID)
	if err != nil {
		return fmt.Errorf("list published decision assignments: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, rows.Close()) }()
	for rows.Next() {
		var entrypointID, ruleID, decisionID string
		var strategy sql.NullString
		var on []byte
		if err := rows.Scan(&entrypointID, &ruleID, &decisionID, &strategy, &on); err != nil {
			return fmt.Errorf("scan published decision assignment: %w", err)
		}
		location, exists := rules[publishedRuleKey{entrypointID: entrypointID, ruleID: ruleID}]
		if !exists {
			return fmt.Errorf("decision assignment references absent published rule %s/%s", entrypointID, ruleID)
		}
		assignmentSet := routingsnapshot.AssignmentSet{}
		if strategy.Valid {
			assignmentSet.Fallback = &routingsnapshot.FallbackPolicy{Strategy: strategy.String}
			if err := strictJSON(on, &assignmentSet.Fallback.On); err != nil {
				return fmt.Errorf("decode published fallback policy: %w", err)
			}
		}
		entrypoints[location.entrypointIndex].Rules[location.ruleIndex].Assignments[decisionID] = assignmentSet
	}
	return rows.Err()
}

func loadPublishedAssignmentModels(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	entrypoints []routingsnapshot.Entrypoint,
	rules map[publishedRuleKey]publishedRuleLocation,
) (returnErr error) {
	rows, err := tx.QueryContext(ctx, `SELECT a.entrypoint_id, a.rule_id, a.decision_id, a.model_id,
a.model_revision, a.priority, a.weight::text, a.lora_name, a.reasoning
FROM routing_assignment_models a
JOIN routing_entrypoints e ON e.id = a.entrypoint_id AND e.published_revision = a.entrypoint_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY a.entrypoint_id, a.rule_id, a.decision_id, a.priority, a.ordinal`, namespaceID)
	if err != nil {
		return fmt.Errorf("list published assignments: %w", err)
	}
	defer func() { returnErr = errors.Join(returnErr, rows.Close()) }()
	for rows.Next() {
		var entrypointID, ruleID, decisionID string
		var assignment routingsnapshot.Assignment
		var lora sql.NullString
		var reasoning []byte
		if err := rows.Scan(
			&entrypointID, &ruleID, &decisionID, &assignment.ModelID,
			&assignment.ModelRevision, &assignment.Priority, &assignment.Weight, &lora, &reasoning,
		); err != nil {
			return fmt.Errorf("scan published assignment: %w", err)
		}
		if lora.Valid {
			assignment.LoRAName = lora.String
		}
		if len(reasoning) != 0 && string(reasoning) != "null" {
			var value routingsnapshot.AssignmentReasoning
			if err := strictJSON(reasoning, &value); err != nil {
				return fmt.Errorf("decode published assignment reasoning: %w", err)
			}
			assignment.Reasoning = &value
		}
		location, exists := rules[publishedRuleKey{entrypointID: entrypointID, ruleID: ruleID}]
		if !exists {
			return fmt.Errorf("assignment references absent published rule %s/%s", entrypointID, ruleID)
		}
		rule := &entrypoints[location.entrypointIndex].Rules[location.ruleIndex]
		assignmentSet, exists := rule.Assignments[decisionID]
		if !exists {
			return fmt.Errorf("model assignment references absent published decision assignment %s/%s/%s", entrypointID, ruleID, decisionID)
		}
		assignmentSet.Models = append(assignmentSet.Models, assignment)
		rule.Assignments[decisionID] = assignmentSet
	}
	return rows.Err()
}

func loadPublishedRecipes(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
) (_ []routingsnapshot.Recipe, returnErr error) {
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT DISTINCT rr.recipe_id, rr.revision, rr.name, rr.description, rr.document
FROM routing_entrypoints e
JOIN routing_entrypoint_rules er ON er.entrypoint_id = e.id AND er.entrypoint_revision = e.published_revision
JOIN routing_recipe_revisions rr ON rr.recipe_id = er.recipe_id AND rr.revision = er.recipe_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY rr.recipe_id`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list published Recipes: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]routingsnapshot.Recipe, 0)
	byID := make(map[string]int)
	for rows.Next() {
		var recipe routingsnapshot.Recipe
		if err := rows.Scan(&recipe.ID, &recipe.Revision, &recipe.Name, &recipe.Description, &recipe.Document); err != nil {
			return nil, fmt.Errorf("scan published Recipe: %w", err)
		}
		priorIndex, exists := byID[recipe.ID]
		if exists && result[priorIndex].Revision != recipe.Revision {
			return nil, fmt.Errorf("published Entrypoints pin conflicting revisions of Recipe %s", recipe.ID)
		}
		if !exists {
			result = append(result, recipe)
			byID[recipe.ID] = len(result) - 1
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	decisionRows, queryContextErr := tx.QueryContext(ctx, `SELECT DISTINCT d.recipe_id, d.recipe_revision, d.decision_id, d.name, d.dispatch_cardinality, d.ordinal
FROM routing_entrypoints e
JOIN routing_entrypoint_rules er ON er.entrypoint_id = e.id AND er.entrypoint_revision = e.published_revision
JOIN routing_recipe_decisions d ON d.recipe_id = er.recipe_id AND d.recipe_revision = er.recipe_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY d.recipe_id, d.ordinal`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list published Recipe decisions: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, decisionRows.Close())
	}()
	for decisionRows.Next() {
		var recipeID string
		var revision, ordinal int64
		var decision routingsnapshot.Decision
		if err := decisionRows.Scan(&recipeID, &revision, &decision.ID, &decision.Name, &decision.DispatchCardinality, &ordinal); err != nil {
			return nil, fmt.Errorf("scan published Recipe decision: %w", err)
		}
		recipeIndex, exists := byID[recipeID]
		if !exists || result[recipeIndex].Revision != revision {
			return nil, fmt.Errorf("decision references absent published Recipe %s@%d", recipeID, revision)
		}
		recipe := &result[recipeIndex]
		recipe.Decisions = append(recipe.Decisions, decision)
	}
	return result, decisionRows.Err()
}

func loadPublishedModels(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
) (_ []routingsnapshot.Model, returnErr error) {
	rows, queryContextErr := tx.QueryContext(ctx, `SELECT DISTINCT mr.model_id, mr.revision, mr.provider_catalog_revision,
mr.name, mr.aliases, mr.param_size, mr.context_window_size, mr.description,
mr.capabilities, mr.reasoning, mr.loras, mr.quality_score, mr.modality, mr.tags,
mr.execution, mr.pricing
FROM routing_entrypoints e
JOIN routing_assignment_models a ON a.entrypoint_id = e.id AND a.entrypoint_revision = e.published_revision
JOIN routing_model_revisions mr ON mr.model_id = a.model_id AND mr.revision = a.model_revision
JOIN routing_models m ON m.id = mr.model_id AND m.namespace_id = e.namespace_id
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY mr.model_id`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list published Models: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	result := make([]routingsnapshot.Model, 0)
	byID := make(map[string]int)
	for rows.Next() {
		var model routingsnapshot.Model
		var aliases, capabilities, reasoning, loras, tags, execution, pricing []byte
		if err := rows.Scan(&model.ID, &model.Revision, &model.CatalogRevision, &model.Name,
			&aliases, &model.ParamSize, &model.ContextWindowSize, &model.Description,
			&capabilities, &reasoning, &loras, &model.QualityScore, &model.Modality, &tags,
			&execution, &pricing); err != nil {
			return nil, fmt.Errorf("scan published Model: %w", err)
		}
		priorIndex, exists := byID[model.ID]
		if exists && result[priorIndex].Revision != model.Revision {
			return nil, fmt.Errorf("published Entrypoints pin conflicting revisions of Model %s", model.ID)
		}
		if exists {
			continue
		}
		if err := decodePublishedModelMetadata(
			&model, aliases, capabilities, reasoning, loras, tags, execution, pricing,
		); err != nil {
			return nil, fmt.Errorf("decode published Model %s: %w", model.ID, err)
		}
		result = append(result, model)
		byID[model.ID] = len(result) - 1
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	backendRows, queryContextErr := tx.QueryContext(ctx, `SELECT DISTINCT b.id, b.model_id, b.model_revision, b.provider_id,
b.wire_format, b.normalized_origin, b.provider_model_id, b.provider_credential_id,
b.connection, b.weight::text, b.ordinal
FROM routing_entrypoints e
JOIN routing_assignment_models a ON a.entrypoint_id = e.id AND a.entrypoint_revision = e.published_revision
JOIN routing_model_backends b ON b.model_id = a.model_id AND b.model_revision = a.model_revision
WHERE e.namespace_id = $1 AND e.status = 'active' AND e.deleted_at IS NULL
ORDER BY b.model_id, b.ordinal`, namespaceID)
	if queryContextErr != nil {
		return nil, fmt.Errorf("list published Model backends: %w", queryContextErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, backendRows.Close())
	}()
	for backendRows.Next() {
		var backend routingsnapshot.Backend
		var modelID string
		var modelRevision, ordinal int64
		var credential sql.NullString
		var connection []byte
		if err := backendRows.Scan(&backend.ID, &modelID, &modelRevision, &backend.ProviderID,
			&backend.WireFormat, &backend.Origin, &backend.ProviderModelID, &credential,
			&connection, &backend.Weight, &ordinal); err != nil {
			return nil, fmt.Errorf("scan published Model backend: %w", err)
		}
		if credential.Valid {
			backend.ProviderCredentialID = credential.String
		}
		if err := strictJSON(connection, &backend.Connection); err != nil {
			return nil, fmt.Errorf("decode published backend %s: %w", backend.ID, err)
		}
		modelIndex, exists := byID[modelID]
		if !exists || result[modelIndex].Revision != modelRevision {
			return nil, fmt.Errorf("backend references absent published Model %s@%d", modelID, modelRevision)
		}
		model := &result[modelIndex]
		model.Backends = append(model.Backends, backend)
	}
	return result, backendRows.Err()
}

func decodePublishedModelMetadata(model *routingsnapshot.Model, payloads ...[]byte) error {
	targets := []any{
		&model.Aliases,
		&model.Capabilities,
		&model.Reasoning,
		&model.LoRAs,
		&model.Tags,
		&model.Execution,
		&model.Pricing,
	}
	if len(payloads) != len(targets) {
		return fmt.Errorf("published Model metadata is incomplete")
	}
	for index, payload := range payloads {
		if err := strictJSON(payload, targets[index]); err != nil {
			return err
		}
	}
	return nil
}

func strictJSON(payload []byte, target any) error {
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return fmt.Errorf("JSON contains trailing values")
	}
	return nil
}
