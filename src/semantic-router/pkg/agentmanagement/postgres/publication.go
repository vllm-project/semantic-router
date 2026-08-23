package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
)

const publicationPlanSelect = `SELECT id::text,session_id::text,turn_id::text,recipe_id,
       recipe_content_revision,recipe_resource_revision,entrypoint_id,
       entrypoint_content_revision,entrypoint_resource_revision,catalog_revision,
       exact_diff,diagnostics,gate_results,plan_digest,status,expires_at,revision,
       COALESCE(committed_operation_id::text,''),created_at,updated_at
  FROM agent_publication_plans`

func (store *Store) CreatePublicationPlan(
	ctx context.Context, namespaceID string, value agentmanagement.PublicationPlan,
	_ agentmanagement.MutationContext,
) (agentmanagement.PublicationPlan, error) {
	if err := validatePublicationPlan(value); err != nil {
		return agentmanagement.PublicationPlan{}, err
	}
	canonical, err := canonicalPublicationPlan(value)
	if err != nil {
		return agentmanagement.PublicationPlan{}, err
	}
	digest := sha256.Sum256(canonical)
	if value.Digest != "" {
		expected, parseErr := parseDigest(value.Digest)
		if parseErr != nil || !equalDigest(expected, digest[:]) {
			return agentmanagement.PublicationPlan{}, agentmanagement.ErrInvalid
		}
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.PublicationPlan, error) {
		if _, err := tx.ExecContext(ctx, `INSERT INTO agent_publication_plans
  (id,namespace_id,session_id,turn_id,recipe_id,recipe_content_revision,
   recipe_resource_revision,entrypoint_id,entrypoint_content_revision,
   entrypoint_resource_revision,catalog_revision,exact_diff,diagnostics,gate_results,
   plan_digest,status,expires_at,revision)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,'ready',$16,1)`,
			value.ID, namespaceID, value.SessionID, value.TurnID, value.RecipeID,
			value.RecipeContentRevision, value.RecipeResourceRevision, value.EntrypointID,
			value.EntrypointContentRevision, value.EntrypointResourceRevision,
			value.CatalogRevision, value.ExactDiff, value.Diagnostics, value.GateResults,
			digest[:], value.ExpiresAt.UTC()); err != nil {
			return agentmanagement.PublicationPlan{}, classifyWriteError(err)
		}
		return scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, value.ID))
	})
}

func (store *Store) GetPublicationPlan(
	ctx context.Context, namespaceID, id string,
) (agentmanagement.PublicationPlan, error) {
	return scanPublicationPlan(store.db.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, id))
}

func (store *Store) GetPublicationModelIDs(
	ctx context.Context, namespaceID, planID string,
) (_ []string, returnErr error) {
	rows, err := store.db.QueryContext(ctx, `SELECT DISTINCT assignment.model_id
FROM agent_publication_plans plan
JOIN routing_assignment_models assignment
  ON assignment.entrypoint_id=plan.entrypoint_id
 AND assignment.entrypoint_revision=plan.entrypoint_content_revision
WHERE plan.namespace_id=$1 AND plan.id=$2
ORDER BY assignment.model_id`, namespaceID, planID)
	if err != nil {
		return nil, fmt.Errorf("list Agent publication Models: %w", err)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	models := make([]string, 0, 16)
	for rows.Next() {
		var modelID string
		if err := rows.Scan(&modelID); err != nil {
			return nil, err
		}
		models = append(models, modelID)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate Agent publication Models: %w", err)
	}
	return models, nil
}

func (store *Store) ReservePublicationCommit(
	ctx context.Context,
	namespaceID string,
	planID string,
	planDigest string,
	expectedRevision int64,
	mutation agentmanagement.MutationContext,
) (agentmanagement.PublicationCommitReservation, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(planID) != nil ||
		uuid.Validate(mutation.PrincipalID) != nil || expectedRevision < 1 {
		return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrInvalid
	}
	digest, err := parseDigest(planDigest)
	if err != nil {
		return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrApproval
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.PublicationCommitReservation, error) {
		// Turn-first locking is shared with cancellation so approval and cancel
		// have one linearization order and cannot deadlock across replicas.
		var sessionID, turnID string
		if err := tx.QueryRowContext(ctx, `SELECT session_id::text,turn_id::text
FROM agent_publication_plans WHERE namespace_id=$1 AND id=$2`, namespaceID, planID).Scan(
			&sessionID, &turnID,
		); err != nil {
			return agentmanagement.PublicationCommitReservation{}, mapNotFound(err)
		}
		var turnStatus string
		var cancelled bool
		if err := tx.QueryRowContext(ctx, `SELECT status,cancel_requested_at IS NOT NULL
FROM agent_turns WHERE namespace_id=$1 AND session_id=$2 AND id=$3 FOR UPDATE`,
			namespaceID, sessionID, turnID).Scan(&turnStatus, &cancelled); err != nil {
			return agentmanagement.PublicationCommitReservation{}, mapNotFound(err)
		}
		plan, reservePublicationCommitErr := scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2 FOR UPDATE`, namespaceID, planID))
		if reservePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitReservation{}, reservePublicationCommitErr
		}
		storedDigest, parseErr := parseDigest(plan.Digest)
		if parseErr != nil || subtle.ConstantTimeCompare(storedDigest, digest) != 1 {
			return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrApproval
		}
		if plan.Status == agentmanagement.PublicationCommitted {
			var desired sql.NullInt64
			if plan.OperationID == "" {
				return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrConflict
			}
			if err := tx.QueryRowContext(ctx, `SELECT desired_revision FROM management_operations
WHERE namespace_id=$1 AND id=$2`, namespaceID, plan.OperationID).Scan(&desired); err != nil {
				return agentmanagement.PublicationCommitReservation{}, mapNotFound(err)
			}
			if !desired.Valid || desired.Int64 < 1 {
				return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrConflict
			}
			return agentmanagement.PublicationCommitReservation{
				Plan: plan, PrincipalID: mutation.PrincipalID,
				OperationID: plan.OperationID, DesiredRevision: desired.Int64, Replayed: true,
			}, nil
		}
		if plan.Status == agentmanagement.PublicationPublishing {
			return agentmanagement.PublicationCommitReservation{
				Plan: plan, PrincipalID: mutation.PrincipalID, Replayed: true,
			}, nil
		}
		if plan.Status != agentmanagement.PublicationReady || plan.Revision != expectedRevision ||
			turnStatus != string(agentmanagement.TurnWaitingApproval) || cancelled {
			return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrApproval
		}
		if !time.Now().UTC().Before(plan.ExpiresAt.UTC()) {
			return agentmanagement.PublicationCommitReservation{}, agentmanagement.ErrApproval
		}
		if err := verifyPublicationRoots(ctx, tx, namespaceID, plan); err != nil {
			return agentmanagement.PublicationCommitReservation{}, err
		}
		result, reservePublicationCommitErr := tx.ExecContext(ctx, `UPDATE agent_publication_plans
SET status='publishing',committed_by=$3,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND status='ready' AND revision=$4`,
			namespaceID, planID, mutation.PrincipalID, expectedRevision)
		if reservePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitReservation{}, classifyWriteError(reservePublicationCommitErr)
		}
		if err := requireOneRow(result); err != nil {
			return agentmanagement.PublicationCommitReservation{}, err
		}
		plan, reservePublicationCommitErr = scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, planID))
		return agentmanagement.PublicationCommitReservation{
			Plan: plan, PrincipalID: mutation.PrincipalID,
		}, reservePublicationCommitErr
	})
}

func (store *Store) FinalizePublicationCommit(
	ctx context.Context,
	namespaceID string,
	planID string,
	operationID string,
	desiredRevision int64,
	committedAt time.Time,
) (agentmanagement.PublicationCommitResult, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(planID) != nil ||
		uuid.Validate(operationID) != nil || desiredRevision < 1 || committedAt.IsZero() {
		return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.PublicationCommitResult, error) {
		var sessionID, turnID string
		if err := tx.QueryRowContext(ctx, `SELECT session_id::text,turn_id::text
FROM agent_publication_plans WHERE namespace_id=$1 AND id=$2`, namespaceID, planID).Scan(
			&sessionID, &turnID,
		); err != nil {
			return agentmanagement.PublicationCommitResult{}, mapNotFound(err)
		}
		var turnStatus string
		var cancelled bool
		if err := tx.QueryRowContext(ctx, `SELECT status,cancel_requested_at IS NOT NULL
FROM agent_turns WHERE namespace_id=$1 AND session_id=$2 AND id=$3 FOR UPDATE`,
			namespaceID, sessionID, turnID).Scan(&turnStatus, &cancelled); err != nil {
			return agentmanagement.PublicationCommitResult{}, mapNotFound(err)
		}
		plan, finalizePublicationCommitErr := scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2 FOR UPDATE`, namespaceID, planID))
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, finalizePublicationCommitErr
		}
		if plan.Status == agentmanagement.PublicationCommitted {
			if plan.OperationID != operationID {
				return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrConflict
			}
			return agentmanagement.PublicationCommitResult{
				Plan: plan, DesiredRevision: desiredRevision, Replayed: true,
			}, nil
		}
		if plan.Status != agentmanagement.PublicationPublishing ||
			turnStatus != string(agentmanagement.TurnWaitingApproval) || cancelled {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrApproval
		}
		var storedDesired sql.NullInt64
		if err := tx.QueryRowContext(ctx, `SELECT desired_revision FROM management_operations
WHERE namespace_id=$1 AND id=$2`, namespaceID, operationID).Scan(&storedDesired); err != nil {
			return agentmanagement.PublicationCommitResult{}, mapNotFound(err)
		}
		if !storedDesired.Valid || storedDesired.Int64 != desiredRevision {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrConflict
		}
		approvalPayload, finalizePublicationCommitErr := json.Marshal(agentmanagement.ApprovalResultEvent{
			PlanID: planID, Status: "committed", OperationID: operationID,
		})
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
		}
		approvalEvent, finalizePublicationCommitErr := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
			Origin: "control", Type: agentmanagement.EventApprovalResult, Payload: approvalPayload,
		})
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, finalizePublicationCommitErr
		}
		terminalPayload, finalizePublicationCommitErr := json.Marshal(agentmanagement.TerminalEvent{Status: agentmanagement.TurnCompleted})
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
		}
		terminalEvent, finalizePublicationCommitErr := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
			Origin: "control", Type: agentmanagement.EventTerminal, Payload: terminalPayload,
		})
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, finalizePublicationCommitErr
		}
		turnResult, finalizePublicationCommitErr := tx.ExecContext(ctx, `UPDATE agent_turns
SET status='completed',completed_at=$4,revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND status='waiting_approval'
  AND cancel_requested_at IS NULL`, namespaceID, sessionID, turnID, committedAt.UTC())
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, classifyWriteError(finalizePublicationCommitErr)
		}
		if err := requireOneRow(turnResult); err != nil {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrApproval
		}
		planResult, finalizePublicationCommitErr := tx.ExecContext(ctx, `UPDATE agent_publication_plans
SET status='committed',committed_operation_id=$3,committed_at=$4,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND status='publishing'`,
			namespaceID, planID, operationID, committedAt.UTC())
		if finalizePublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, classifyWriteError(finalizePublicationCommitErr)
		}
		if err := requireOneRow(planResult); err != nil {
			return agentmanagement.PublicationCommitResult{}, err
		}
		plan, finalizePublicationCommitErr = scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, planID))
		return agentmanagement.PublicationCommitResult{
			Plan: plan, DesiredRevision: desiredRevision,
			ApprovalEvent: approvalEvent, TerminalEvent: terminalEvent,
		}, finalizePublicationCommitErr
	})
}

func (store *Store) FailPublicationCommit(
	ctx context.Context,
	namespaceID string,
	planID string,
	failureCode string,
	failedAt time.Time,
) (agentmanagement.PublicationCommitResult, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(planID) != nil ||
		failureCode == "" || failedAt.IsZero() {
		return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
	}
	return inTransaction(ctx, store, func(tx *sql.Tx) (agentmanagement.PublicationCommitResult, error) {
		var sessionID, turnID string
		if err := tx.QueryRowContext(ctx, `SELECT session_id::text,turn_id::text
FROM agent_publication_plans WHERE namespace_id=$1 AND id=$2`, namespaceID, planID).Scan(
			&sessionID, &turnID,
		); err != nil {
			return agentmanagement.PublicationCommitResult{}, mapNotFound(err)
		}
		var turnStatus string
		if err := tx.QueryRowContext(ctx, `SELECT status FROM agent_turns
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 FOR UPDATE`,
			namespaceID, sessionID, turnID).Scan(&turnStatus); err != nil {
			return agentmanagement.PublicationCommitResult{}, mapNotFound(err)
		}
		plan, failPublicationCommitErr := scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2 FOR UPDATE`, namespaceID, planID))
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, failPublicationCommitErr
		}
		if plan.Status == agentmanagement.PublicationFailed && turnStatus == string(agentmanagement.TurnFailed) {
			return agentmanagement.PublicationCommitResult{Plan: plan, Replayed: true}, nil
		}
		if plan.Status != agentmanagement.PublicationPublishing ||
			turnStatus != string(agentmanagement.TurnWaitingApproval) {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrApproval
		}
		failure := &agentmanagement.Failure{
			Code: failureCode, Message: "The approved publication could not be applied.", Retryable: false,
		}
		approvalPayload, failPublicationCommitErr := json.Marshal(agentmanagement.ApprovalResultEvent{
			PlanID: planID, Status: "failed",
		})
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
		}
		approvalEvent, failPublicationCommitErr := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
			Origin: "control", Type: agentmanagement.EventApprovalResult, Payload: approvalPayload,
		})
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, failPublicationCommitErr
		}
		terminalPayload, failPublicationCommitErr := json.Marshal(agentmanagement.TerminalEvent{
			Status: agentmanagement.TurnFailed, Error: failure,
		})
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, agentmanagement.ErrInvalid
		}
		terminalEvent, failPublicationCommitErr := appendEventTx(ctx, tx, agentmanagement.EventAppend{
			NamespaceID: namespaceID, SessionID: sessionID, TurnID: turnID,
			Origin: "control", Type: agentmanagement.EventTerminal, Payload: terminalPayload,
		})
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, failPublicationCommitErr
		}
		turnResult, failPublicationCommitErr := tx.ExecContext(ctx, `UPDATE agent_turns
SET status='failed',completed_at=$4,failure_code=$5,failure_message=$6,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND session_id=$2 AND id=$3 AND status='waiting_approval'`,
			namespaceID, sessionID, turnID, failedAt.UTC(), failure.Code, failure.Message)
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, classifyWriteError(failPublicationCommitErr)
		}
		if err := requireOneRow(turnResult); err != nil {
			return agentmanagement.PublicationCommitResult{}, err
		}
		planResult, failPublicationCommitErr := tx.ExecContext(ctx, `UPDATE agent_publication_plans
SET status='failed',failure_code=$3,committed_by=NULL,
    revision=revision+1,updated_at=clock_timestamp()
WHERE namespace_id=$1 AND id=$2 AND status='publishing'`, namespaceID, planID, failureCode)
		if failPublicationCommitErr != nil {
			return agentmanagement.PublicationCommitResult{}, classifyWriteError(failPublicationCommitErr)
		}
		if err := requireOneRow(planResult); err != nil {
			return agentmanagement.PublicationCommitResult{}, err
		}
		plan, failPublicationCommitErr = scanPublicationPlan(tx.QueryRowContext(ctx, publicationPlanSelect+`
 WHERE namespace_id=$1 AND id=$2`, namespaceID, planID))
		return agentmanagement.PublicationCommitResult{
			Plan: plan, ApprovalEvent: approvalEvent, TerminalEvent: terminalEvent,
		}, failPublicationCommitErr
	})
}

func verifyPublicationRoots(
	ctx context.Context, tx *sql.Tx, namespaceID string, plan agentmanagement.PublicationPlan,
) error {
	var recipeContent, recipeResource, entrypointContent, entrypointResource int64
	var recipeStatus, entrypointStatus string
	err := tx.QueryRowContext(ctx, `SELECT recipe.current_revision,recipe.revision,recipe.status,
       entrypoint.current_revision,entrypoint.revision,entrypoint.status
FROM routing_recipes recipe
JOIN routing_entrypoints entrypoint ON entrypoint.namespace_id=recipe.namespace_id
WHERE recipe.namespace_id=$1 AND recipe.id=$2 AND entrypoint.id=$3
FOR KEY SHARE`, namespaceID, plan.RecipeID, plan.EntrypointID).Scan(
		&recipeContent, &recipeResource, &recipeStatus,
		&entrypointContent, &entrypointResource, &entrypointStatus,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return agentmanagement.ErrApproval
	}
	if err != nil {
		return fmt.Errorf("verify Agent publication roots: %w", err)
	}
	if recipeContent != plan.RecipeContentRevision || recipeResource != plan.RecipeResourceRevision ||
		entrypointContent != plan.EntrypointContentRevision || entrypointResource != plan.EntrypointResourceRevision ||
		(recipeStatus != "draft" && recipeStatus != "active") ||
		(entrypointStatus != "draft" && entrypointStatus != "active") {
		return agentmanagement.ErrConflict
	}
	return nil
}

func validatePublicationPlan(value agentmanagement.PublicationPlan) error {
	if value.RecipeID == "" || value.RecipeContentRevision < 1 || value.RecipeResourceRevision < 1 ||
		value.EntrypointID == "" || value.EntrypointContentRevision < 1 || value.EntrypointResourceRevision < 1 ||
		value.CatalogRevision == "" || value.ExpiresAt.IsZero() {
		return agentmanagement.ErrInvalid
	}
	var diff map[string]any
	var diagnostics, gates []any
	if json.Unmarshal(value.ExactDiff, &diff) != nil || diff == nil ||
		json.Unmarshal(value.Diagnostics, &diagnostics) != nil ||
		json.Unmarshal(value.GateResults, &gates) != nil {
		return agentmanagement.ErrInvalid
	}
	return nil
}

func canonicalPublicationPlan(value agentmanagement.PublicationPlan) ([]byte, error) {
	return json.Marshal(struct {
		SessionID                  string          `json:"sessionId"`
		TurnID                     string          `json:"turnId"`
		RecipeID                   string          `json:"recipeId"`
		RecipeContentRevision      int64           `json:"recipeContentRevision"`
		RecipeResourceRevision     int64           `json:"recipeResourceRevision"`
		EntrypointID               string          `json:"entrypointId"`
		EntrypointContentRevision  int64           `json:"entrypointContentRevision"`
		EntrypointResourceRevision int64           `json:"entrypointResourceRevision"`
		CatalogRevision            string          `json:"catalogRevision"`
		ExactDiff                  json.RawMessage `json:"exactDiff"`
		Diagnostics                json.RawMessage `json:"diagnostics"`
		GateResults                json.RawMessage `json:"gateResults"`
	}{
		value.SessionID, value.TurnID, value.RecipeID, value.RecipeContentRevision,
		value.RecipeResourceRevision, value.EntrypointID, value.EntrypointContentRevision,
		value.EntrypointResourceRevision, value.CatalogRevision, value.ExactDiff,
		value.Diagnostics, value.GateResults,
	})
}
