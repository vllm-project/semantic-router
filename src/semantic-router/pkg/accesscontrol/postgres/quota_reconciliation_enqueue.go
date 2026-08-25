package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	managementcommandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

type preparedUnknownUsageReconciliation struct {
	fence            quotareconciliation.Fence
	reconciliationID string
	previousRevision int64
	digest           []byte
	payload          []byte
}

func prepareUnknownUsageInTransaction(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	now time.Time,
) (quotareconciliation.EnqueueResult, error) {
	stored, replayed, err := managementcommandpostgres.Lock(ctx, tx, command)
	if err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	if replayed {
		return replayUnknownUsageReconciliation(ctx, tx, request.NamespaceID, stored)
	}
	prepared, err := prepareUnknownUsagePlan(ctx, tx, request, operationID, now)
	if err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	if err := persistUnknownUsagePlan(ctx, tx, command, request, operationID, prepared, now); err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	return newReconciliationEnqueueResult(request, operationID, prepared.fence.ID, now), nil
}

func replayUnknownUsageReconciliation(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	stored managementcommand.StoredResult,
) (quotareconciliation.EnqueueResult, error) {
	if stored.Operation == nil {
		return quotareconciliation.EnqueueResult{}, errors.New("unknown-usage command replay is not an operation")
	}
	operation, err := loadQuotaReconciliationOperation(ctx, tx, namespaceID, stored.Operation.OperationID)
	if err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	return quotareconciliation.EnqueueResult{Operation: operation, Replayed: true}, nil
}

func prepareUnknownUsagePlan(
	ctx context.Context,
	tx *sql.Tx,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	now time.Time,
) (preparedUnknownUsageReconciliation, error) {
	revision, err := lockOpenUnknownUsageFence(ctx, tx, request)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, err
	}
	fence, err := loadQuotaFence(ctx, tx, request.NamespaceID, request.FenceID)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, err
	}
	ledger, err := loadReconciliationLedger(ctx, tx, fence)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, err
	}
	reconciliationID := uuid.NewString()
	plan, err := buildReconciliationPlan(
		request, fence, ledger, operationID, reconciliationID, uuid.NewString(), now,
	)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, err
	}
	digestText, payload, err := quotareconciliation.DigestPlan(plan)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, err
	}
	digest, err := hex.DecodeString(digestText)
	if err != nil {
		return preparedUnknownUsageReconciliation{}, fmt.Errorf("decode reconciliation plan digest: %w", err)
	}
	return preparedUnknownUsageReconciliation{
		fence: fence, reconciliationID: reconciliationID,
		previousRevision: revision, digest: digest, payload: payload,
	}, nil
}

func lockOpenUnknownUsageFence(
	ctx context.Context,
	tx *sql.Tx,
	request quotareconciliation.ReconcileRequest,
) (int64, error) {
	var state quotareconciliation.FenceState
	var revision int64
	err := tx.QueryRowContext(ctx, `SELECT state, etag_revision
FROM unknown_usage_fences WHERE namespace_id=$1 AND id=$2 FOR UPDATE`,
		request.NamespaceID, request.FenceID,
	).Scan(&state, &revision)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, quotareconciliation.ErrNotFound
	}
	if err != nil {
		return 0, fmt.Errorf("lock unknown-usage fence: %w", err)
	}
	switch state {
	case quotareconciliation.FenceResolved:
		return 0, quotareconciliation.ErrResolved
	case quotareconciliation.FenceReconciling:
		return 0, quotareconciliation.ErrReconciliationConflict
	}
	if revision <= 0 || uint64(revision) != request.ExpectedRevision {
		return 0, quotareconciliation.ErrRevisionConflict
	}
	return revision, nil
}

func persistUnknownUsagePlan(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	prepared preparedUnknownUsageReconciliation,
	now time.Time,
) error {
	if err := insertUnknownUsageOperation(ctx, tx, command, request, operationID, prepared.fence, now); err != nil {
		return err
	}
	if err := insertUnknownUsagePlan(ctx, tx, request, operationID, prepared, now); err != nil {
		return err
	}
	newRevision, revisionErr := markUnknownUsageFenceReconciling(ctx, tx, request, prepared, now)
	if revisionErr != nil {
		return revisionErr
	}
	if completionErr := managementcommandpostgres.CompleteOperation(ctx, tx, command, managementcommand.OperationResult{
		OperationID: operationID, ResponseStatus: 202,
	}); completionErr != nil {
		return completionErr
	}
	revisionValue, conversionErr := positiveUint64(newRevision, "unknown-usage fence revision")
	if conversionErr != nil {
		return conversionErr
	}
	return appendQuotaReconciliationAudit(
		ctx, tx, request, prepared.fence.ID, revisionValue,
		"quota.unknown_usage_fence.reconcile_requested", "Reconcile unknown usage.",
		map[string]string{
			"strategy": string(request.Strategy), "reconciliation_id": prepared.reconciliationID,
		},
	)
}

func insertUnknownUsageOperation(
	ctx context.Context,
	tx *sql.Tx,
	command managementcommand.Command,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	fence quotareconciliation.Fence,
	now time.Time,
) error {
	actorChain, err := json.Marshal(request.Actor.ActorChain)
	if err != nil {
		return fmt.Errorf("encode reconciliation actor chain: %w", err)
	}
	targetScope, err := json.Marshal(struct {
		Version    int      `json:"version"`
		FenceID    string   `json:"fenceId"`
		BindingIDs []string `json:"bindingIds"`
	}{1, fence.ID, fenceBindingIDs(fence.Bindings)})
	if err != nil {
		return fmt.Errorf("encode reconciliation target scope: %w", err)
	}
	targetIDs, err := json.Marshal([]string{fence.ID})
	if err != nil {
		return fmt.Errorf("encode reconciliation target ids: %w", err)
	}
	active := command.ActiveDigest()
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_operations (
  id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
  progress_completed,progress_total,target_scope,target_ids,created_at,updated_at
) VALUES ($1,$2,$3,$4,$5,$6,'pending',0,1,$7,$8,$9,$9)`,
		operationID, request.NamespaceID, quotareconciliation.OperationKind,
		request.Actor.PrincipalID, actorChain, active.RequestDigest[:], targetScope, targetIDs, now,
	); err != nil {
		return fmt.Errorf("insert unknown-usage reconciliation operation: %w", err)
	}
	return nil
}

func insertUnknownUsagePlan(
	ctx context.Context,
	tx *sql.Tx,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	prepared preparedUnknownUsageReconciliation,
	now time.Time,
) error {
	if _, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_reconciliation_plans (
  reconciliation_id,namespace_id,fence_id,operation_id,strategy,plan_digest,plan_payload,
  phase,available_at,created_at,updated_at
) VALUES ($1,$2,$3,$4,$5,$6,$7,'runtime_pending',$8,$8,$8)`,
		prepared.reconciliationID, request.NamespaceID, prepared.fence.ID, operationID,
		request.Strategy, prepared.digest, prepared.payload, now,
	); err != nil {
		return fmt.Errorf("insert unknown-usage reconciliation plan: %w", err)
	}
	return nil
}

func markUnknownUsageFenceReconciling(
	ctx context.Context,
	tx *sql.Tx,
	request quotareconciliation.ReconcileRequest,
	prepared preparedUnknownUsageReconciliation,
	now time.Time,
) (int64, error) {
	newRevision := prepared.previousRevision + 1
	result, err := tx.ExecContext(ctx, `UPDATE unknown_usage_fences
SET state='reconciling',etag_revision=$3,reconciliation_id=$4,
  reconciliation_strategy=$5,reconciliation_actor_id=$6,reconciliation_reason=$7,
  updated_at=$8
WHERE namespace_id=$1 AND id=$2 AND state='open' AND etag_revision=$9`,
		request.NamespaceID, prepared.fence.ID, newRevision, prepared.reconciliationID,
		request.Strategy, request.Actor.PrincipalID, request.Reason, now, prepared.previousRevision,
	)
	if err != nil {
		return 0, fmt.Errorf("mark unknown-usage fence reconciling: %w", err)
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return 0, quotareconciliation.ErrRevisionConflict
	}
	return newRevision, nil
}

func newReconciliationEnqueueResult(
	request quotareconciliation.ReconcileRequest,
	operationID string,
	fenceID string,
	now time.Time,
) quotareconciliation.EnqueueResult {
	return quotareconciliation.EnqueueResult{Operation: quotareconciliation.Operation{
		ID: operationID, NamespaceID: request.NamespaceID, FenceID: fenceID,
		Kind: quotareconciliation.OperationKind, OriginPrincipalID: request.Actor.PrincipalID,
		ActorChain: append([]string(nil), request.Actor.ActorChain...), Version: 1,
		State: quotareconciliation.OperationPending,
		Total: 1, CreatedAt: now, UpdatedAt: now,
	}}
}
