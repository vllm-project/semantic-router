package postgres

import (
	"bytes"
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

func (s *Store) Claim(
	ctx context.Context,
	workerID string,
	now time.Time,
	lease time.Duration,
) (quotareconciliation.Claim, bool, error) {
	if s == nil || s.db == nil || workerID == "" || now.IsZero() || lease <= 0 {
		return quotareconciliation.Claim{}, false, quotareconciliation.ErrInvalidRequest
	}
	tx, claimErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if claimErr != nil {
		return quotareconciliation.Claim{}, false, claimErr
	}
	defer func() { _ = tx.Rollback() }()
	leaseToken := uuid.NewString()
	var claim quotareconciliation.Claim
	var payload, digest []byte
	claimErr = tx.QueryRowContext(ctx, `WITH candidate AS (
  SELECT reconciliation_id
  FROM unknown_usage_reconciliation_plans
  WHERE phase <> 'completed' AND available_at <= $1
    AND (lease_expires_at IS NULL OR lease_expires_at <= $1)
  ORDER BY available_at,created_at,reconciliation_id
  FOR UPDATE SKIP LOCKED LIMIT 1
)
UPDATE unknown_usage_reconciliation_plans plan
SET lease_owner=$2,lease_token=$3,lease_expires_at=$1+($4 * interval '1 millisecond'),
    attempt_count=attempt_count+1,last_error=NULL,updated_at=$1
FROM candidate WHERE plan.reconciliation_id=candidate.reconciliation_id
RETURNING plan.plan_payload,plan.plan_digest,plan.phase,
  COALESCE(plan.runtime_stream_id,''),plan.attempt_count,
  plan.lease_owner,plan.lease_token::text,plan.lease_expires_at`,
		now, workerID, leaseToken, lease.Milliseconds()).Scan(
		&payload, &digest, &claim.Phase, &claim.RuntimeStreamID, &claim.Attempt,
		&claim.LeaseOwner, &claim.LeaseToken, &claim.LeaseExpiresAt)
	if errors.Is(claimErr, sql.ErrNoRows) {
		return quotareconciliation.Claim{}, false, nil
	}
	if claimErr != nil {
		return quotareconciliation.Claim{}, false, fmt.Errorf("claim unknown-usage reconciliation: %w", claimErr)
	}
	if err := decodeReconciliationPlan(payload, &claim.Plan); err != nil {
		return quotareconciliation.Claim{}, false, err
	}
	computed, _, claimErr := quotareconciliation.DigestPlan(claim.Plan)
	if claimErr != nil || len(digest) != sha256.Size || computed != hex.EncodeToString(digest) {
		return quotareconciliation.Claim{}, false, errors.New("stored unknown-usage reconciliation plan digest is invalid")
	}
	claim.PlanDigest = computed
	if claim.Plan.ReconciliationID == "" || claim.LeaseToken != leaseToken || claim.LeaseOwner != workerID {
		return quotareconciliation.Claim{}, false, errors.New("stored unknown-usage reconciliation claim is invalid")
	}
	if _, err := tx.ExecContext(ctx, `UPDATE management_operations
SET state='running',updated_at=$2
WHERE id=$1 AND state='pending'`, claim.Plan.OperationID, now); err != nil {
		return quotareconciliation.Claim{}, false, err
	}
	if err := tx.Commit(); err != nil {
		return quotareconciliation.Claim{}, false, err
	}
	return claim, true, nil
}

func (s *Store) MarkRuntimeApplied(
	ctx context.Context,
	claim quotareconciliation.Claim,
	streamID string,
	now time.Time,
) error {
	if streamID == "" || now.IsZero() {
		return quotareconciliation.ErrInvalidRequest
	}
	result, err := s.db.ExecContext(ctx, `UPDATE unknown_usage_reconciliation_plans
SET phase='runtime_applied',runtime_stream_id=$4,updated_at=$5
WHERE reconciliation_id=$1 AND phase='runtime_pending'
  AND lease_owner=$2 AND lease_token=$3 AND lease_expires_at>$5`,
		claim.Plan.ReconciliationID, claim.LeaseOwner, claim.LeaseToken, streamID, now)
	if err != nil {
		return err
	}
	if count, _ := result.RowsAffected(); count == 1 {
		return nil
	}
	var phase quotareconciliation.Phase
	var storedStream string
	err = s.db.QueryRowContext(ctx, `SELECT phase,COALESCE(runtime_stream_id,'')
FROM unknown_usage_reconciliation_plans WHERE reconciliation_id=$1`, claim.Plan.ReconciliationID).
		Scan(&phase, &storedStream)
	if err == nil && phase != quotareconciliation.PhaseRuntimePending && storedStream == streamID {
		return nil
	}
	return quotareconciliation.ErrLeaseLost
}

func (s *Store) SettleLedger(ctx context.Context, claim quotareconciliation.Claim, now time.Time) error {
	if s == nil || s.db == nil || now.IsZero() {
		return quotareconciliation.ErrInvalidRequest
	}
	tx, settleLedgerErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if settleLedgerErr != nil {
		return settleLedgerErr
	}
	defer func() { _ = tx.Rollback() }()
	phase, settleLedgerErr := lockReconciliationClaim(ctx, tx, claim, now)
	if settleLedgerErr != nil {
		return settleLedgerErr
	}
	if phase == quotareconciliation.PhaseLedgerApplied || phase == quotareconciliation.PhaseCompleted {
		return tx.Commit()
	}
	if phase != quotareconciliation.PhaseRuntimeApplied {
		return quotareconciliation.ErrLeaseLost
	}
	state := "settled"
	if claim.Plan.Strategy == quotareconciliation.StrategyWaive {
		state = "waived"
	}
	digest, settleLedgerErr := hex.DecodeString(claim.PlanDigest)
	if settleLedgerErr != nil {
		return settleLedgerErr
	}
	result, settleLedgerErr := tx.ExecContext(ctx, `UPDATE usage_settlements
SET state=$3,canonical_usage_digest=$4,reconciliation_id=$5,revision=revision+1,settled_at=$6
WHERE namespace_id=$1 AND admission_id=$2 AND state='unknown' AND reconciliation_id IS NULL`,
		claim.Plan.NamespaceID, claim.Plan.AdmissionID, state, digest,
		claim.Plan.ReconciliationID, now)
	if settleLedgerErr != nil {
		return fmt.Errorf("settle unknown usage: %w", settleLedgerErr)
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return quotareconciliation.ErrReconciliationConflict
	}
	if err := insertReconciliationUsage(ctx, tx, claim.Plan, now); err != nil {
		return err
	}
	result, settleLedgerErr = tx.ExecContext(ctx, `UPDATE unknown_usage_reconciliation_plans
SET phase='ledger_applied',updated_at=$4
WHERE reconciliation_id=$1 AND phase='runtime_applied'
  AND lease_owner=$2 AND lease_token=$3 AND lease_expires_at>$4`,
		claim.Plan.ReconciliationID, claim.LeaseOwner, claim.LeaseToken, now)
	if settleLedgerErr != nil {
		return settleLedgerErr
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return quotareconciliation.ErrLeaseLost
	}
	if err := appendReconciliationPlanAudit(ctx, tx, claim.Plan,
		"quota.unknown_usage_fence.ledger_applied", "Apply unknown usage correction.", now); err != nil {
		return err
	}
	return tx.Commit()
}

func (s *Store) Complete(
	ctx context.Context,
	claim quotareconciliation.Claim,
	now time.Time,
) (quotareconciliation.Operation, error) {
	if s == nil || s.db == nil || now.IsZero() {
		return quotareconciliation.Operation{}, quotareconciliation.ErrInvalidRequest
	}
	tx, completeErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if completeErr != nil {
		return quotareconciliation.Operation{}, completeErr
	}
	defer func() { _ = tx.Rollback() }()
	phase, completeErr := lockReconciliationClaim(ctx, tx, claim, now)
	if completeErr != nil {
		return quotareconciliation.Operation{}, completeErr
	}
	if phase == quotareconciliation.PhaseCompleted {
		operation, err := loadQuotaReconciliationOperation(ctx, tx, claim.Plan.NamespaceID, claim.Plan.OperationID)
		if err != nil {
			return quotareconciliation.Operation{}, err
		}
		return operation, tx.Commit()
	}
	if phase != quotareconciliation.PhaseLedgerApplied {
		return quotareconciliation.Operation{}, quotareconciliation.ErrLeaseLost
	}
	var revision int64
	completeErr = tx.QueryRowContext(ctx, `UPDATE unknown_usage_fences
SET state='resolved',etag_revision=etag_revision+1,updated_at=$4,resolved_at=$4
WHERE namespace_id=$1 AND id=$2 AND state='reconciling' AND reconciliation_id=$3
RETURNING etag_revision`, claim.Plan.NamespaceID, claim.Plan.FenceID,
		claim.Plan.ReconciliationID, now).Scan(&revision)
	if errors.Is(completeErr, sql.ErrNoRows) {
		return quotareconciliation.Operation{}, quotareconciliation.ErrReconciliationConflict
	}
	if completeErr != nil {
		return quotareconciliation.Operation{}, completeErr
	}
	result, completeErr := tx.ExecContext(ctx, `UPDATE unknown_usage_reconciliation_plans
SET phase='completed',completed_at=$4,updated_at=$4,
  lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL
WHERE reconciliation_id=$1 AND phase='ledger_applied'
  AND lease_owner=$2 AND lease_token=$3 AND lease_expires_at>$4`,
		claim.Plan.ReconciliationID, claim.LeaseOwner, claim.LeaseToken, now)
	if completeErr != nil {
		return quotareconciliation.Operation{}, completeErr
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return quotareconciliation.Operation{}, quotareconciliation.ErrLeaseLost
	}
	if _, err := tx.ExecContext(ctx, `UPDATE management_operations
SET state='succeeded',progress_completed=progress_total,updated_at=$2
WHERE id=$1 AND state IN ('pending','running')`, claim.Plan.OperationID, now); err != nil {
		return quotareconciliation.Operation{}, err
	}
	if err := appendReconciliationPlanAudit(ctx, tx, claim.Plan,
		"quota.unknown_usage_fence.resolved", "Resolve unknown usage fence.", now); err != nil {
		return quotareconciliation.Operation{}, err
	}
	operation, completeErr := loadQuotaReconciliationOperation(ctx, tx, claim.Plan.NamespaceID, claim.Plan.OperationID)
	if completeErr != nil {
		return quotareconciliation.Operation{}, completeErr
	}
	if err := tx.Commit(); err != nil {
		return quotareconciliation.Operation{}, err
	}
	_ = revision
	return operation, nil
}

func (s *Store) Release(
	ctx context.Context,
	claim quotareconciliation.Claim,
	now time.Time,
	_ error,
) error {
	if s == nil || s.db == nil || now.IsZero() {
		return quotareconciliation.ErrInvalidRequest
	}
	backoff := time.Duration(claim.Attempt) * 250 * time.Millisecond
	if backoff < 250*time.Millisecond {
		backoff = 250 * time.Millisecond
	}
	if backoff > 30*time.Second {
		backoff = 30 * time.Second
	}
	result, err := s.db.ExecContext(ctx, `UPDATE unknown_usage_reconciliation_plans
SET lease_owner=NULL,lease_token=NULL,lease_expires_at=NULL,
  available_at=$4,last_error='retryable_reconciliation_failure',updated_at=$3
WHERE reconciliation_id=$1 AND lease_owner=$2 AND lease_token=$5`,
		claim.Plan.ReconciliationID, claim.LeaseOwner, now, now.Add(backoff), claim.LeaseToken)
	if err != nil {
		return err
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return quotareconciliation.ErrLeaseLost
	}
	return nil
}

func lockReconciliationClaim(
	ctx context.Context,
	tx *sql.Tx,
	claim quotareconciliation.Claim,
	now time.Time,
) (quotareconciliation.Phase, error) {
	var phase quotareconciliation.Phase
	var digest []byte
	err := tx.QueryRowContext(ctx, `SELECT phase,plan_digest
FROM unknown_usage_reconciliation_plans
WHERE reconciliation_id=$1 AND lease_owner=$2 AND lease_token=$3 AND lease_expires_at>$4
FOR UPDATE`, claim.Plan.ReconciliationID, claim.LeaseOwner, claim.LeaseToken, now).
		Scan(&phase, &digest)
	if errors.Is(err, sql.ErrNoRows) {
		return "", quotareconciliation.ErrLeaseLost
	}
	if err != nil {
		return "", err
	}
	if hex.EncodeToString(digest) != claim.PlanDigest {
		return "", quotareconciliation.ErrReconciliationConflict
	}
	return phase, nil
}

func insertReconciliationUsage(
	ctx context.Context,
	tx *sql.Tx,
	plan quotareconciliation.Plan,
	now time.Time,
) error {
	input, output, total, err := reconciliationEventQuantities(plan)
	if err != nil {
		return err
	}
	costs, err := reconciliationCosts(plan)
	if err != nil {
		return err
	}
	costPayload, err := json.Marshal(costs)
	if err != nil {
		return err
	}
	metadata, err := json.Marshal(struct {
		CompletedAt        time.Time `json:"completedAt"`
		Reason             string    `json:"reason"`
		EvidenceReferences []string  `json:"evidenceReferences"`
	}{plan.RequestSnapshot.CompletedAt, plan.Reason, plan.EvidenceReferences})
	if err != nil {
		return err
	}
	eventKind, usageState := "correction", "known_zero"
	switch plan.Strategy {
	case quotareconciliation.StrategyWaive:
		eventKind = "waiver"
	case quotareconciliation.StrategyActual:
		usageState = "known_actual"
	}
	unknownDelta := "-" + plan.UnknownDispatchCount
	eventDate := plan.RequestSnapshot.OccurredAt.UTC().Format("2006-01-02")
	_, err = tx.ExecContext(ctx, `INSERT INTO usage_events (
  namespace_id,admission_id,event_date,event_id,event_kind,external_request_id,
  protocol,path,api_key_id,credential_id,user_id,team_id,entrypoint_id,entrypoint_rule_id,
  recipe_id,routing_revision,status_code,error_code,input_tokens,output_tokens,total_tokens,
  served_input_tokens,served_output_tokens,served_total_tokens,latency_ms,usage_state,costs,
  request_metadata,occurred_at,ingested_at,reconciliation_id,reconciliation_strategy,
  corrects_event_id,incomplete_dispatch_delta
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
  $19,$20,$21,$22,$23,$24,0,$25,$26,$27,$28,$29,$30,$31,$32,$33)`,
		plan.NamespaceID, plan.AdmissionID, eventDate, plan.CorrectionEventID, eventKind,
		nullableString(plan.RequestSnapshot.ExternalRequestID), plan.RequestSnapshot.Protocol,
		plan.RequestSnapshot.Path, nullableString(plan.RequestSnapshot.APIKeyID),
		nullableString(plan.RequestSnapshot.CredentialID), nullableString(plan.RequestSnapshot.UserID),
		nullableString(plan.RequestSnapshot.TeamID), nullableString(plan.RequestSnapshot.EntrypointID),
		nullableString(plan.RequestSnapshot.EntrypointRuleID), nullableString(plan.RequestSnapshot.RecipeID),
		nullRevision(plan.RequestSnapshot.RoutingRevision), plan.RequestSnapshot.StatusCode,
		nullableString(plan.RequestSnapshot.ErrorCode), input, output, total,
		plan.ServedInputTokens, plan.ServedOutputTokens, sumQuotaText(plan.ServedInputTokens, plan.ServedOutputTokens),
		usageState, costPayload, metadata, plan.RequestSnapshot.OccurredAt, now,
		plan.ReconciliationID, plan.Strategy, plan.OriginalEventID, unknownDelta)
	if err != nil {
		return fmt.Errorf("insert reconciliation usage event: %w", err)
	}
	for _, dispatch := range plan.Dispatches {
		state := "known_zero"
		if plan.Strategy == quotareconciliation.StrategyActual {
			state = "known_actual"
		}
		evidence, err := hex.DecodeString(dispatch.EvidenceDigest)
		if err != nil || len(evidence) != sha256.Size {
			return errors.New("reconciliation dispatch evidence digest is invalid")
		}
		_, err = tx.ExecContext(ctx, `INSERT INTO usage_dispatches (
  namespace_id,event_date,event_id,admission_id,dispatch_id,dispatch_ordinal,attempt_count,
  dispatch_type,logical_model_id,model_revision,backend_id,provider_id,provider_model_id,
  pricing_revision,input_tokens,cache_read_tokens,cache_write_tokens,output_tokens,
  usage_state,cost_numerator,currency,evidence_digest,started_at,completed_at,corrects_dispatch_id
) VALUES ($1,$2,$3,$4,$5,$6,1,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
  $18,$19,$20,$21,$22,$23,$24)`,
			plan.NamespaceID, eventDate, plan.CorrectionEventID, plan.AdmissionID,
			dispatch.DispatchID, dispatch.Ordinal, dispatch.DispatchType,
			nullableString(dispatch.ModelID), nullRevision(dispatch.ModelRevision), nullableString(dispatch.BackendID),
			nullableString(dispatch.ProviderID), nullableString(dispatch.ProviderModelID),
			nullRevision(dispatch.PricingRevision), dispatch.InputTokens, dispatch.CacheReadTokens,
			dispatch.CacheWriteTokens, dispatch.OutputTokens, state, dispatch.Cost.Numerator,
			nullableString(dispatch.Cost.Currency), evidence, dispatch.StartedAt, dispatch.CompletedAt,
			dispatch.CorrectsDispatchID)
		if err != nil {
			return fmt.Errorf("insert reconciliation usage dispatch: %w", err)
		}
	}
	return nil
}

type reconciliationCostRow struct {
	Currency             string `json:"currency"`
	KnownNumerator       string `json:"knownNumerator"`
	KnownDispatches      string `json:"knownDispatches"`
	IncompleteDispatches string `json:"incompleteDispatches"`
}

func reconciliationCosts(plan quotareconciliation.Plan) ([]reconciliationCostRow, error) {
	counts := make(map[string]int)
	for _, dispatch := range plan.Dispatches {
		if dispatch.Cost.Currency == "" {
			return nil, errors.New("reconciliation dispatch currency is missing")
		}
		counts[dispatch.Cost.Currency]++
	}
	known := make(map[string]string)
	for _, cost := range plan.CorrectionCharge.Costs {
		known[cost.Currency] = cost.Numerator
	}
	result := make([]reconciliationCostRow, 0, len(counts))
	for currency, count := range counts {
		amount := "0"
		knownDispatches := "0"
		if plan.Strategy == quotareconciliation.StrategyActual {
			amount = known[currency]
			knownDispatches = strconv.Itoa(count)
		}
		result = append(result, reconciliationCostRow{
			Currency: currency, KnownNumerator: amount,
			KnownDispatches: knownDispatches, IncompleteDispatches: "-" + strconv.Itoa(count),
		})
	}
	sort.Slice(result, func(left, right int) bool {
		return result[left].Currency < result[right].Currency
	})
	return result, nil
}

func reconciliationEventQuantities(plan quotareconciliation.Plan) (string, string, string, error) {
	input, err := quota.ParseQuotaInteger(plan.CorrectionCharge.InputTokens)
	if err != nil {
		return "", "", "", err
	}
	output, err := quota.ParseQuotaInteger(plan.CorrectionCharge.OutputTokens)
	if err != nil {
		return "", "", "", err
	}
	total, err := input.Add(output)
	if err != nil || total.String() != plan.CorrectionCharge.TotalTokens {
		return "", "", "", quotareconciliation.ErrEvidenceConflict
	}
	return input.String(), output.String(), total.String(), nil
}

func sumQuotaText(leftText, rightText string) string {
	left, leftErr := quota.ParseQuotaInteger(leftText)
	right, rightErr := quota.ParseQuotaInteger(rightText)
	if leftErr != nil || rightErr != nil {
		return "0"
	}
	total, err := left.Add(right)
	if err != nil {
		return "0"
	}
	return total.String()
}

func nullRevision(value int64) any {
	if value <= 0 {
		return nil
	}
	return value
}

func appendReconciliationPlanAudit(
	ctx context.Context,
	tx *sql.Tx,
	plan quotareconciliation.Plan,
	action, reason string,
	_ time.Time,
) error {
	var revision int64
	if err := tx.QueryRowContext(ctx, `SELECT etag_revision FROM unknown_usage_fences
WHERE namespace_id=$1 AND id=$2`, plan.NamespaceID, plan.FenceID).Scan(&revision); err != nil {
		return err
	}
	return appendQuotaReconciliationAudit(ctx, tx, quotareconciliation.ReconcileRequest{
		NamespaceID: plan.NamespaceID, FenceID: plan.FenceID, Strategy: plan.Strategy,
		Reason: plan.Reason, Actor: plan.Actor,
	}, plan.FenceID, uint64(revision), action, reason, map[string]string{
		"strategy": string(plan.Strategy), "reconciliation_id": plan.ReconciliationID,
	})
}

func decodeReconciliationPlan(payload []byte, destination *quotareconciliation.Plan) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return fmt.Errorf("decode unknown-usage reconciliation plan: %w", err)
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return errors.New("unknown-usage reconciliation plan has trailing data")
	}
	return nil
}
