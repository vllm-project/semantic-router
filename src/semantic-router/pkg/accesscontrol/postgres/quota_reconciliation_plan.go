package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	managementcommandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
)

func (s *Store) PrepareUnknownUsageReconciliation(
	ctx context.Context,
	command managementcommand.Command,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	now time.Time,
) (quotareconciliation.EnqueueResult, error) {
	if s == nil || s.db == nil || validateUUID("operation id", operationID) != nil || now.IsZero() {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrInvalidRequest
	}
	tx, prepareUnknownUsageReconciliationErr := s.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("begin unknown-usage reconciliation: %w", prepareUnknownUsageReconciliationErr)
	}
	defer func() { _ = tx.Rollback() }()
	replay, replayed, prepareUnknownUsageReconciliationErr := managementcommandpostgres.Lock(ctx, tx, command)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	if replayed {
		if replay.Operation == nil {
			return quotareconciliation.EnqueueResult{}, errors.New("unknown-usage command replay is not an operation")
		}
		operation, err := loadQuotaReconciliationOperation(ctx, tx, request.NamespaceID, replay.Operation.OperationID)
		if err != nil {
			return quotareconciliation.EnqueueResult{}, err
		}
		if err := tx.Commit(); err != nil {
			return quotareconciliation.EnqueueResult{}, err
		}
		return quotareconciliation.EnqueueResult{Operation: operation, Replayed: true}, nil
	}

	var state quotareconciliation.FenceState
	var revision int64
	if err := tx.QueryRowContext(ctx, `SELECT state, etag_revision
FROM unknown_usage_fences WHERE namespace_id=$1 AND id=$2 FOR UPDATE`,
		request.NamespaceID, request.FenceID).Scan(&state, &revision); errors.Is(err, sql.ErrNoRows) {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrNotFound
	} else if err != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("lock unknown-usage fence: %w", err)
	}
	if state == quotareconciliation.FenceResolved {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrResolved
	}
	if state == quotareconciliation.FenceReconciling {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrReconciliationConflict
	}
	if revision <= 0 || uint64(revision) != request.ExpectedRevision {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrRevisionConflict
	}
	fence, prepareUnknownUsageReconciliationErr := loadQuotaFence(ctx, tx, request.NamespaceID, request.FenceID)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	ledger, prepareUnknownUsageReconciliationErr := loadReconciliationLedger(ctx, tx, fence)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	reconciliationID := uuid.NewString()
	plan, prepareUnknownUsageReconciliationErr := buildReconciliationPlan(request, fence, ledger, operationID,
		reconciliationID, uuid.NewString(), now)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	digestText, payload, prepareUnknownUsageReconciliationErr := quotareconciliation.DigestPlan(plan)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	digest, _ := hex.DecodeString(digestText)
	actorChain, prepareUnknownUsageReconciliationErr := json.Marshal(request.Actor.ActorChain)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("encode reconciliation actor chain: %w", prepareUnknownUsageReconciliationErr)
	}
	targetScope, prepareUnknownUsageReconciliationErr := json.Marshal(struct {
		Version    int      `json:"version"`
		FenceID    string   `json:"fenceId"`
		BindingIDs []string `json:"bindingIds"`
	}{1, fence.ID, fenceBindingIDs(fence.Bindings)})
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("encode reconciliation target scope: %w", prepareUnknownUsageReconciliationErr)
	}
	targetIDs, prepareUnknownUsageReconciliationErr := json.Marshal([]string{fence.ID})
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("encode reconciliation target ids: %w", prepareUnknownUsageReconciliationErr)
	}
	active := command.ActiveDigest()
	if _, err := tx.ExecContext(ctx, `INSERT INTO management_operations (
  id,namespace_id,kind,origin_principal_id,actor_chain,request_digest,state,
  progress_completed,progress_total,target_scope,target_ids,created_at,updated_at
) VALUES ($1,$2,$3,$4,$5,$6,'pending',0,1,$7,$8,$9,$9)`,
		operationID, request.NamespaceID, quotareconciliation.OperationKind,
		request.Actor.PrincipalID, actorChain, active.RequestDigest[:], targetScope, targetIDs, now); err != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("insert unknown-usage reconciliation operation: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_reconciliation_plans (
  reconciliation_id,namespace_id,fence_id,operation_id,strategy,plan_digest,plan_payload,
  phase,available_at,created_at,updated_at
) VALUES ($1,$2,$3,$4,$5,$6,$7,'runtime_pending',$8,$8,$8)`,
		reconciliationID, request.NamespaceID, fence.ID, operationID, request.Strategy,
		digest, payload, now); err != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("insert unknown-usage reconciliation plan: %w", err)
	}
	newRevision := revision + 1
	result, prepareUnknownUsageReconciliationErr := tx.ExecContext(ctx, `UPDATE unknown_usage_fences
SET state='reconciling',etag_revision=$3,reconciliation_id=$4,
  reconciliation_strategy=$5,reconciliation_actor_id=$6,reconciliation_reason=$7,
  updated_at=$8
WHERE namespace_id=$1 AND id=$2 AND state='open' AND etag_revision=$9`,
		request.NamespaceID, fence.ID, newRevision, reconciliationID, request.Strategy,
		request.Actor.PrincipalID, request.Reason, now, revision)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("mark unknown-usage fence reconciling: %w", prepareUnknownUsageReconciliationErr)
	}
	if count, _ := result.RowsAffected(); count != 1 {
		return quotareconciliation.EnqueueResult{}, quotareconciliation.ErrRevisionConflict
	}
	if err := managementcommandpostgres.CompleteOperation(ctx, tx, command, managementcommand.OperationResult{
		OperationID: operationID, ResponseStatus: 202,
	}); err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	if err := appendQuotaReconciliationAudit(ctx, tx, request, fence.ID,
		uint64(newRevision), "quota.unknown_usage_fence.reconcile_requested", "Reconcile unknown usage.",
		map[string]string{"strategy": string(request.Strategy), "reconciliation_id": reconciliationID}); err != nil {
		return quotareconciliation.EnqueueResult{}, err
	}
	if err := tx.Commit(); err != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("commit unknown-usage reconciliation: %w", err)
	}
	return quotareconciliation.EnqueueResult{Operation: quotareconciliation.Operation{
		ID: operationID, NamespaceID: request.NamespaceID, FenceID: fence.ID,
		Kind: quotareconciliation.OperationKind, OriginPrincipalID: request.Actor.PrincipalID,
		ActorChain: append([]string(nil), request.Actor.ActorChain...), Version: 1,
		State: quotareconciliation.OperationPending,
		Total: 1, CreatedAt: now, UpdatedAt: now,
	}}, nil
}

func (s *Store) Prepare(
	ctx context.Context,
	command managementcommand.Command,
	request quotareconciliation.ReconcileRequest,
	operationID string,
	now time.Time,
) (quotareconciliation.EnqueueResult, error) {
	return s.PrepareUnknownUsageReconciliation(ctx, command, request, operationID, now)
}

type reconciliationLedger struct {
	EventID                 string
	Partition               string
	Snapshot                quotareconciliation.RequestSnapshot
	KnownInput, KnownOutput quota.QuotaInteger
	KnownCosts              map[string]quota.QuotaInteger
	Dispatches              []ledgerDispatch
}

type ledgerDispatch struct {
	DispatchID, ParentDispatchID, DispatchType string
	Ordinal                                    int
	ModelID                                    string
	ModelRevision                              int64
	BackendID, ProviderID, ProviderModelID     string
	PricingRevision                            int64
	Input, CacheRead, CacheWrite, Output       quota.QuotaInteger
	Unknown                                    bool
	Cost                                       *quota.QuotaInteger
	Currency, EvidenceDigest                   string
	StartedAt                                  time.Time
	CompletedAt                                *time.Time
}

func loadReconciliationLedger(ctx context.Context, tx *sql.Tx, fence quotareconciliation.Fence) (reconciliationLedger, error) {
	var value reconciliationLedger
	var knownInput, knownOutput string
	err := tx.QueryRowContext(ctx, `SELECT e.event_id::text, n.quota_partition_id,
  COALESCE(e.external_request_id,''), e.protocol, e.path,
  COALESCE(e.api_key_id::text,''), COALESCE(e.credential_id::text,''), COALESCE(e.user_id::text,''),
  COALESCE(e.team_id::text,''), COALESCE(e.entrypoint_id,''), COALESCE(e.entrypoint_rule_id,''),
  COALESCE(e.recipe_id,''), COALESCE(e.routing_revision,0), e.status_code, COALESCE(e.error_code,''),
  e.occurred_at, (e.request_metadata->>'completedAt')::timestamptz,
  e.input_tokens::text, e.output_tokens::text
FROM usage_events e JOIN access_namespaces n ON n.id=e.namespace_id
WHERE e.namespace_id=$1 AND e.admission_id=$2 AND e.event_kind='unknown'`,
		fence.NamespaceID, fence.AdmissionID).Scan(&value.EventID, &value.Partition,
		&value.Snapshot.ExternalRequestID, &value.Snapshot.Protocol, &value.Snapshot.Path,
		&value.Snapshot.APIKeyID, &value.Snapshot.CredentialID, &value.Snapshot.UserID,
		&value.Snapshot.TeamID, &value.Snapshot.EntrypointID, &value.Snapshot.EntrypointRuleID,
		&value.Snapshot.RecipeID, &value.Snapshot.RoutingRevision, &value.Snapshot.StatusCode,
		&value.Snapshot.ErrorCode, &value.Snapshot.OccurredAt, &value.Snapshot.CompletedAt,
		&knownInput, &knownOutput)
	if err != nil {
		return reconciliationLedger{}, fmt.Errorf("load unknown usage event: %w", err)
	}
	value.KnownInput, err = quota.ParseQuotaInteger(knownInput)
	if err != nil {
		return reconciliationLedger{}, errors.New("stored known input usage is invalid")
	}
	value.KnownOutput, err = quota.ParseQuotaInteger(knownOutput)
	if err != nil {
		return reconciliationLedger{}, errors.New("stored known output usage is invalid")
	}
	value.KnownCosts = make(map[string]quota.QuotaInteger)
	rows, err := tx.QueryContext(ctx, `SELECT dispatch_id, COALESCE(parent_dispatch_id,''),
  dispatch_ordinal, dispatch_type, COALESCE(logical_model_id::text,''), COALESCE(model_revision,0),
  COALESCE(backend_id::text,''), COALESCE(provider_id,''), COALESCE(provider_model_id,''),
  COALESCE(pricing_revision,0), input_tokens::text, cache_read_tokens::text,
  cache_write_tokens::text, output_tokens::text, usage_state,
  cost_numerator::text, COALESCE(currency,''), evidence_digest, started_at, completed_at
FROM usage_dispatches
WHERE namespace_id=$1 AND admission_id=$2 AND corrects_dispatch_id IS NULL
ORDER BY dispatch_ordinal, dispatch_id`, fence.NamespaceID, fence.AdmissionID)
	if err != nil {
		return reconciliationLedger{}, fmt.Errorf("load immutable dispatch ledger: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var dispatch ledgerDispatch
		var input, cacheRead, cacheWrite, output string
		var state string
		var cost sql.NullString
		var evidence []byte
		var completed sql.NullTime
		if err := rows.Scan(&dispatch.DispatchID, &dispatch.ParentDispatchID, &dispatch.Ordinal,
			&dispatch.DispatchType, &dispatch.ModelID, &dispatch.ModelRevision, &dispatch.BackendID,
			&dispatch.ProviderID, &dispatch.ProviderModelID, &dispatch.PricingRevision,
			&input, &cacheRead, &cacheWrite, &output, &state, &cost, &dispatch.Currency,
			&evidence, &dispatch.StartedAt, &completed); err != nil {
			return reconciliationLedger{}, fmt.Errorf("scan immutable dispatch ledger: %w", err)
		}
		for text, target := range map[string]*quota.QuotaInteger{
			input: &dispatch.Input, cacheRead: &dispatch.CacheRead, cacheWrite: &dispatch.CacheWrite, output: &dispatch.Output,
		} {
			parsed, parseErr := quota.ParseQuotaInteger(text)
			if parseErr != nil {
				return reconciliationLedger{}, errors.New("stored dispatch usage is invalid")
			}
			*target = parsed
		}
		dispatch.Unknown = state == "unknown"
		dispatch.EvidenceDigest = hex.EncodeToString(evidence)
		if cost.Valid {
			parsed, parseErr := quota.ParseQuotaInteger(cost.String)
			if parseErr != nil {
				return reconciliationLedger{}, errors.New("stored dispatch cost is invalid")
			}
			dispatch.Cost = &parsed
			total := value.KnownCosts[dispatch.Currency]
			total, parseErr = total.Add(parsed)
			if parseErr != nil {
				return reconciliationLedger{}, errors.New("stored dispatch cost aggregate overflows")
			}
			value.KnownCosts[dispatch.Currency] = total
		}
		if completed.Valid {
			completedAt := completed.Time.UTC()
			dispatch.CompletedAt = &completedAt
		}
		value.Dispatches = append(value.Dispatches, dispatch)
	}
	if err := rows.Err(); err != nil {
		return reconciliationLedger{}, err
	}
	if len(value.Dispatches) == 0 {
		return reconciliationLedger{}, errors.New("unknown settlement has no dispatch ledger")
	}
	return value, nil
}

func buildReconciliationPlan(
	request quotareconciliation.ReconcileRequest,
	fence quotareconciliation.Fence,
	ledger reconciliationLedger,
	operationID, reconciliationID, correctionEventID string,
	now time.Time,
) (quotareconciliation.Plan, error) {
	actual, correctionDispatches, correctionCharge, servedInput, servedOutput, buildReconciliationPlanErr := validateActualReconciliation(request, fence, ledger)
	if buildReconciliationPlanErr != nil {
		return quotareconciliation.Plan{}, buildReconciliationPlanErr
	}
	corrections := make([]quotaruntime.CounterCorrection, 0, len(fence.Bindings))
	for _, binding := range fence.Bindings {
		amount := "0"
		charge := request.Strategy != quotareconciliation.StrategyWaive
		switch request.Strategy {
		case quotareconciliation.StrategyActual:
			amount, buildReconciliationPlanErr = actualMetricAmount(binding, ledger, actual, servedInput, servedOutput)
		case quotareconciliation.StrategyConservativeDebit:
			amount = binding.MaximumDebit
			if amount == "" {
				amount = binding.AdmissionLimit
			}
			if amount == "" {
				buildReconciliationPlanErr = quotareconciliation.ErrEvidenceConflict
			}
		case quotareconciliation.StrategyWaive:
			amount = "0"
		}
		if buildReconciliationPlanErr != nil {
			return quotareconciliation.Plan{}, buildReconciliationPlanErr
		}
		if _, err := quota.ParseQuotaInteger(amount); err != nil {
			return quotareconciliation.Plan{}, quotareconciliation.ErrEvidenceConflict
		}
		correction := quotaruntime.CounterCorrection{
			BindingID: binding.BindingID, RuleID: binding.RuleID, Metric: binding.Metric,
			Algorithm: binding.Algorithm, Enforcement: binding.Enforcement, Amount: amount,
			CounterIncompleteCount: binding.CounterIncompleteCount,
			ChargeAt:               ledger.Snapshot.OccurredAt.UTC(), Window: binding.Window, Charge: charge,
			Known: request.Strategy == quotareconciliation.StrategyActual,
		}
		if binding.Algorithm == quota.AlgorithmCalendarWindow {
			correction.CalendarStart, correction.CalendarEnd, buildReconciliationPlanErr = calendarInterval(
				ledger.Snapshot.OccurredAt, binding.CalendarPeriod, binding.Timezone)
			if buildReconciliationPlanErr != nil {
				return quotareconciliation.Plan{}, quotareconciliation.ErrEvidenceConflict
			}
		}
		corrections = append(corrections, correction)
	}
	unknownCount := 0
	for _, dispatch := range ledger.Dispatches {
		if dispatch.Unknown {
			unknownCount++
		}
	}
	return quotareconciliation.Plan{
		ReconciliationID: reconciliationID, NamespaceID: fence.NamespaceID,
		Partition: ledger.Partition, FenceID: fence.ID,
		AdmissionID: fence.AdmissionID, OriginalEventID: ledger.EventID,
		CorrectionEventID: correctionEventID, OperationID: operationID,
		Strategy: request.Strategy, Reason: request.Reason, Actor: request.Actor,
		EvidenceReferences: append([]string(nil), request.EvidenceReferences...),
		Corrections:        corrections, Dispatches: correctionDispatches,
		UnknownDispatchCount: fmt.Sprintf("%d", unknownCount), CorrectionCharge: correctionCharge,
		ServedInputTokens: servedInput.String(), ServedOutputTokens: servedOutput.String(),
		RequestSnapshot: ledger.Snapshot, CreatedAt: now,
	}, nil
}

type reconciledActual struct {
	ByID          map[string]quotareconciliation.ActualDispatchUsage
	Input, Output quota.QuotaInteger
	Costs         map[string]quota.QuotaInteger
}

func validateActualReconciliation(
	request quotareconciliation.ReconcileRequest,
	fence quotareconciliation.Fence,
	ledger reconciliationLedger,
) (reconciledActual, []quotareconciliation.CorrectionDispatch, quotareconciliation.Charge,
	quota.QuotaInteger, quota.QuotaInteger, error,
) {
	zero, _ := quota.ParseQuotaInteger("0")
	actual := reconciledActual{
		ByID:  make(map[string]quotareconciliation.ActualDispatchUsage),
		Input: zero, Output: zero, Costs: make(map[string]quota.QuotaInteger),
	}
	servedInput, servedOutput := zero, zero
	if request.Strategy == quotareconciliation.StrategyActual {
		var err error
		servedInput, err = quota.ParseQuotaInteger(request.Actual.ServedInputTokens)
		if err != nil {
			return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrInvalidRequest
		}
		servedOutput, err = quota.ParseQuotaInteger(request.Actual.ServedOutputTokens)
		if err != nil {
			return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrInvalidRequest
		}
		for _, supplied := range request.Actual.Dispatches {
			if _, duplicate := actual.ByID[supplied.DispatchID]; duplicate {
				return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
			}
			actual.ByID[supplied.DispatchID] = supplied
		}
	}
	corrections := make([]quotareconciliation.CorrectionDispatch, 0)
	unknownEvidence := make(map[string]quotareconciliation.UnknownDispatch, len(fence.Unknown))
	for _, evidence := range fence.Unknown {
		unknownEvidence[evidence.DispatchID] = evidence
	}
	for _, original := range ledger.Dispatches {
		if !original.Unknown {
			continue
		}
		input, cacheRead, cacheWrite, output := zero, zero, zero, zero
		cost := zero
		if request.Strategy == quotareconciliation.StrategyActual {
			supplied, exists := actual.ByID[original.DispatchID]
			evidence := unknownEvidence[original.DispatchID]
			if !exists || supplied.EvidenceDigest != original.EvidenceDigest ||
				supplied.EvidenceDigest != evidence.EvidenceDigest || supplied.Cost.Currency != original.Currency {
				return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
			}
			var err error
			values := []struct {
				text   string
				target *quota.QuotaInteger
			}{
				{supplied.InputTokens, &input},
				{supplied.CacheReadTokens, &cacheRead},
				{supplied.CacheWriteTokens, &cacheWrite},
				{supplied.OutputTokens, &output},
				{supplied.Cost.Numerator, &cost},
			}
			for _, value := range values {
				*value.target, err = quota.ParseQuotaInteger(value.text)
				if err != nil {
					return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrInvalidRequest
				}
			}
			actual.Input, err = actual.Input.Add(input)
			if err == nil {
				actual.Output, err = actual.Output.Add(output)
			}
			if err != nil {
				return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
			}
			total := actual.Costs[original.Currency]
			total, err = total.Add(cost)
			if err != nil {
				return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
			}
			actual.Costs[original.Currency] = total
		}
		completed := ledger.Snapshot.CompletedAt
		if original.CompletedAt != nil {
			completed = *original.CompletedAt
		}
		corrections = append(corrections, quotareconciliation.CorrectionDispatch{
			DispatchID: uuid.NewString(), CorrectsDispatchID: original.DispatchID,
			Ordinal: original.Ordinal, DispatchType: original.DispatchType,
			ModelID: original.ModelID, ModelRevision: original.ModelRevision,
			BackendID: original.BackendID, ProviderID: original.ProviderID,
			ProviderModelID: original.ProviderModelID, PricingRevision: original.PricingRevision,
			InputTokens: input.String(), CacheReadTokens: cacheRead.String(),
			CacheWriteTokens: cacheWrite.String(), OutputTokens: output.String(),
			Cost:           quotareconciliation.Cost{Currency: original.Currency, Numerator: cost.String()},
			EvidenceDigest: original.EvidenceDigest, StartedAt: original.StartedAt, CompletedAt: completed,
		})
	}
	if request.Strategy == quotareconciliation.StrategyActual && len(actual.ByID) != len(corrections) {
		return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
	}
	charge := quotareconciliation.Charge{InputTokens: actual.Input.String(), OutputTokens: actual.Output.String()}
	total, err := actual.Input.Add(actual.Output)
	if err != nil {
		return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
	}
	charge.TotalTokens = total.String()
	for currency, numerator := range actual.Costs {
		charge.Costs = append(charge.Costs, quotareconciliation.Cost{Currency: currency, Numerator: numerator.String()})
	}
	sort.Slice(charge.Costs, func(i, j int) bool { return charge.Costs[i].Currency < charge.Costs[j].Currency })
	return actual, corrections, charge, servedInput, servedOutput, nil
}

func actualMetricAmount(binding quotareconciliation.Binding, ledger reconciliationLedger,
	actual reconciledActual, servedInput, servedOutput quota.QuotaInteger,
) (string, error) {
	input, err := ledger.KnownInput.Add(actual.Input)
	if err != nil {
		return "", quotareconciliation.ErrEvidenceConflict
	}
	output, err := ledger.KnownOutput.Add(actual.Output)
	if err != nil {
		return "", quotareconciliation.ErrEvidenceConflict
	}
	switch binding.Metric {
	case quota.MetricInputTokens:
		return input.String(), nil
	case quota.MetricOutputTokens:
		return output.String(), nil
	case quota.MetricTotalTokens:
		total, err := input.Add(output)
		if err != nil {
			return "", quotareconciliation.ErrEvidenceConflict
		}
		return total.String(), nil
	case quota.MetricServedInputTokens:
		return servedInput.String(), nil
	case quota.MetricServedOutputTokens:
		return servedOutput.String(), nil
	case quota.MetricServedTotalTokens:
		total, err := servedInput.Add(servedOutput)
		if err != nil {
			return "", quotareconciliation.ErrEvidenceConflict
		}
		return total.String(), nil
	case quota.MetricCost:
		known := ledger.KnownCosts[binding.Currency]
		total, err := known.Add(actual.Costs[binding.Currency])
		if err != nil {
			return "", quotareconciliation.ErrEvidenceConflict
		}
		return total.String(), nil
	default:
		return "", quotareconciliation.ErrEvidenceConflict
	}
}

func calendarInterval(at time.Time, period quota.CalendarPeriod, timezone string) (time.Time, time.Time, error) {
	location, err := time.LoadLocation(timezone)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	local := at.In(location)
	var start, end time.Time
	switch period {
	case quota.CalendarPeriodDay:
		start = time.Date(local.Year(), local.Month(), local.Day(), 0, 0, 0, 0, location)
		end = start.AddDate(0, 0, 1)
	case quota.CalendarPeriodMonth:
		start = time.Date(local.Year(), local.Month(), 1, 0, 0, 0, 0, location)
		end = start.AddDate(0, 1, 0)
	default:
		return time.Time{}, time.Time{}, errors.New("invalid calendar period")
	}
	return start.UTC().Truncate(time.Millisecond), end.UTC().Truncate(time.Millisecond), nil
}

func fenceBindingIDs(bindings []quotareconciliation.Binding) []string {
	ids := make([]string, 0, len(bindings))
	seen := make(map[string]struct{}, len(bindings))
	for _, binding := range bindings {
		if _, ok := seen[binding.BindingID]; ok {
			continue
		}
		seen[binding.BindingID] = struct{}{}
		ids = append(ids, binding.BindingID)
	}
	sort.Strings(ids)
	return ids
}

func loadQuotaReconciliationOperation(ctx context.Context, queryer quotaFenceQueryer,
	namespaceID, operationID string,
) (quotareconciliation.Operation, error) {
	var operation quotareconciliation.Operation
	var actorChain []byte
	var completed, total int64
	var completedAt sql.NullTime
	err := queryer.QueryRowContext(ctx, `SELECT o.id::text,o.namespace_id::text,p.fence_id::text,
  o.kind,o.origin_principal_id::text,o.actor_chain,o.version,o.state,
  o.progress_completed,o.progress_total,o.created_at,o.updated_at,p.completed_at
FROM management_operations o
JOIN unknown_usage_reconciliation_plans p ON p.operation_id=o.id
WHERE o.namespace_id=$1 AND o.id=$2 AND o.kind=$3`, namespaceID, operationID,
		quotareconciliation.OperationKind).Scan(&operation.ID, &operation.NamespaceID, &operation.FenceID,
		&operation.Kind, &operation.OriginPrincipalID, &actorChain, &operation.Version,
		&operation.State, &completed, &total,
		&operation.CreatedAt, &operation.UpdatedAt, &completedAt)
	if errors.Is(err, sql.ErrNoRows) {
		return quotareconciliation.Operation{}, quotareconciliation.ErrNotFound
	}
	if err != nil {
		return quotareconciliation.Operation{}, err
	}
	if completed < 0 || total < 0 || completed > total {
		return quotareconciliation.Operation{}, errors.New("stored operation progress is invalid")
	}
	if err := json.Unmarshal(actorChain, &operation.ActorChain); err != nil {
		return quotareconciliation.Operation{}, fmt.Errorf("decode reconciliation operation actor chain: %w", err)
	}
	operation.Completed, operation.Total = uint64(completed), uint64(total)
	operation.CreatedAt = operation.CreatedAt.UTC()
	operation.UpdatedAt = operation.UpdatedAt.UTC()
	if completedAt.Valid {
		value := completedAt.Time.UTC()
		operation.CompletedAt = &value
	}
	return operation, nil
}

func (s *Store) GetOperation(ctx context.Context, namespaceID, operationID string) (quotareconciliation.Operation, error) {
	if s == nil || s.db == nil || validateUUID("namespace id", namespaceID) != nil ||
		validateUUID("operation id", operationID) != nil {
		return quotareconciliation.Operation{}, quotareconciliation.ErrInvalidRequest
	}
	return loadQuotaReconciliationOperation(ctx, s.db, namespaceID, operationID)
}
