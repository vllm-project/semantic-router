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
	result, prepareUnknownUsageReconciliationErr := prepareUnknownUsageInTransaction(
		ctx, tx, command, request, operationID, now,
	)
	if prepareUnknownUsageReconciliationErr != nil {
		return quotareconciliation.EnqueueResult{}, prepareUnknownUsageReconciliationErr
	}
	if err := tx.Commit(); err != nil {
		return quotareconciliation.EnqueueResult{}, fmt.Errorf("commit unknown-usage reconciliation: %w", err)
	}
	return result, nil
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
	actual, servedInput, servedOutput, err := parseActualReconciliation(request, zero)
	if err != nil {
		return actual, nil, quotareconciliation.Charge{}, zero, zero, err
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
		correction, reconcileErr := reconcileUnknownDispatch(
			request.Strategy, original, unknownEvidence[original.DispatchID], ledger.Snapshot.CompletedAt,
			&actual, zero,
		)
		if reconcileErr != nil {
			return actual, nil, quotareconciliation.Charge{}, zero, zero, reconcileErr
		}
		corrections = append(corrections, correction)
	}
	if request.Strategy == quotareconciliation.StrategyActual && len(actual.ByID) != len(corrections) {
		return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
	}
	charge, err := reconciliationCharge(actual)
	if err != nil {
		return actual, nil, quotareconciliation.Charge{}, zero, zero, quotareconciliation.ErrEvidenceConflict
	}
	return actual, corrections, charge, servedInput, servedOutput, nil
}

func parseActualReconciliation(
	request quotareconciliation.ReconcileRequest,
	zero quota.QuotaInteger,
) (reconciledActual, quota.QuotaInteger, quota.QuotaInteger, error) {
	actual := reconciledActual{
		ByID:  make(map[string]quotareconciliation.ActualDispatchUsage),
		Input: zero, Output: zero, Costs: make(map[string]quota.QuotaInteger),
	}
	if request.Strategy != quotareconciliation.StrategyActual {
		return actual, zero, zero, nil
	}
	servedInput, err := quota.ParseQuotaInteger(request.Actual.ServedInputTokens)
	if err != nil {
		return actual, zero, zero, quotareconciliation.ErrInvalidRequest
	}
	servedOutput, err := quota.ParseQuotaInteger(request.Actual.ServedOutputTokens)
	if err != nil {
		return actual, zero, zero, quotareconciliation.ErrInvalidRequest
	}
	for _, supplied := range request.Actual.Dispatches {
		if _, duplicate := actual.ByID[supplied.DispatchID]; duplicate {
			return actual, zero, zero, quotareconciliation.ErrEvidenceConflict
		}
		actual.ByID[supplied.DispatchID] = supplied
	}
	return actual, servedInput, servedOutput, nil
}

func reconcileUnknownDispatch(
	strategy quotareconciliation.Strategy,
	original ledgerDispatch,
	evidence quotareconciliation.UnknownDispatch,
	defaultCompleted time.Time,
	actual *reconciledActual,
	zero quota.QuotaInteger,
) (quotareconciliation.CorrectionDispatch, error) {
	input, cacheRead, cacheWrite, output, cost := zero, zero, zero, zero, zero
	if strategy == quotareconciliation.StrategyActual {
		supplied, exists := actual.ByID[original.DispatchID]
		if !exists || supplied.EvidenceDigest != original.EvidenceDigest ||
			supplied.EvidenceDigest != evidence.EvidenceDigest || supplied.Cost.Currency != original.Currency {
			return quotareconciliation.CorrectionDispatch{}, quotareconciliation.ErrEvidenceConflict
		}
		var err error
		input, cacheRead, cacheWrite, output, cost, err = parseActualDispatchUsage(supplied)
		if err != nil {
			return quotareconciliation.CorrectionDispatch{}, err
		}
		if err := accumulateActualDispatch(actual, original.Currency, input, output, cost); err != nil {
			return quotareconciliation.CorrectionDispatch{}, err
		}
	}
	completed := defaultCompleted
	if original.CompletedAt != nil {
		completed = *original.CompletedAt
	}
	return quotareconciliation.CorrectionDispatch{
		DispatchID: uuid.NewString(), CorrectsDispatchID: original.DispatchID,
		Ordinal: original.Ordinal, DispatchType: original.DispatchType,
		ModelID: original.ModelID, ModelRevision: original.ModelRevision,
		BackendID: original.BackendID, ProviderID: original.ProviderID,
		ProviderModelID: original.ProviderModelID, PricingRevision: original.PricingRevision,
		InputTokens: input.String(), CacheReadTokens: cacheRead.String(),
		CacheWriteTokens: cacheWrite.String(), OutputTokens: output.String(),
		Cost:           quotareconciliation.Cost{Currency: original.Currency, Numerator: cost.String()},
		EvidenceDigest: original.EvidenceDigest, StartedAt: original.StartedAt, CompletedAt: completed,
	}, nil
}

func parseActualDispatchUsage(
	supplied quotareconciliation.ActualDispatchUsage,
) (quota.QuotaInteger, quota.QuotaInteger, quota.QuotaInteger, quota.QuotaInteger, quota.QuotaInteger, error) {
	values := []string{
		supplied.InputTokens, supplied.CacheReadTokens, supplied.CacheWriteTokens,
		supplied.OutputTokens, supplied.Cost.Numerator,
	}
	parsed := make([]quota.QuotaInteger, len(values))
	for index, value := range values {
		amount, err := quota.ParseQuotaInteger(value)
		if err != nil {
			return quota.QuotaInteger{}, quota.QuotaInteger{}, quota.QuotaInteger{},
				quota.QuotaInteger{}, quota.QuotaInteger{}, quotareconciliation.ErrInvalidRequest
		}
		parsed[index] = amount
	}
	return parsed[0], parsed[1], parsed[2], parsed[3], parsed[4], nil
}

func accumulateActualDispatch(
	actual *reconciledActual,
	currency string,
	input quota.QuotaInteger,
	output quota.QuotaInteger,
	cost quota.QuotaInteger,
) error {
	var err error
	actual.Input, err = actual.Input.Add(input)
	if err == nil {
		actual.Output, err = actual.Output.Add(output)
	}
	if err != nil {
		return quotareconciliation.ErrEvidenceConflict
	}
	total, err := actual.Costs[currency].Add(cost)
	if err != nil {
		return quotareconciliation.ErrEvidenceConflict
	}
	actual.Costs[currency] = total
	return nil
}

func reconciliationCharge(actual reconciledActual) (quotareconciliation.Charge, error) {
	charge := quotareconciliation.Charge{
		InputTokens: actual.Input.String(), OutputTokens: actual.Output.String(),
	}
	total, err := actual.Input.Add(actual.Output)
	if err != nil {
		return quotareconciliation.Charge{}, err
	}
	charge.TotalTokens = total.String()
	for currency, numerator := range actual.Costs {
		charge.Costs = append(charge.Costs, quotareconciliation.Cost{
			Currency: currency, Numerator: numerator.String(),
		})
	}
	sort.Slice(charge.Costs, func(i, j int) bool { return charge.Costs[i].Currency < charge.Costs[j].Currency })
	return charge, nil
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
