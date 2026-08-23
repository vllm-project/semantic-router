package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quota"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotareconciliation"
)

const quotaFenceBaseColumns = `f.id::text, f.namespace_id::text, f.admission_id,
  f.state, f.etag_revision, f.reason, f.evidence,
  f.reconciliation_id::text, f.reconciliation_strategy,
  f.reconciliation_actor_id::text, f.reconciliation_reason,
  p.created_at,
  f.created_at, f.updated_at, f.resolved_at,
  e.input_tokens::text, e.output_tokens::text, e.total_tokens::text, e.costs`

const getQuotaFenceQuery = `SELECT ` + quotaFenceBaseColumns + `
FROM unknown_usage_fences f
JOIN usage_events e ON e.namespace_id=f.namespace_id
  AND e.admission_id=f.admission_id AND e.event_kind='unknown'
LEFT JOIN unknown_usage_reconciliation_plans p
  ON p.namespace_id=f.namespace_id AND p.reconciliation_id=f.reconciliation_id
WHERE f.namespace_id=$1 AND f.id=$2`

const quotaFenceBindingsQuery = `SELECT fb.binding_id::text, fb.rule_id::text,
  b.policy_id::text, s.kind, s.id::text, fb.metric, fb.algorithm, fb.enforcement,
  COALESCE(fb.admission_limit::text,''), COALESCE(fb.maximum_debit::text,''), fb.window_seconds,
  fb.calendar_period, fb.timezone, fb.currency,
  fb.unknown_dispatch_count::text, fb.counter_incomplete_count::text,
  (r.metric=fb.metric AND r.algorithm=fb.algorithm AND r.enforcement=fb.enforcement
    AND r.accounting='response_actual'
    AND r.window_seconds IS NOT DISTINCT FROM fb.window_seconds
    AND r.calendar_period IS NOT DISTINCT FROM fb.calendar_period
    AND r.timezone IS NOT DISTINCT FROM fb.timezone) AS semantics_match
FROM unknown_usage_fence_bindings fb
JOIN rate_limit_bindings b ON b.id=fb.binding_id
JOIN access_subjects s ON s.namespace_id=b.namespace_id AND s.id=b.subject_id
JOIN rate_limit_rules r ON r.policy_id=b.policy_id AND r.id=fb.rule_id
WHERE fb.fence_id=$1
ORDER BY fb.binding_id, fb.rule_id`

func (s *Store) GetUnknownUsageFence(
	ctx context.Context,
	namespaceID, fenceID string,
) (quotareconciliation.Fence, error) {
	if s == nil || s.db == nil {
		return quotareconciliation.Fence{}, quotareconciliation.ErrUnavailable
	}
	if validateUUID("namespace id", namespaceID) != nil || validateUUID("fence id", fenceID) != nil {
		return quotareconciliation.Fence{}, quotareconciliation.ErrInvalidRequest
	}
	return loadQuotaFence(ctx, s.db, namespaceID, fenceID)
}

func (s *Store) Get(ctx context.Context, namespaceID, fenceID string) (quotareconciliation.Fence, error) {
	return s.GetUnknownUsageFence(ctx, namespaceID, fenceID)
}

func (s *Store) ListUnknownUsageFences(
	ctx context.Context,
	query quotareconciliation.FenceQuery,
) (quotareconciliation.RepositoryPage, error) {
	if s == nil || s.db == nil {
		return quotareconciliation.RepositoryPage{}, quotareconciliation.ErrUnavailable
	}
	scope, err := query.Scope.Canonical()
	if err != nil || string(scope.NamespaceID) != query.NamespaceID || query.Limit < 1 || query.Limit > 200 ||
		(query.State != "" && query.State != quotareconciliation.FenceOpen &&
			query.State != quotareconciliation.FenceReconciling && query.State != quotareconciliation.FenceResolved) {
		return quotareconciliation.RepositoryPage{}, quotareconciliation.ErrInvalidRequest
	}
	var afterTime any
	var afterID any
	if query.After != nil {
		afterTime, afterID = query.After.CreatedAt, query.After.ID
	}
	rows, err := s.db.QueryContext(ctx, `SELECT f.id::text
FROM unknown_usage_fences f
WHERE f.namespace_id=$1 AND ($2='' OR f.state=$2)
  AND NOT EXISTS (
    SELECT 1
    FROM unknown_usage_fence_bindings fb
    JOIN rate_limit_bindings b ON b.id=fb.binding_id AND b.namespace_id=f.namespace_id
    JOIN access_subjects subject ON subject.namespace_id=b.namespace_id AND subject.id=b.subject_id
    WHERE fb.fence_id=f.id AND NOT (
      $3
      OR fb.binding_id=ANY($4::uuid[])
      OR (subject.kind='user' AND subject.id=ANY($5::uuid[]))
      OR (subject.kind='team' AND subject.id=ANY($6::uuid[]))
      OR (subject.kind='api_key' AND subject.id=ANY($7::uuid[]))
    )
  )
  AND ($8::timestamptz IS NULL OR f.created_at < $8
    OR (f.created_at=$8 AND f.id > $9::uuid))
ORDER BY f.created_at DESC, f.id ASC
LIMIT $10`, query.NamespaceID, string(query.State), scope.All,
		pq.Array(scope.IDs(accesscontrol.ScopeResourceRateLimitBinding)),
		pq.Array(scope.UserIDs), pq.Array(scope.TeamIDs), pq.Array(scope.APIKeyIDs),
		afterTime, afterID, query.Limit+1)
	if err != nil {
		return quotareconciliation.RepositoryPage{}, fmt.Errorf("list unknown-usage fences: %w", err)
	}
	defer rows.Close()
	ids := make([]string, 0, query.Limit+1)
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return quotareconciliation.RepositoryPage{}, fmt.Errorf("scan unknown-usage fence page: %w", err)
		}
		ids = append(ids, id)
	}
	if err := rows.Err(); err != nil {
		return quotareconciliation.RepositoryPage{}, fmt.Errorf("iterate unknown-usage fence page: %w", err)
	}
	hasMore := len(ids) > query.Limit
	if hasMore {
		ids = ids[:query.Limit]
	}
	items := make([]quotareconciliation.Fence, 0, len(ids))
	for _, id := range ids {
		fence, err := loadQuotaFence(ctx, s.db, query.NamespaceID, id)
		if err != nil {
			return quotareconciliation.RepositoryPage{}, err
		}
		items = append(items, fence)
	}
	return quotareconciliation.RepositoryPage{Items: items, HasMore: hasMore}, nil
}

func (s *Store) List(
	ctx context.Context,
	query quotareconciliation.FenceQuery,
) (quotareconciliation.RepositoryPage, error) {
	return s.ListUnknownUsageFences(ctx, query)
}

type quotaFenceQueryer interface {
	QueryRowContext(context.Context, string, ...any) *sql.Row
	QueryContext(context.Context, string, ...any) (*sql.Rows, error)
}

func loadQuotaFence(
	ctx context.Context,
	queryer quotaFenceQueryer,
	namespaceID, fenceID string,
) (quotareconciliation.Fence, error) {
	fence, err := scanQuotaFence(queryer.QueryRowContext(ctx, getQuotaFenceQuery, namespaceID, fenceID))
	if errors.Is(err, sql.ErrNoRows) {
		return quotareconciliation.Fence{}, quotareconciliation.ErrNotFound
	}
	if err != nil {
		return quotareconciliation.Fence{}, fmt.Errorf("get unknown-usage fence: %w", err)
	}
	bindings, err := loadQuotaFenceBindings(ctx, queryer, fence.ID)
	if err != nil {
		return quotareconciliation.Fence{}, err
	}
	if len(bindings) == 0 {
		return quotareconciliation.Fence{}, fmt.Errorf("stored unknown-usage fence has no bindings")
	}
	fence.Bindings = bindings
	if err := enrichUnknownDispatches(ctx, queryer, &fence); err != nil {
		return quotareconciliation.Fence{}, err
	}
	return fence, nil
}

func enrichUnknownDispatches(ctx context.Context, queryer quotaFenceQueryer, fence *quotareconciliation.Fence) error {
	rows, err := queryer.QueryContext(ctx, `SELECT dispatch_id,
  COALESCE(logical_model_id::text,''), COALESCE(backend_id::text,''),
  COALESCE(provider_id,''), COALESCE(provider_model_id,''), COALESCE(pricing_revision,0)
FROM usage_dispatches
WHERE namespace_id=$1 AND admission_id=$2 AND usage_state='unknown'
  AND corrects_dispatch_id IS NULL
ORDER BY dispatch_ordinal, dispatch_id`, fence.NamespaceID, fence.AdmissionID)
	if err != nil {
		return fmt.Errorf("read unknown dispatch dimensions: %w", err)
	}
	defer rows.Close()
	byID := make(map[string]*quotareconciliation.UnknownDispatch, len(fence.Unknown))
	for index := range fence.Unknown {
		byID[fence.Unknown[index].DispatchID] = &fence.Unknown[index]
	}
	seen := 0
	for rows.Next() {
		var id string
		var target quotareconciliation.UnknownDispatch
		if err := rows.Scan(&id, &target.ModelID, &target.BackendID, &target.ProviderID,
			&target.ProviderModelID, &target.PricingRevision); err != nil {
			return fmt.Errorf("scan unknown dispatch dimensions: %w", err)
		}
		value := byID[id]
		if value == nil {
			return errors.New("unknown dispatch evidence does not match immutable ledger")
		}
		value.ModelID, value.BackendID, value.ProviderID = target.ModelID, target.BackendID, target.ProviderID
		value.ProviderModelID, value.PricingRevision = target.ProviderModelID, target.PricingRevision
		seen++
	}
	if err := rows.Err(); err != nil {
		return err
	}
	if seen != len(fence.Unknown) {
		return errors.New("unknown dispatch evidence is incomplete")
	}
	return nil
}

func scanQuotaFence(scanner rowScanner) (quotareconciliation.Fence, error) {
	var fence quotareconciliation.Fence
	var revision int64
	var evidence, costs []byte
	var reconciliationID, strategy, actorID, reconciliationReason sql.NullString
	var reconciliationCreatedAt, resolvedAt sql.NullTime
	if err := scanner.Scan(&fence.ID, &fence.NamespaceID, &fence.AdmissionID,
		&fence.State, &revision, &fence.Reason, &evidence,
		&reconciliationID, &strategy, &actorID, &reconciliationReason,
		&reconciliationCreatedAt,
		&fence.CreatedAt, &fence.UpdatedAt, &resolvedAt,
		&fence.KnownCharge.InputTokens, &fence.KnownCharge.OutputTokens,
		&fence.KnownCharge.TotalTokens, &costs); err != nil {
		return quotareconciliation.Fence{}, err
	}
	if revision <= 0 {
		return quotareconciliation.Fence{}, errors.New("stored unknown-usage fence revision is invalid")
	}
	fence.Revision = uint64(revision)
	fence.CreatedAt, fence.UpdatedAt = fence.CreatedAt.UTC(), fence.UpdatedAt.UTC()
	if resolvedAt.Valid {
		value := resolvedAt.Time.UTC()
		fence.ResolvedAt = &value
	}
	if err := decodeQuotaFenceEvidence(evidence, &fence.Unknown); err != nil {
		return quotareconciliation.Fence{}, err
	}
	if err := decodeQuotaFenceCosts(costs, &fence.KnownCharge.Costs); err != nil {
		return quotareconciliation.Fence{}, err
	}
	if reconciliationID.Valid {
		if !reconciliationCreatedAt.Valid {
			return quotareconciliation.Fence{}, errors.New("stored unknown-usage reconciliation timestamp is missing")
		}
		fence.Reconciliation = &quotareconciliation.ReconciliationInfo{
			ID: reconciliationID.String, Strategy: quotareconciliation.Strategy(strategy.String),
			ActorID: actorID.String, Reason: reconciliationReason.String,
			CreatedAt: reconciliationCreatedAt.Time.UTC(), AppliedAt: fence.ResolvedAt,
		}
	}
	return fence, nil
}

func loadQuotaFenceBindings(
	ctx context.Context,
	queryer quotaFenceQueryer,
	fenceID string,
) ([]quotareconciliation.Binding, error) {
	rows, err := queryer.QueryContext(ctx, quotaFenceBindingsQuery, fenceID)
	if err != nil {
		return nil, fmt.Errorf("read unknown-usage fence bindings: %w", err)
	}
	defer rows.Close()
	bindings := make([]quotareconciliation.Binding, 0)
	for rows.Next() {
		var value quotareconciliation.Binding
		var window sql.NullInt64
		var period, timezone, currency sql.NullString
		var semanticsMatch bool
		if err := rows.Scan(&value.BindingID, &value.RuleID, &value.PolicyID,
			&value.Subject.Kind, &value.Subject.ID, &value.Metric, &value.Algorithm, &value.Enforcement,
			&value.AdmissionLimit, &value.MaximumDebit, &window, &period, &timezone, &currency,
			&value.UnknownDispatchCount, &value.CounterIncompleteCount, &semanticsMatch); err != nil {
			return nil, fmt.Errorf("scan unknown-usage fence binding: %w", err)
		}
		if !semanticsMatch {
			return nil, quotareconciliation.ErrReconciliationConflict
		}
		if window.Valid {
			value.Window = time.Duration(window.Int64) * time.Second
		}
		value.CalendarPeriod = quota.CalendarPeriod(period.String)
		value.Timezone, value.Currency = timezone.String, currency.String
		bindings = append(bindings, value)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate unknown-usage fence bindings: %w", err)
	}
	return bindings, nil
}

type storedFenceEvidence struct {
	EventID    string `json:"eventId"`
	Dispatches []struct {
		DispatchID     string `json:"dispatchId"`
		Reason         string `json:"reason"`
		EvidenceDigest string `json:"evidenceDigest"`
	} `json:"dispatches"`
}

func decodeQuotaFenceEvidence(payload []byte, destination *[]quotareconciliation.UnknownDispatch) error {
	var evidence storedFenceEvidence
	if err := json.Unmarshal(payload, &evidence); err != nil || evidence.EventID == "" || len(evidence.Dispatches) == 0 {
		return errors.New("stored unknown-usage fence evidence is invalid")
	}
	result := make([]quotareconciliation.UnknownDispatch, len(evidence.Dispatches))
	for index, dispatch := range evidence.Dispatches {
		result[index] = quotareconciliation.UnknownDispatch{
			DispatchID: dispatch.DispatchID, EvidenceDigest: dispatch.EvidenceDigest, Reason: dispatch.Reason,
		}
	}
	*destination = result
	return nil
}

func decodeQuotaFenceCosts(payload []byte, destination *[]quotareconciliation.Cost) error {
	var costs []struct {
		Currency       string `json:"currency"`
		KnownNumerator string `json:"knownNumerator"`
	}
	if err := json.Unmarshal(payload, &costs); err != nil {
		return errors.New("stored unknown-usage cost aggregate is invalid")
	}
	result := make([]quotareconciliation.Cost, 0, len(costs))
	for _, cost := range costs {
		result = append(result, quotareconciliation.Cost{Currency: cost.Currency, Numerator: cost.KnownNumerator})
	}
	*destination = result
	return nil
}

var _ quotareconciliation.Repository = (*Store)(nil)
