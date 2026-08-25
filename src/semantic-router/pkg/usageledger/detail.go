package usageledger

import (
	"context"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"
)

var ErrNotFound = errors.New("usage record not found")

type AttemptDetail struct {
	AttemptID   string     `json:"attemptId"`
	Ordinal     int        `json:"ordinal"`
	BackendID   string     `json:"backendId,omitempty"`
	ProviderID  string     `json:"providerId,omitempty"`
	State       UsageState `json:"state"`
	StatusCode  int        `json:"statusCode,omitempty"`
	ErrorCode   string     `json:"errorCode,omitempty"`
	StartedAt   time.Time  `json:"startedAt"`
	CompletedAt time.Time  `json:"completedAt"`
}

type DispatchDetail struct {
	DispatchID       string          `json:"dispatchId"`
	ParentDispatchID string          `json:"parentDispatchId,omitempty"`
	Ordinal          int             `json:"ordinal"`
	DispatchType     string          `json:"dispatchType"`
	ModelID          string          `json:"modelId,omitempty"`
	ModelRevision    int64           `json:"modelRevision,omitempty"`
	BackendID        string          `json:"backendId,omitempty"`
	ProviderID       string          `json:"providerId,omitempty"`
	ProviderModelID  string          `json:"providerModelId,omitempty"`
	PricingRevision  int64           `json:"pricingRevision,omitempty"`
	InputTokens      string          `json:"inputTokens"`
	CacheReadTokens  string          `json:"cacheReadTokens"`
	CacheWriteTokens string          `json:"cacheWriteTokens"`
	OutputTokens     string          `json:"outputTokens"`
	UsageState       UsageState      `json:"usageState"`
	Cost             CostSummary     `json:"cost"`
	EvidenceDigest   string          `json:"evidenceDigest"`
	StartedAt        time.Time       `json:"startedAt"`
	CompletedAt      time.Time       `json:"completedAt"`
	Attempts         []AttemptDetail `json:"attempts"`
}

type RequestDetail struct {
	Request       RequestLog       `json:"request"`
	Routing       RoutingSnapshot  `json:"routing"`
	QuotaReceipts []QuotaReceipt   `json:"quotaReceipts"`
	Dispatches    []DispatchDetail `json:"dispatches,omitempty"`
}

func (q PostgresQueries) RequestDetail(
	ctx context.Context,
	namespaceID, admissionID string,
	visibility QueryVisibility,
) (RequestDetail, error) {
	if q.DB == nil {
		return RequestDetail{}, fmt.Errorf("usage query database is required")
	}
	if err := requireUUID("namespace ID", namespaceID, false); err != nil {
		return RequestDetail{}, invalidQuery(err)
	}
	if err := boundedIdentifier("admission ID", admissionID, 256); err != nil {
		return RequestDetail{}, invalidQuery(err)
	}
	if err := validateQueryVisibility(visibility); err != nil {
		return RequestDetail{}, err
	}
	var eventDate time.Time
	var eventRetained bool
	if err := q.DB.QueryRowContext(ctx, `SELECT event_partition_date,event_retained
FROM usage_settlements
WHERE namespace_id=$1 AND admission_id=$2`, namespaceID, admissionID).
		Scan(&eventDate, &eventRetained); errors.Is(err, sql.ErrNoRows) {
		return RequestDetail{}, ErrNotFound
	} else if err != nil {
		return RequestDetail{}, fmt.Errorf("resolve usage event partition: %w", err)
	}
	// Rollups and the permanent settlement tombstone survive raw retention, but
	// request detail deliberately does not synthesize a partial raw record.
	if !eventRetained {
		return RequestDetail{}, ErrNotFound
	}
	clauses := []string{
		"namespace_id = $1", "event_date = $2::date", "admission_id = $3",
		"event_kind IN ('actual','unknown')",
	}
	eventPartitionDate := eventDate.UTC().Format("2006-01-02")
	args := []any{namespaceID, eventPartitionDate, admissionID}
	appendRawLogVisibility(&clauses, &args, visibility)
	where := strings.Join(clauses, " AND ")
	// #nosec G201 -- predicate clauses are assembled from fixed fields and parameter placeholders only.
	statement := fmt.Sprintf(`SELECT admission_id, event_id::text, occurred_at, protocol, path,
  status_code, COALESCE(error_code,''), usage_state, input_tokens::text, output_tokens::text,
  latency_ms, ttft_ms, COALESCE(api_key_id::text,''), COALESCE(user_id::text,''), COALESCE(team_id::text,''),
  COALESCE(entrypoint_id::text,''), COALESCE(recipe_id::text,''), costs, request_metadata
FROM usage_events WHERE %s`, where)
	row := q.DB.QueryRowContext(ctx, statement, args...)
	request, metadata, err := scanRequestLogWithMetadata(row)
	if errors.Is(err, sql.ErrNoRows) {
		return RequestDetail{}, ErrNotFound
	}
	if err != nil {
		return RequestDetail{}, err
	}
	dispatches, err := q.loadDispatchDetails(
		ctx, namespaceID, eventPartitionDate, request.EventID, admissionID,
	)
	if err != nil {
		return RequestDetail{}, err
	}
	return RequestDetail{
		Request: request, Routing: metadata.RoutingSnapshots,
		QuotaReceipts: metadata.QuotaReceipts, Dispatches: dispatches,
	}, nil
}

func (q PostgresQueries) loadDispatchDetails(
	ctx context.Context,
	namespaceID string,
	eventDate string,
	eventID, admissionID string,
) ([]DispatchDetail, error) {
	rows, loadDispatchDetailsErr := q.DB.QueryContext(ctx, `SELECT dispatch_id, COALESCE(parent_dispatch_id,''),
  dispatch_ordinal, dispatch_type, COALESCE(logical_model_id::text,''), COALESCE(model_revision,0),
  COALESCE(backend_id::text,''), COALESCE(provider_id,''), COALESCE(provider_model_id,''),
  COALESCE(pricing_revision,0), input_tokens::text, cache_read_tokens::text,
  cache_write_tokens::text, output_tokens::text, usage_state,
  COALESCE(cost_numerator::text,''), COALESCE(currency,''), evidence_digest,
  started_at, completed_at
FROM usage_dispatches
WHERE namespace_id = $1 AND event_date = $2::date AND event_id = $3::uuid
  AND admission_id = $4 AND corrects_dispatch_id IS NULL
ORDER BY dispatch_ordinal`, namespaceID, eventDate, eventID, admissionID)
	if loadDispatchDetailsErr != nil {
		return nil, fmt.Errorf("read usage dispatches: %w", loadDispatchDetailsErr)
	}
	result := make([]DispatchDetail, 0)
	for rows.Next() {
		var dispatch DispatchDetail
		var numerator, currency string
		var evidence []byte
		if err := rows.Scan(&dispatch.DispatchID, &dispatch.ParentDispatchID, &dispatch.Ordinal,
			&dispatch.DispatchType, &dispatch.ModelID, &dispatch.ModelRevision, &dispatch.BackendID,
			&dispatch.ProviderID, &dispatch.ProviderModelID, &dispatch.PricingRevision,
			&dispatch.InputTokens, &dispatch.CacheReadTokens, &dispatch.CacheWriteTokens,
			&dispatch.OutputTokens, &dispatch.UsageState, &numerator, &currency, &evidence,
			&dispatch.StartedAt, &dispatch.CompletedAt); err != nil {
			return nil, fmt.Errorf("scan usage dispatch: %w", err)
		}
		dispatch.EvidenceDigest = hex.EncodeToString(evidence)
		if numerator == "" {
			dispatch.Cost = CostSummary{Currency: currency, KnownAmount: "0", Completeness: CompletenessUnknown, KnownDispatches: "0", IncompleteDispatches: "1"}
		} else {
			cost, err := parseCostAggregate(currency, numerator, "1", "0")
			if err != nil {
				return nil, err
			}
			dispatch.Cost = publicCosts([]CostAggregate{cost})[0]
		}
		result = append(result, dispatch)
	}
	if err := rows.Err(); err != nil {
		rows.Close()
		return nil, fmt.Errorf("iterate usage dispatches: %w", err)
	}
	if err := rows.Close(); err != nil {
		return nil, fmt.Errorf("close usage dispatch rows: %w", err)
	}
	attempts, loadDispatchDetailsErr := q.loadAttemptDetails(ctx, namespaceID, eventDate, eventID, admissionID)
	if loadDispatchDetailsErr != nil {
		return nil, loadDispatchDetailsErr
	}
	for index := range result {
		result[index].Attempts = attempts[result[index].DispatchID]
		if len(result[index].Attempts) == 0 {
			return nil, fmt.Errorf("%w: dispatch %q has no attempt ledger", ErrLedgerCorrupt, result[index].DispatchID)
		}
	}
	return result, nil
}

func (q PostgresQueries) loadAttemptDetails(
	ctx context.Context,
	namespaceID string,
	eventDate string,
	eventID, admissionID string,
) (map[string][]AttemptDetail, error) {
	rows, err := q.DB.QueryContext(ctx, `SELECT dispatch_id, attempt_id, attempt_ordinal,
  COALESCE(backend_id::text,''), COALESCE(provider_id,''), state,
  COALESCE(status_code,0), COALESCE(error_code,''), started_at, completed_at
FROM usage_dispatch_attempts
WHERE namespace_id = $1 AND event_date = $2::date AND event_id = $3::uuid
  AND admission_id = $4
ORDER BY dispatch_id, attempt_ordinal`, namespaceID, eventDate, eventID, admissionID)
	if err != nil {
		return nil, fmt.Errorf("read usage attempts: %w", err)
	}
	defer rows.Close()
	result := make(map[string][]AttemptDetail)
	for rows.Next() {
		var dispatchID string
		var attempt AttemptDetail
		if err := rows.Scan(&dispatchID, &attempt.AttemptID, &attempt.Ordinal, &attempt.BackendID,
			&attempt.ProviderID, &attempt.State, &attempt.StatusCode, &attempt.ErrorCode,
			&attempt.StartedAt, &attempt.CompletedAt); err != nil {
			return nil, fmt.Errorf("scan usage attempt: %w", err)
		}
		result[dispatchID] = append(result[dispatchID], attempt)
	}
	return result, rows.Err()
}
