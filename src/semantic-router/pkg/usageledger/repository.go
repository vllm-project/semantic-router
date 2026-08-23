package usageledger

import (
	"context"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/outcomefeedback"
)

var ErrLedgerCorrupt = errors.New("usage ledger invariant violation")

type BatchResult struct {
	Inserted         int
	Duplicate        int
	projectionEvents []TerminalEvent
}

type Store interface {
	PersistBatch(context.Context, []TerminalEvent) (BatchResult, error)
}

type PostgresStore struct {
	DB         *sql.DB
	Partitions StorageLifecycle
}

func (s PostgresStore) PersistBatch(ctx context.Context, events []TerminalEvent) (BatchResult, error) {
	if s.DB == nil {
		return BatchResult{}, fmt.Errorf("usage ledger database is required")
	}
	if len(events) == 0 {
		return BatchResult{}, nil
	}
	if len(events) > 1000 {
		return BatchResult{}, fmt.Errorf("usage ledger batch exceeds 1000 events")
	}
	type prepared struct {
		event     TerminalEvent
		aggregate EventAggregate
		digest    []byte
	}
	values := make([]prepared, 0, len(events))
	for index, event := range events {
		aggregate, err := event.Validate()
		if err != nil {
			return BatchResult{}, fmt.Errorf("event %d: %w", index, err)
		}
		digest, err := event.CanonicalDigest()
		if err != nil {
			return BatchResult{}, fmt.Errorf("event %d digest: %w", index, err)
		}
		values = append(values, prepared{event: event, aggregate: aggregate, digest: digest})
	}

	tx, err := s.DB.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		return BatchResult{}, fmt.Errorf("begin usage ledger batch: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	partitions := s.Partitions
	if partitions == nil {
		partitions, err = NewPostgresStorageLifecycle(s.DB, StorageLifecycleOptions{
			CreateAheadMonths: 1, MaintenanceInterval: time.Hour,
		})
		if err != nil {
			return BatchResult{}, err
		}
	}
	if err := partitions.LockWriterTx(ctx, tx); err != nil {
		return BatchResult{}, err
	}
	result := BatchResult{}
	lastUsed := make(map[string]time.Time)
	ensuredMonths := make(map[time.Time]struct{})
	for _, value := range values {
		inserted, projectable, err := persistTerminal(
			ctx, tx, value.event, value.aggregate, value.digest, partitions, ensuredMonths,
		)
		if err != nil {
			return BatchResult{}, err
		}
		if projectable {
			result.projectionEvents = append(result.projectionEvents, value.event)
		}
		if !inserted {
			result.Duplicate++
			continue
		}
		result.Inserted++
		if keyID := value.event.Principal.APIKeyID; keyID != "" && value.event.CompletedAt.After(lastUsed[keyID]) {
			lastUsed[keyID] = value.event.CompletedAt
		}
	}
	for keyID, usedAt := range lastUsed {
		if _, err := tx.ExecContext(ctx, `UPDATE access_api_keys
SET last_used_at = GREATEST(COALESCE(last_used_at, $2), $2)
WHERE id = $1`, keyID, usedAt); err != nil {
			return BatchResult{}, fmt.Errorf("update API-key last-used projection: %w", err)
		}
	}
	if err := tx.Commit(); err != nil {
		return BatchResult{}, fmt.Errorf("commit usage ledger batch: %w", err)
	}
	return result, nil
}

func persistTerminal(
	ctx context.Context,
	tx *sql.Tx,
	event TerminalEvent,
	aggregate EventAggregate,
	digest []byte,
	partitions StorageLifecycle,
	ensuredMonths map[time.Time]struct{},
) (bool, bool, error) {
	settlementState := "settled"
	eventKind := "actual"
	if aggregate.UsageState == UsageUnknown {
		settlementState = "unknown"
		eventKind = "unknown"
	}
	partitionDate := event.OccurredAt.UTC().Format("2006-01-02")
	result, persistTerminalErr := tx.ExecContext(ctx, `INSERT INTO usage_settlements (
  namespace_id, admission_id, state, canonical_usage_digest, settled_at, event_partition_date
) VALUES ($1, $2, $3, $4, $5, $6)
ON CONFLICT (namespace_id, admission_id) DO NOTHING`,
		event.NamespaceID, event.AdmissionID, settlementState, digest, event.CompletedAt, partitionDate,
	)
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("insert usage settlement: %w", persistTerminalErr)
	}
	rows, persistTerminalErr := result.RowsAffected()
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("inspect usage settlement insert: %w", persistTerminalErr)
	}
	if rows == 0 {
		var existingDigest []byte
		var eventRetained bool
		var eventExists bool
		if err := tx.QueryRowContext(ctx, `SELECT canonical_usage_digest,event_retained,
  EXISTS (SELECT 1 FROM usage_events e
    WHERE e.namespace_id = s.namespace_id
      AND e.event_date = s.event_partition_date
      AND e.admission_id = s.admission_id)
FROM usage_settlements s
WHERE namespace_id = $1 AND admission_id = $2
FOR UPDATE`, event.NamespaceID, event.AdmissionID).Scan(&existingDigest, &eventRetained, &eventExists); err != nil {
			return false, false, fmt.Errorf("read duplicate usage settlement: %w", err)
		}
		if !equalDigest(existingDigest, digest) {
			return false, false, fmt.Errorf("%w: admission %q already has a different canonical usage digest", ErrConflict, event.AdmissionID)
		}
		if eventRetained != eventExists {
			return false, false, fmt.Errorf("%w: settlement %q raw-retention state does not match its immutable event", ErrLedgerCorrupt, event.AdmissionID)
		}
		return false, eventRetained, nil
	}
	month := usageMonth(event.OccurredAt)
	if _, exists := ensuredMonths[month]; !exists {
		if err := partitions.EnsureTx(ctx, tx, []time.Time{event.OccurredAt}); err != nil {
			return false, false, err
		}
		ensuredMonths[month] = struct{}{}
	}

	costs, persistTerminalErr := json.Marshal(internalCostRows(aggregate.Costs))
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("encode usage costs: %w", persistTerminalErr)
	}
	metadata, persistTerminalErr := json.Marshal(eventMetadata(event))
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("encode request metadata: %w", persistTerminalErr)
	}
	servedTotal, persistTerminalErr := aggregate.ServedInputTokens.Add(aggregate.ServedOutputTokens)
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("served token total: %w", persistTerminalErr)
	}
	total, persistTerminalErr := aggregate.InputTokens.Add(aggregate.OutputTokens)
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("backend token total: %w", persistTerminalErr)
	}
	_, persistTerminalErr = tx.ExecContext(ctx, `INSERT INTO usage_events (
  namespace_id, admission_id, event_date, event_id, event_kind, external_request_id,
  protocol, path, api_key_id, credential_id, user_id, team_id, entrypoint_id,
  entrypoint_rule_id, recipe_id, routing_revision, status_code, error_code,
  input_tokens, output_tokens, total_tokens, served_input_tokens, served_output_tokens,
  served_total_tokens, latency_ms, ttft_ms, usage_state, costs, request_metadata,
  occurred_at, ingested_at
) VALUES (
  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
  $19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,clock_timestamp()
)`,
		event.NamespaceID, event.AdmissionID, partitionDate, event.EventID, eventKind,
		nullString(event.ExternalRequestID), event.Protocol, event.Path,
		nullString(event.Principal.APIKeyID), nullString(event.Principal.CredentialID),
		nullString(event.Principal.UserID), nullString(event.Principal.TeamID),
		nullString(event.Routing.EntrypointID), nullString(event.Routing.EntrypointRuleID),
		nullString(event.Routing.RecipeID), nullInt64(event.Routing.RoutingRevision),
		event.StatusCode, nullString(event.ErrorCode), aggregate.InputTokens.String(),
		aggregate.OutputTokens.String(), total.String(), aggregate.ServedInputTokens.String(),
		aggregate.ServedOutputTokens.String(), servedTotal.String(), event.LatencyMilliseconds,
		nullInt64Pointer(event.TTFTMilliseconds), string(aggregate.UsageState), costs, metadata,
		event.OccurredAt,
	)
	if persistTerminalErr != nil {
		return false, false, fmt.Errorf("insert immutable usage event: %w", persistTerminalErr)
	}
	for _, dispatch := range event.Dispatches {
		if err := persistDispatch(ctx, tx, event, partitionDate, dispatch); err != nil {
			return false, false, err
		}
	}
	if event.ReplayID != "" {
		if err := persistInferenceReplay(ctx, tx, event, partitionDate); err != nil {
			return false, false, err
		}
	}
	if event.Fence != nil {
		if err := persistFence(ctx, tx, event, *event.Fence); err != nil {
			return false, false, err
		}
	}
	return true, true, nil
}

func persistInferenceReplay(ctx context.Context, tx *sql.Tx, event TerminalEvent, partitionDate string) error {
	routing := outcomefeedback.ReplayRoutingContext{
		RecipeID: event.Routing.RecipeID, RecipeName: event.Routing.RecipeName,
		RecipeRevision: event.Routing.RecipeRevision,
	}
	modelsByKey := make(map[string]outcomefeedback.ServedModel)
	for _, dispatch := range event.Dispatches {
		if routing.DecisionID == "" && dispatch.DecisionID != "" {
			routing.DecisionID = dispatch.DecisionID
			routing.DecisionName = dispatch.DecisionName
			routing.DecisionTier = dispatch.DecisionTier
		}
		if !dispatchActuallyServed(dispatch) || dispatch.ModelID == "" ||
			dispatch.ModelName == "" || dispatch.ModelRevision <= 0 {
			continue
		}
		model := outcomefeedback.ServedModel{
			ID: dispatch.ModelID, Name: dispatch.ModelName, Revision: dispatch.ModelRevision,
		}
		modelsByKey[model.ID+"\x00"+model.Name+"\x00"+fmt.Sprint(model.Revision)] = model
	}
	keys := make([]string, 0, len(modelsByKey))
	for key := range modelsByKey {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	models := make([]outcomefeedback.ServedModel, 0, len(keys))
	for _, key := range keys {
		models = append(models, modelsByKey[key])
	}
	routingJSON, err := json.Marshal(routing)
	if err != nil {
		return fmt.Errorf("encode inference replay routing context: %w", err)
	}
	modelsJSON, err := json.Marshal(models)
	if err != nil {
		return fmt.Errorf("encode inference replay served models: %w", err)
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO inference_replays (
  namespace_id, replay_id, api_key_id, user_id, team_id, event_date, event_id,
  routing_context, served_models, created_at
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`,
		event.NamespaceID, event.ReplayID, event.Principal.APIKeyID,
		nullString(event.Principal.UserID), nullString(event.Principal.TeamID),
		partitionDate, event.EventID, routingJSON, modelsJSON, event.CompletedAt,
	)
	if err != nil {
		return fmt.Errorf("insert durable inference replay ownership: %w", err)
	}
	return nil
}

func dispatchActuallyServed(dispatch Dispatch) bool {
	if dispatch.UsageState == UsageKnownActual {
		return true
	}
	for _, attempt := range dispatch.Attempts {
		if attempt.StatusCode >= 200 && attempt.StatusCode < 400 {
			return true
		}
	}
	return dispatch.CacheState == "hit"
}

func persistDispatch(ctx context.Context, tx *sql.Tx, event TerminalEvent, partitionDate string, d Dispatch) error {
	var cost any
	if d.Cost.State == CostComplete {
		cost = d.Cost.Numerator
	}
	evidence, _ := hex.DecodeString(d.EvidenceDigest)
	_, err := tx.ExecContext(ctx, `INSERT INTO usage_dispatches (
  namespace_id, event_date, event_id, admission_id, dispatch_id, parent_dispatch_id,
  dispatch_ordinal, attempt_count, dispatch_type, logical_model_id, model_revision,
  backend_id, provider_id, provider_model_id, pricing_revision, input_tokens,
  cache_read_tokens, cache_write_tokens, output_tokens, usage_state, cost_numerator,
  currency, evidence_digest, started_at, completed_at
) VALUES (
  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,
  $21,$22,$23,$24,$25
)`, event.NamespaceID, partitionDate, event.EventID, event.AdmissionID, d.DispatchID,
		nullString(d.ParentDispatchID), d.Ordinal, len(d.Attempts), d.DispatchType,
		nullString(d.ModelID), nullInt64(d.ModelRevision), nullString(d.BackendID),
		nullString(d.ProviderID), nullString(d.ProviderModelID), nullInt64(d.PricingRevision),
		d.InputTokens, d.CacheReadTokens, d.CacheWriteTokens, d.OutputTokens,
		string(d.UsageState), cost, d.Cost.Currency, evidence, d.StartedAt, d.CompletedAt,
	)
	if err != nil {
		return fmt.Errorf("insert usage dispatch %q: %w", d.DispatchID, err)
	}
	for _, attempt := range d.Attempts {
		if _, err := tx.ExecContext(ctx, `INSERT INTO usage_dispatch_attempts (
  namespace_id, event_date, event_id, dispatch_id, admission_id, attempt_id,
  attempt_ordinal, backend_id, provider_id, state, status_code, error_code,
  started_at, completed_at
) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)`,
			event.NamespaceID, partitionDate, event.EventID, d.DispatchID, event.AdmissionID,
			attempt.AttemptID, attempt.Ordinal, nullString(attempt.BackendID), nullString(attempt.ProviderID),
			string(attempt.State), nullStatus(attempt.StatusCode), nullString(attempt.ErrorCode),
			attempt.StartedAt, attempt.CompletedAt,
		); err != nil {
			return fmt.Errorf("insert usage attempt %q: %w", attempt.AttemptID, err)
		}
	}
	return nil
}

func persistFence(ctx context.Context, tx *sql.Tx, event TerminalEvent, fence UnknownFence) error {
	type unknownDispatchEvidence struct {
		DispatchID     string `json:"dispatchId"`
		Reason         string `json:"reason"`
		EvidenceDigest string `json:"evidenceDigest"`
	}
	unknown := make([]unknownDispatchEvidence, 0)
	for _, dispatch := range event.Dispatches {
		if dispatch.UsageState == UsageUnknown {
			unknown = append(unknown, unknownDispatchEvidence{
				DispatchID: dispatch.DispatchID, Reason: dispatch.UnknownReason,
				EvidenceDigest: dispatch.EvidenceDigest,
			})
		}
	}
	evidence, err := json.Marshal(struct {
		EventID       string                    `json:"eventId"`
		EvidenceState EvidenceState             `json:"evidenceState"`
		Dispatches    []unknownDispatchEvidence `json:"dispatches"`
	}{EventID: event.EventID, EvidenceState: event.EvidenceState, Dispatches: unknown})
	if err != nil {
		return fmt.Errorf("encode unknown fence evidence: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_fences (
  id, namespace_id, admission_id, reason, evidence, state
) VALUES ($1,$2,$3,$4,$5,'open')`, fence.FenceID, event.NamespaceID, event.AdmissionID, fence.Reason, evidence); err != nil {
		return fmt.Errorf("insert unknown-usage fence: %w", err)
	}
	unknownDispatchCount := 0
	for _, dispatch := range event.Dispatches {
		if dispatch.UsageState == UsageUnknown {
			unknownDispatchCount++
		}
	}
	for _, binding := range fence.Bindings {
		result, err := tx.ExecContext(ctx, `INSERT INTO unknown_usage_fence_bindings (
  fence_id, binding_id, rule_id, admission_limit, maximum_debit,
  metric, algorithm, enforcement, window_seconds, calendar_period, timezone, currency,
  unknown_dispatch_count, counter_incomplete_count
)
SELECT $1, b.id, r.id, $4, $5, r.metric, r.algorithm, r.enforcement, r.window_seconds,
  r.calendar_period, r.timezone,
  CASE WHEN r.metric='cost' THEN n.billing_currency ELSE NULL END, $6, $8
FROM rate_limit_bindings b
JOIN rate_limit_rules r ON r.policy_id=b.policy_id AND r.id=$3
JOIN access_namespaces n ON n.id=b.namespace_id
WHERE b.id=$2 AND b.namespace_id=$7 AND r.accounting='response_actual'`,
			fence.FenceID, binding.BindingID, binding.RuleID,
			nullString(binding.AdmissionLimit), nullString(binding.MaximumDebit),
			unknownDispatchCount, event.NamespaceID, len(event.Dispatches))
		if err != nil {
			return fmt.Errorf("insert unknown-fence binding: %w", err)
		}
		rows, err := result.RowsAffected()
		if err != nil || rows != 1 {
			return fmt.Errorf("insert unknown-fence binding: referenced counter is unavailable")
		}
	}
	return nil
}

type storedCost struct {
	Currency             string `json:"currency"`
	KnownNumerator       string `json:"knownNumerator"`
	KnownDispatches      string `json:"knownDispatches"`
	IncompleteDispatches string `json:"incompleteDispatches"`
}

func internalCostRows(values []CostAggregate) []storedCost {
	result := make([]storedCost, 0, len(values))
	for _, value := range values {
		result = append(result, storedCost{
			Currency: value.Currency, KnownNumerator: value.KnownNumerator.String(),
			KnownDispatches: value.KnownDispatches.String(), IncompleteDispatches: value.IncompleteDispatches.String(),
		})
	}
	return result
}

type safeEventMetadata struct {
	CompletedAt        time.Time         `json:"completedAt"`
	Stream             bool              `json:"stream"`
	ToolCall           bool              `json:"toolCall"`
	CacheState         string            `json:"cacheState,omitempty"`
	PrincipalSnapshots PrincipalSnapshot `json:"principalSnapshots"`
	RoutingSnapshots   RoutingSnapshot   `json:"routingSnapshots"`
	ServedInputKnown   bool              `json:"servedInputKnown"`
	ServedOutputKnown  bool              `json:"servedOutputKnown"`
	QuotaReceipts      []QuotaReceipt    `json:"quotaReceipts,omitempty"`
	Metadata           map[string]string `json:"metadata,omitempty"`
}

func eventMetadata(event TerminalEvent) safeEventMetadata {
	return safeEventMetadata{
		CompletedAt: event.CompletedAt, Stream: event.Stream, ToolCall: event.ToolCall,
		CacheState: event.CacheState, PrincipalSnapshots: event.Principal,
		RoutingSnapshots: event.Routing, ServedInputKnown: event.Served.InputKnown,
		ServedOutputKnown: event.Served.OutputKnown, QuotaReceipts: event.QuotaReceipts,
		Metadata: event.Metadata,
	}
}

func nullString(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func nullInt64(value int64) any {
	if value == 0 {
		return nil
	}
	return value
}

func nullInt64Pointer(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

func nullStatus(value int) any {
	if value == 0 {
		return nil
	}
	return value
}
