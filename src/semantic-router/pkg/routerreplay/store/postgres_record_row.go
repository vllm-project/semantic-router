package store

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

const postgresRecordSelectColumns = `
	id, timestamp, request_id, decision, decision_tier, decision_priority, category,
	original_model, selected_model, reasoning_mode,
	signals, projections, projection_scores, signal_confidences, signal_values, tool_trace, projection_trace, session_policy, route_diagnostics, learning, outcomes,
	request_body, response_body, response_status,
	from_cache, streaming, request_body_truncated, response_body_truncated,
	guardrails_enabled, jailbreak_enabled, pii_enabled,
	prompt, prompt_truncated, tool_definitions, tool_definitions_truncated,
	rag_enabled, rag_backend, rag_context_length, rag_similarity_score,
	hallucination_enabled, hallucination_detected, hallucination_confidence, hallucination_spans,
	prompt_tokens, cached_prompt_tokens, completion_tokens, total_tokens,
	actual_cost, baseline_cost, cost_savings, currency, baseline_model,
	session_id, turn_index, previous_response_id, conversation_id,
	cache_similarity, context_token_count
`

type postgresRowScanner interface {
	Scan(dest ...interface{}) error
}

type postgresInsertRecord struct {
	record                 Record
	signalsJSON            []byte
	projectionsJSON        []byte
	projectionScoresJSON   []byte
	signalConfidencesJSON  []byte
	signalValuesJSON       []byte
	toolTraceJSON          []byte
	projectionTraceJSON    []byte
	sessionPolicyJSON      []byte
	routeDiagnosticsJSON   []byte
	learningJSON           []byte
	outcomesJSON           []byte
	hallucinationSpansJSON []byte
}

type postgresRecordRow struct {
	record                 Record
	signalsJSON            []byte
	projectionsJSON        []byte
	projectionScoresJSON   []byte
	signalConfidencesJSON  []byte
	signalValuesJSON       []byte
	toolTraceJSON          []byte
	projectionTraceJSON    []byte
	sessionPolicyJSON      []byte
	routeDiagnosticsJSON   []byte
	learningJSON           []byte
	outcomesJSON           []byte
	hallucinationSpansJSON []byte
	promptTokens           sql.NullInt64
	cachedPromptTokens     sql.NullInt64
	completionTokens       sql.NullInt64
	totalTokens            sql.NullInt64
	actualCost             sql.NullFloat64
	baselineCost           sql.NullFloat64
	costSavings            sql.NullFloat64
	currency               sql.NullString
	baselineModel          sql.NullString
	sessionID              sql.NullString
	turnIndex              sql.NullInt64
	previousResponseID     sql.NullString
	conversationID         sql.NullString
}

func newPostgresInsertRecord(record Record) (postgresInsertRecord, error) {
	prepared, err := preparePostgresInsertRecord(record)
	if err != nil {
		return postgresInsertRecord{}, err
	}

	insertRecord := postgresInsertRecord{record: prepared}
	if err := marshalPostgresInsertJSON(prepared, &insertRecord); err != nil {
		return postgresInsertRecord{}, err
	}
	return insertRecord, nil
}

func marshalPostgresInsertJSON(record Record, out *postgresInsertRecord) error {
	fields := []struct {
		name    string
		target  *[]byte
		marshal func() ([]byte, error)
	}{
		{"signals", &out.signalsJSON, func() ([]byte, error) { return json.Marshal(record.Signals) }},
		{"projections", &out.projectionsJSON, func() ([]byte, error) { return json.Marshal(record.Projections) }},
		{"projection scores", &out.projectionScoresJSON, func() ([]byte, error) { return json.Marshal(record.ProjectionScores) }},
		{"signal confidences", &out.signalConfidencesJSON, func() ([]byte, error) { return json.Marshal(record.SignalConfidences) }},
		{"signal values", &out.signalValuesJSON, func() ([]byte, error) { return json.Marshal(record.SignalValues) }},
		{"tool trace", &out.toolTraceJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.ToolTrace) }},
		{"projection trace", &out.projectionTraceJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.ProjectionTrace) }},
		{"session policy", &out.sessionPolicyJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.SessionPolicy) }},
		{"route diagnostics", &out.routeDiagnosticsJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.RouteDiagnostics) }},
		{"learning diagnostics", &out.learningJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.Learning) }},
		{"outcomes", &out.outcomesJSON, func() ([]byte, error) { return marshalReplayOptionalJSON(record.Outcomes) }},
		{"hallucination spans", &out.hallucinationSpansJSON, func() ([]byte, error) { return json.Marshal(record.HallucinationSpans) }},
	}
	for _, field := range fields {
		encoded, err := field.marshal()
		if err != nil {
			return fmt.Errorf("failed to marshal %s: %w", field.name, err)
		}
		*field.target = encoded
	}
	return nil
}

func preparePostgresInsertRecord(record Record) (Record, error) {
	if record.ID == "" {
		id, err := generateID()
		if err != nil {
			return Record{}, err
		}
		record.ID = id
	}
	if record.Timestamp.IsZero() {
		record.Timestamp = time.Now().UTC()
	}
	return record, nil
}

func (record postgresInsertRecord) args() []interface{} {
	return []interface{}{
		record.record.ID,
		record.record.Timestamp,
		record.record.RequestID,
		record.record.Decision,
		record.record.DecisionTier,
		record.record.DecisionPriority,
		record.record.Category,
		record.record.OriginalModel,
		record.record.SelectedModel,
		record.record.ReasoningMode,
		record.signalsJSON,
		record.projectionsJSON,
		record.projectionScoresJSON,
		record.signalConfidencesJSON,
		record.signalValuesJSON,
		record.toolTraceJSON,
		record.projectionTraceJSON,
		record.sessionPolicyJSON,
		record.routeDiagnosticsJSON,
		record.learningJSON,
		record.outcomesJSON,
		record.record.RequestBody,
		record.record.ResponseBody,
		record.record.ResponseStatus,
		record.record.FromCache,
		record.record.Streaming,
		record.record.RequestBodyTruncated,
		record.record.ResponseBodyTruncated,
		record.record.GuardrailsEnabled,
		record.record.JailbreakEnabled,
		record.record.PIIEnabled,
		record.record.Prompt,
		record.record.PromptTruncated,
		record.record.ToolDefinitions,
		record.record.ToolDefinitionsTruncated,
		record.record.RAGEnabled,
		record.record.RAGBackend,
		record.record.RAGContextLength,
		record.record.RAGSimilarityScore,
		record.record.HallucinationEnabled,
		record.record.HallucinationDetected,
		record.record.HallucinationConfidence,
		record.hallucinationSpansJSON,
		nullableIntArg(record.record.PromptTokens),
		nullableIntArg(record.record.CachedPromptTokens),
		nullableIntArg(record.record.CompletionTokens),
		nullableIntArg(record.record.TotalTokens),
		nullableFloat64Arg(record.record.ActualCost),
		nullableFloat64Arg(record.record.BaselineCost),
		nullableFloat64Arg(record.record.CostSavings),
		nullableStringArg(record.record.Currency),
		nullableStringArg(record.record.BaselineModel),
		emptyStringSQL(record.record.SessionID),
		record.record.TurnIndex,
		emptyStringSQL(record.record.PreviousResponseID),
		emptyStringSQL(record.record.ConversationID),
		record.record.CacheSimilarity,
		record.record.ContextTokenCount,
	}
}

func scanPostgresRecord(scanner postgresRowScanner) (Record, error) {
	row := postgresRecordRow{}
	if err := scanner.Scan(row.scanDestinations()...); err != nil {
		return Record{}, err
	}
	return row.decode()
}

func scanPostgresRecordList(rows *sql.Rows) (_ []Record, err error) {
	defer func() {
		err = errors.Join(err, rows.Close())
	}()

	records := make([]Record, 0)
	for rows.Next() {
		record, scanErr := scanPostgresRecord(rows)
		if scanErr != nil {
			continue
		}
		records = append(records, record)
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		return nil, rowsErr
	}
	return records, nil
}

func (row *postgresRecordRow) scanDestinations() []interface{} {
	return []interface{}{
		&row.record.ID,
		&row.record.Timestamp,
		&row.record.RequestID,
		&row.record.Decision,
		&row.record.DecisionTier,
		&row.record.DecisionPriority,
		&row.record.Category,
		&row.record.OriginalModel,
		&row.record.SelectedModel,
		&row.record.ReasoningMode,
		&row.signalsJSON,
		&row.projectionsJSON,
		&row.projectionScoresJSON,
		&row.signalConfidencesJSON,
		&row.signalValuesJSON,
		&row.toolTraceJSON,
		&row.projectionTraceJSON,
		&row.sessionPolicyJSON,
		&row.routeDiagnosticsJSON,
		&row.learningJSON,
		&row.outcomesJSON,
		&row.record.RequestBody,
		&row.record.ResponseBody,
		&row.record.ResponseStatus,
		&row.record.FromCache,
		&row.record.Streaming,
		&row.record.RequestBodyTruncated,
		&row.record.ResponseBodyTruncated,
		&row.record.GuardrailsEnabled,
		&row.record.JailbreakEnabled,
		&row.record.PIIEnabled,
		&row.record.Prompt,
		&row.record.PromptTruncated,
		&row.record.ToolDefinitions,
		&row.record.ToolDefinitionsTruncated,
		&row.record.RAGEnabled,
		&row.record.RAGBackend,
		&row.record.RAGContextLength,
		&row.record.RAGSimilarityScore,
		&row.record.HallucinationEnabled,
		&row.record.HallucinationDetected,
		&row.record.HallucinationConfidence,
		&row.hallucinationSpansJSON,
		&row.promptTokens,
		&row.cachedPromptTokens,
		&row.completionTokens,
		&row.totalTokens,
		&row.actualCost,
		&row.baselineCost,
		&row.costSavings,
		&row.currency,
		&row.baselineModel,
		&row.sessionID,
		&row.turnIndex,
		&row.previousResponseID,
		&row.conversationID,
		&row.record.CacheSimilarity,
		&row.record.ContextTokenCount,
	}
}

func (row *postgresRecordRow) decode() (Record, error) {
	if err := row.unmarshalDecodedJSON(); err != nil {
		return Record{}, err
	}
	if len(row.hallucinationSpansJSON) > 0 {
		_ = json.Unmarshal(row.hallucinationSpansJSON, &row.record.HallucinationSpans)
	}
	assignUsageCostFields(
		&row.record,
		row.promptTokens,
		row.cachedPromptTokens,
		row.completionTokens,
		row.totalTokens,
		row.actualCost,
		row.baselineCost,
		row.costSavings,
		row.currency,
		row.baselineModel,
	)
	row.assignReplaySessionIdentifiers()
	return row.record, nil
}

func (row *postgresRecordRow) unmarshalDecodedJSON() error {
	if err := json.Unmarshal(row.signalsJSON, &row.record.Signals); err != nil {
		return fmt.Errorf("failed to unmarshal signals: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.projectionsJSON, &row.record.Projections); err != nil {
		return fmt.Errorf("failed to unmarshal projections: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.projectionScoresJSON, &row.record.ProjectionScores); err != nil {
		return fmt.Errorf("failed to unmarshal projection scores: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.signalConfidencesJSON, &row.record.SignalConfidences); err != nil {
		return fmt.Errorf("failed to unmarshal signal confidences: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.signalValuesJSON, &row.record.SignalValues); err != nil {
		return fmt.Errorf("failed to unmarshal signal values: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.toolTraceJSON, &row.record.ToolTrace); err != nil {
		return fmt.Errorf("failed to unmarshal tool trace: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.projectionTraceJSON, &row.record.ProjectionTrace); err != nil {
		return fmt.Errorf("failed to unmarshal projection trace: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.sessionPolicyJSON, &row.record.SessionPolicy); err != nil {
		return fmt.Errorf("failed to unmarshal session policy: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.routeDiagnosticsJSON, &row.record.RouteDiagnostics); err != nil {
		return fmt.Errorf("failed to unmarshal route diagnostics: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.learningJSON, &row.record.Learning); err != nil {
		return fmt.Errorf("failed to unmarshal learning diagnostics: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.outcomesJSON, &row.record.Outcomes); err != nil {
		return fmt.Errorf("failed to unmarshal outcomes: %w", err)
	}
	return nil
}

func (row *postgresRecordRow) assignReplaySessionIdentifiers() {
	if row.sessionID.Valid {
		row.record.SessionID = row.sessionID.String
	}
	if row.turnIndex.Valid {
		row.record.TurnIndex = int(row.turnIndex.Int64)
	}
	if row.previousResponseID.Valid {
		row.record.PreviousResponseID = row.previousResponseID.String
	}
	if row.conversationID.Valid {
		row.record.ConversationID = row.conversationID.String
	}
}
