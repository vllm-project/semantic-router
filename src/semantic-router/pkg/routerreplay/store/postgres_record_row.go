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
	original_model, selected_model, reasoning_mode, confidence_score, selection_method,
	session_id, turn_index, previous_model, cache_warmth_estimate,
	signals, projections, projection_scores, signal_confidences, signal_values, tool_trace,
	request_body, response_body, response_status,
	from_cache, streaming, request_body_truncated, response_body_truncated,
	guardrails_enabled, jailbreak_enabled, pii_enabled,
	rag_enabled, rag_backend, rag_context_length, rag_similarity_score,
	hallucination_enabled, hallucination_detected, hallucination_confidence, hallucination_spans,
	prompt_tokens, completion_tokens, total_tokens,
	actual_cost, baseline_cost, cost_savings, currency, baseline_model
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
	hallucinationSpansJSON []byte
	promptTokens           sql.NullInt64
	completionTokens       sql.NullInt64
	totalTokens            sql.NullInt64
	actualCost             sql.NullFloat64
	baselineCost           sql.NullFloat64
	costSavings            sql.NullFloat64
	currency               sql.NullString
	baselineModel          sql.NullString
}

func newPostgresInsertRecord(record Record) (postgresInsertRecord, error) {
	prepared, err := preparePostgresInsertRecord(record)
	if err != nil {
		return postgresInsertRecord{}, err
	}

	signalsJSON, err := json.Marshal(prepared.Signals)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal signals: %w", err)
	}
	projectionsJSON, err := json.Marshal(prepared.Projections)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal projections: %w", err)
	}
	projectionScoresJSON, err := json.Marshal(prepared.ProjectionScores)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal projection scores: %w", err)
	}
	signalConfidencesJSON, err := json.Marshal(prepared.SignalConfidences)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal signal confidences: %w", err)
	}
	signalValuesJSON, err := json.Marshal(prepared.SignalValues)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal signal values: %w", err)
	}
	toolTraceJSON, err := marshalReplayOptionalJSON(prepared.ToolTrace)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal tool trace: %w", err)
	}
	hallucinationSpansJSON, err := json.Marshal(prepared.HallucinationSpans)
	if err != nil {
		return postgresInsertRecord{}, fmt.Errorf("failed to marshal hallucination spans: %w", err)
	}

	return postgresInsertRecord{
		record:                 prepared,
		signalsJSON:            signalsJSON,
		projectionsJSON:        projectionsJSON,
		projectionScoresJSON:   projectionScoresJSON,
		signalConfidencesJSON:  signalConfidencesJSON,
		signalValuesJSON:       signalValuesJSON,
		toolTraceJSON:          toolTraceJSON,
		hallucinationSpansJSON: hallucinationSpansJSON,
	}, nil
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
		record.record.ConfidenceScore,
		record.record.SelectionMethod,
		record.record.SessionID,
		record.record.TurnIndex,
		record.record.PreviousModel,
		record.record.CacheWarmthEstimate,
		record.signalsJSON,
		record.projectionsJSON,
		record.projectionScoresJSON,
		record.signalConfidencesJSON,
		record.signalValuesJSON,
		record.toolTraceJSON,
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
		record.record.RAGEnabled,
		record.record.RAGBackend,
		record.record.RAGContextLength,
		record.record.RAGSimilarityScore,
		record.record.HallucinationEnabled,
		record.record.HallucinationDetected,
		record.record.HallucinationConfidence,
		record.hallucinationSpansJSON,
		nullableIntArg(record.record.PromptTokens),
		nullableIntArg(record.record.CompletionTokens),
		nullableIntArg(record.record.TotalTokens),
		nullableFloat64Arg(record.record.ActualCost),
		nullableFloat64Arg(record.record.BaselineCost),
		nullableFloat64Arg(record.record.CostSavings),
		nullableStringArg(record.record.Currency),
		nullableStringArg(record.record.BaselineModel),
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
		&row.record.ConfidenceScore,
		&row.record.SelectionMethod,
		&row.record.SessionID,
		&row.record.TurnIndex,
		&row.record.PreviousModel,
		&row.record.CacheWarmthEstimate,
		&row.signalsJSON,
		&row.projectionsJSON,
		&row.projectionScoresJSON,
		&row.signalConfidencesJSON,
		&row.signalValuesJSON,
		&row.toolTraceJSON,
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
		&row.record.RAGEnabled,
		&row.record.RAGBackend,
		&row.record.RAGContextLength,
		&row.record.RAGSimilarityScore,
		&row.record.HallucinationEnabled,
		&row.record.HallucinationDetected,
		&row.record.HallucinationConfidence,
		&row.hallucinationSpansJSON,
		&row.promptTokens,
		&row.completionTokens,
		&row.totalTokens,
		&row.actualCost,
		&row.baselineCost,
		&row.costSavings,
		&row.currency,
		&row.baselineModel,
	}
}

func (row *postgresRecordRow) decode() (Record, error) {
	if err := json.Unmarshal(row.signalsJSON, &row.record.Signals); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal signals: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.projectionsJSON, &row.record.Projections); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal projections: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.projectionScoresJSON, &row.record.ProjectionScores); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal projection scores: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.signalConfidencesJSON, &row.record.SignalConfidences); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal signal confidences: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.signalValuesJSON, &row.record.SignalValues); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal signal values: %w", err)
	}
	if err := unmarshalReplayOptionalJSON(row.toolTraceJSON, &row.record.ToolTrace); err != nil {
		return Record{}, fmt.Errorf("failed to unmarshal tool trace: %w", err)
	}
	if len(row.hallucinationSpansJSON) > 0 {
		_ = json.Unmarshal(row.hallucinationSpansJSON, &row.record.HallucinationSpans)
	}
	assignUsageCostFields(
		&row.record,
		row.promptTokens,
		row.completionTokens,
		row.totalTokens,
		row.actualCost,
		row.baselineCost,
		row.costSavings,
		row.currency,
		row.baselineModel,
	)
	return row.record, nil
}
