package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// QueryPage counts and reads one snapshot, so concurrent inserts cannot make
// the page disagree with its total. Only the requested page loads full bodies.
func (p *PostgresStore) QueryPage(ctx context.Context, filters QueryFilters, limit, offset int, details bool) (RecordPage, error) {
	if limit <= 0 || offset < 0 {
		return RecordPage{}, errors.New("replay query requires a positive limit and nonnegative offset")
	}
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return RecordPage{}, err
	}
	defer release()
	tx, err := p.db.BeginTx(ctx, &sql.TxOptions{ReadOnly: true, Isolation: sql.LevelRepeatableRead})
	if err != nil {
		return RecordPage{}, err
	}
	defer func() { _ = tx.Rollback() }()
	where, args := postgresReplayWhere(filters)
	page := RecordPage{Records: []Record{}}
	//nolint:gosec // tableName is validated; filters are parameters
	if err = tx.QueryRowContext(ctx, "SELECT COUNT(*) FROM "+p.tableName+where, args...).Scan(&page.Total); err != nil {
		return RecordPage{}, err
	}
	page.Offset = min(offset, page.Total)
	columns := postgresRecordSelectColumns
	if !details {
		columns = postgresSummarySelectColumns()
	}
	//nolint:gosec // tableName is validated; filters and pagination are parameters
	query := fmt.Sprintf("SELECT %s FROM %s%s ORDER BY timestamp DESC, id DESC LIMIT $%d OFFSET $%d", columns, p.tableName, where, len(args)+1, len(args)+2)
	args = append(args, limit, page.Offset)
	rows, err := tx.QueryContext(ctx, query, args...)
	if err != nil {
		return RecordPage{}, err
	}
	page.Records, err = scanPostgresRecordList(rows)
	if err != nil {
		return RecordPage{}, err
	}
	if err = tx.Commit(); err != nil {
		return RecordPage{}, err
	}
	return page, nil
}

func postgresReplayWhere(filters QueryFilters) (string, []interface{}) {
	var clauses []string
	var args []interface{}
	add := func(clause string, value string) {
		args = append(args, value)
		clauses = append(clauses, strings.ReplaceAll(clause, "?", fmt.Sprintf("$%d", len(args))))
	}
	if filters.RecipeSet || filters.Recipe != "" {
		add("COALESCE(recipe, '') = ?", filters.Recipe)
	}
	for _, field := range []struct{ column, value string }{
		{"decision", filters.Decision}, {"session_id", filters.SessionID},
	} {
		if field.value != "" {
			add(field.column+" = ?", field.value)
		}
	}
	if filters.Model != "" {
		add("(selected_model = ? OR original_model = ?)", filters.Model)
	}
	if filters.Search != "" {
		add("(strpos(lower(COALESCE(request_id, '')), lower(?)) > 0 OR strpos(lower(COALESCE(recipe, '')), lower(?)) > 0)", filters.Search)
	}
	switch filters.CacheStatus {
	case "cached":
		clauses = append(clauses, "from_cache = TRUE")
	case "streamed":
		clauses = append(clauses, "streaming = TRUE")
	}
	if len(clauses) == 0 {
		return "", args
	}
	return " WHERE " + strings.Join(clauses, " AND "), args
}

func postgresSummarySelectColumns() string {
	columns := strings.Split(postgresRecordSelectColumns, ",")
	for i, raw := range columns {
		switch strings.TrimSpace(raw) {
		case "request_body", "response_body", "prompt", "tool_definitions":
			columns[i] = "''"
		case "projection_trace":
			columns[i] = "NULL"
		case "tool_trace":
			// Tool names remain discoverable without returning arguments/results.
			columns[i] = `CASE WHEN tool_trace IS NULL OR tool_trace = 'null'::jsonb THEN NULL ELSE
			jsonb_build_object('flow', tool_trace->'flow', 'stage', tool_trace->'stage', 'tool_names',
			(SELECT jsonb_agg(name ORDER BY name) FROM (
			SELECT DISTINCT name FROM (
			SELECT jsonb_array_elements_text(COALESCE(NULLIF(tool_trace->'tool_names', 'null'::jsonb), '[]'::jsonb)) AS name
			UNION ALL SELECT step->>'tool_name' FROM jsonb_array_elements(COALESCE(NULLIF(tool_trace->'steps', 'null'::jsonb), '[]'::jsonb)) AS step
			) names WHERE name IS NOT NULL AND name <> '') distinct_names)) END`
		}
	}
	return strings.Join(columns, ",")
}

// ScanMetadata streams only the columns used by Insights. Large request,
// response, tool, safety, and learning payloads never cross the database wire.
func (p *PostgresStore) ScanMetadata(ctx context.Context, visit func(Record) error) (err error) {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`SELECT COALESCE(request_id,''), COALESCE(recipe,''), COALESCE(decision,''),
		COALESCE(original_model,''), COALESCE(selected_model,''), COALESCE(session_id,''),
		from_cache, streaming, lifecycle_state, COALESCE(signals, '{}'::jsonb),
		prompt_tokens, completion_tokens, total_tokens, actual_cost, baseline_cost, cost_savings, currency, baseline_model
		FROM %s ORDER BY timestamp DESC, id DESC`, p.tableName)
	rows, err := p.db.QueryContext(ctx, query)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, rows.Close()) }()
	for rows.Next() {
		row := postgresRecordRow{}
		if err = rows.Scan(&row.record.RequestID, &row.record.Recipe, &row.record.Decision,
			&row.record.OriginalModel, &row.record.SelectedModel, &row.record.SessionID,
			&row.record.FromCache, &row.record.Streaming, &row.record.LifecycleState, &row.signalsJSON,
			&row.promptTokens, &row.completionTokens, &row.totalTokens, &row.actualCost,
			&row.baselineCost, &row.costSavings, &row.currency, &row.baselineModel); err != nil {
			return err
		}
		if err = json.Unmarshal(row.signalsJSON, &row.record.Signals); err != nil {
			return err
		}
		assignUsageCostFields(&row.record, row.promptTokens, sql.NullInt64{}, sql.NullInt64{}, row.completionTokens,
			row.totalTokens, row.actualCost, row.baselineCost, row.costSavings, row.currency, row.baselineModel)
		if err = visit(row.record); err != nil {
			return err
		}
	}
	return rows.Err()
}
