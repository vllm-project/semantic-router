package accesscontrol

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

func (s *Store) InsertUsage(ctx context.Context, event UsageEvent) error {
	metadata, _ := json.Marshal(event.Metadata)
	_, err := s.pool.Exec(ctx, `
INSERT INTO access_usage_events(id,request_id,key_id,user_id,team_id,model,status_code,prompt_tokens,completion_tokens,total_tokens,latency_ms,ttft_ms,error_code,metadata,created_at)
VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15)
ON CONFLICT(request_id) DO NOTHING`, event.ID, event.RequestID, event.KeyID, event.UserID, event.TeamID,
		event.Model, event.StatusCode, event.PromptTokens, event.CompletionTokens, event.TotalTokens,
		event.LatencyMS, event.TTFTMS, event.ErrorCode, metadata, event.CreatedAt)
	return err
}

func (s *Store) ListUsage(ctx context.Context, filter ListFilter) ([]UsageEvent, error) {
	filter = normalizeFilter(filter)
	rows, err := s.pool.Query(ctx, `
SELECT id,request_id,key_id,user_id,team_id,model,status_code,prompt_tokens,completion_tokens,total_tokens,latency_ms,ttft_ms,error_code,metadata,created_at
FROM access_usage_events
WHERE ($1='' OR user_id=$1) AND ($2='' OR team_id=$2) AND ($3='' OR key_id=$3) AND ($4='' OR model=$4)
  AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at <= $6)
	  AND ($7='' OR request_id ILIKE $7 || '%' OR model ILIKE $7 || '%' OR error_code ILIKE $7 || '%')
ORDER BY created_at DESC LIMIT $8 OFFSET $9`, filter.UserID, filter.TeamID, filter.KeyID, filter.Model, filter.From, filter.To, filter.Query, filter.Limit, filter.Offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []UsageEvent{}
	for rows.Next() {
		item, scanErr := scanUsageEvent(rows)
		if scanErr != nil {
			return nil, scanErr
		}
		items = append(items, item)
	}
	return items, rows.Err()
}

func (s *Store) GetUsage(ctx context.Context, id string) (UsageEvent, error) {
	return scanUsageEvent(s.pool.QueryRow(ctx, `
SELECT id,request_id,key_id,user_id,team_id,model,status_code,prompt_tokens,completion_tokens,total_tokens,latency_ms,ttft_ms,error_code,metadata,created_at
FROM access_usage_events WHERE id=$1`, id))
}

func scanUsageEvent(row rowScanner) (UsageEvent, error) {
	var item UsageEvent
	var metadata []byte
	if err := row.Scan(&item.ID, &item.RequestID, &item.KeyID, &item.UserID, &item.TeamID, &item.Model, &item.StatusCode,
		&item.PromptTokens, &item.CompletionTokens, &item.TotalTokens, &item.LatencyMS, &item.TTFTMS, &item.ErrorCode, &metadata, &item.CreatedAt); err != nil {
		return UsageEvent{}, err
	}
	_ = json.Unmarshal(metadata, &item.Metadata)
	return item, nil
}

func (s *Store) CountUsage(ctx context.Context, filter ListFilter) (int64, error) {
	var total int64
	err := s.pool.QueryRow(ctx, `
SELECT COUNT(*) FROM access_usage_events
WHERE ($1='' OR user_id=$1) AND ($2='' OR team_id=$2) AND ($3='' OR key_id=$3) AND ($4='' OR model=$4)
	  AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at <= $6)
	  AND ($7='' OR request_id ILIKE $7 || '%' OR model ILIKE $7 || '%' OR error_code ILIKE $7 || '%')`,
		filter.UserID, filter.TeamID, filter.KeyID, filter.Model, filter.From, filter.To, filter.Query).Scan(&total)
	return total, err
}

func (s *Store) UsageSummary(ctx context.Context, filter ListFilter) (UsageSummary, error) {
	if filter.From == nil {
		from := time.Now().UTC().Add(-24 * time.Hour)
		filter.From = &from
	}
	var summary UsageSummary
	err := s.pool.QueryRow(ctx, `
SELECT COUNT(*),COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 399),COUNT(*) FILTER (WHERE status_code >= 400),
 COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),COALESCE(SUM(total_tokens),0),COUNT(DISTINCT key_id),
 COALESCE(AVG(latency_ms),0)::BIGINT,
	 COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),0)::BIGINT,
	 COALESCE(AVG(ttft_ms) FILTER (WHERE ttft_ms > 0),0)::BIGINT,
	 COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY ttft_ms) FILTER (WHERE ttft_ms > 0),0)::BIGINT
FROM access_usage_events
WHERE ($1='' OR user_id=$1) AND ($2='' OR team_id=$2) AND ($3='' OR key_id=$3) AND ($4='' OR model=$4)
 AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at <= $6)`,
		filter.UserID, filter.TeamID, filter.KeyID, filter.Model, filter.From, filter.To).
		Scan(&summary.Requests, &summary.Successful, &summary.Failed, &summary.PromptTokens, &summary.CompletionTokens,
			&summary.TotalTokens, &summary.ActiveKeys, &summary.AverageLatencyMS, &summary.P95LatencyMS, &summary.AverageTTFTMS, &summary.P95TTFTMS)
	if err != nil {
		return summary, err
	}
	bucket := "hour"
	if filter.From != nil && time.Since(*filter.From) > 48*time.Hour {
		bucket = "day"
	}
	seriesQuery := fmt.Sprintf(`
	SELECT date_trunc('%s',created_at),COUNT(*),COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 399),
	 COUNT(*) FILTER (WHERE status_code >= 400),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),
	 COALESCE(SUM(total_tokens),0),COALESCE(AVG(latency_ms),0)::BIGINT
FROM access_usage_events
WHERE ($1='' OR user_id=$1) AND ($2='' OR team_id=$2) AND ($3='' OR key_id=$3) AND ($4='' OR model=$4)
 AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at <= $6)
GROUP BY 1 ORDER BY 1`, bucket)
	rows, err := s.pool.Query(ctx, seriesQuery, filter.UserID, filter.TeamID, filter.KeyID, filter.Model, filter.From, filter.To)
	if err != nil {
		return summary, err
	}
	for rows.Next() {
		var point UsagePoint
		if scanErr := rows.Scan(&point.Bucket, &point.Requests, &point.Successful, &point.Failed, &point.PromptTokens, &point.CompletionTokens, &point.TotalTokens, &point.AverageLatencyMS); scanErr != nil {
			rows.Close()
			return summary, scanErr
		}
		summary.Series = append(summary.Series, point)
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		rows.Close()
		return summary, rowsErr
	}
	rows.Close()
	if summary.ByModel, err = s.usageSlices(ctx, filter, "model"); err != nil {
		return summary, err
	}
	if summary.ByUser, err = s.usageSlices(ctx, filter, "user_id"); err != nil {
		return summary, err
	}
	if summary.ByTeam, err = s.usageSlices(ctx, filter, "team_id"); err != nil {
		return summary, err
	}
	if summary.ByKey, err = s.usageSlices(ctx, filter, "key_id"); err != nil {
		return summary, err
	}
	return summary, nil
}

func (s *Store) usageSlices(ctx context.Context, filter ListFilter, dimension string) ([]UsageSlice, error) {
	switch dimension {
	case "model", "user_id", "team_id", "key_id":
	default:
		return nil, fmt.Errorf("unsupported usage dimension %q", dimension)
	}
	query := fmt.Sprintf(`
	SELECT COALESCE(NULLIF(%s,''),'unassigned'),COUNT(*),COUNT(*) FILTER (WHERE status_code BETWEEN 200 AND 399),
	 COUNT(*) FILTER (WHERE status_code >= 400),COALESCE(SUM(prompt_tokens),0),COALESCE(SUM(completion_tokens),0),
	 COALESCE(SUM(total_tokens),0),COALESCE(AVG(latency_ms),0)::BIGINT,
	 COALESCE(percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),0)::BIGINT
FROM access_usage_events
WHERE ($1='' OR user_id=$1) AND ($2='' OR team_id=$2) AND ($3='' OR key_id=$3) AND ($4='' OR model=$4)
 AND ($5::timestamptz IS NULL OR created_at >= $5) AND ($6::timestamptz IS NULL OR created_at <= $6)
GROUP BY 1 ORDER BY 7 DESC LIMIT 12`, dimension)
	rows, err := s.pool.Query(ctx, query, filter.UserID, filter.TeamID, filter.KeyID, filter.Model, filter.From, filter.To)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	items := []UsageSlice{}
	for rows.Next() {
		var item UsageSlice
		if err := rows.Scan(&item.ID, &item.Requests, &item.Successful, &item.Failed, &item.PromptTokens, &item.CompletionTokens, &item.TotalTokens, &item.AverageLatencyMS, &item.P95LatencyMS); err != nil {
			return nil, err
		}
		items = append(items, item)
	}
	return items, rows.Err()
}
