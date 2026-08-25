package usageledger

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

type LogCursorCodec struct {
	key []byte
}

func NewLogCursorCodec(key []byte) (*LogCursorCodec, error) {
	if len(key) < 32 {
		return nil, fmt.Errorf("request-log cursor HMAC key must contain at least 32 bytes")
	}
	return &LogCursorCodec{key: append([]byte(nil), key...)}, nil
}

// Close erases the process-owned cursor signing material. A codec must not be
// reused after Close; Management composition closes it before releasing its
// borrowed deployment keyrings.
func (codec *LogCursorCodec) Close() {
	if codec == nil {
		return
	}
	for index := range codec.key {
		codec.key[index] = 0
	}
	codec.key = nil
}

type LogQuery struct {
	NamespaceID       string
	ExternalRequestID string
	Start             time.Time
	End               time.Time
	Filters           UsageFilters
	Visibility        QueryVisibility
	PageSize          int
	Cursor            string
}

type RequestLog struct {
	AdmissionID         string            `json:"admissionId"`
	EventID             string            `json:"eventId"`
	ExternalRequestID   string            `json:"externalRequestId,omitempty"`
	OccurredAt          time.Time         `json:"occurredAt"`
	CompletedAt         time.Time         `json:"completedAt"`
	Protocol            string            `json:"protocol"`
	Path                string            `json:"path"`
	StatusCode          int               `json:"statusCode"`
	ErrorCode           string            `json:"errorCode,omitempty"`
	UsageState          UsageState        `json:"usageState"`
	InputTokens         string            `json:"inputTokens"`
	OutputTokens        string            `json:"outputTokens"`
	LatencyMilliseconds int64             `json:"latencyMilliseconds"`
	TTFTMilliseconds    *int64            `json:"ttftMilliseconds,omitempty"`
	Stream              bool              `json:"stream"`
	ToolCall            bool              `json:"toolCall"`
	APIKeyID            string            `json:"apiKeyId,omitempty"`
	UserID              string            `json:"userId,omitempty"`
	TeamID              string            `json:"teamId,omitempty"`
	EntrypointID        string            `json:"entrypointId,omitempty"`
	RecipeID            string            `json:"recipeId,omitempty"`
	Metadata            map[string]string `json:"metadata,omitempty"`
	Costs               []CostSummary     `json:"costs"`
}

type LogPage struct {
	Items      []RequestLog `json:"items"`
	NextCursor string       `json:"nextCursor,omitempty"`
}

type logCursor struct {
	Version     int    `json:"v"`
	NamespaceID string `json:"n"`
	QueryDigest string `json:"q"`
	OccurredAt  int64  `json:"t"`
	EventID     string `json:"e"`
}

func (q PostgresQueries) ListLogs(ctx context.Context, query LogQuery, codec *LogCursorCodec) (LogPage, error) {
	if q.DB == nil || codec == nil {
		return LogPage{}, fmt.Errorf("request-log database and cursor codec are required")
	}
	usage := UsageQuery{
		NamespaceID: query.NamespaceID, Start: query.Start, End: query.End,
		Grain: GrainDay, Filters: query.Filters, Visibility: query.Visibility,
	}
	if err := validateUsageQuery(usage); err != nil {
		return LogPage{}, err
	}
	if query.Filters.LogicalModelID != "" || query.Filters.BackendID != "" || query.Filters.ProviderID != "" || query.Filters.DispatchType != "" {
		return LogPage{}, invalidQueryf("raw request-log list does not accept internal dispatch filters")
	}
	if query.ExternalRequestID != "" {
		if err := boundedIdentifier("external request ID", query.ExternalRequestID, 256); err != nil ||
			looksSensitive(query.ExternalRequestID) {
			return LogPage{}, invalidQueryf("external request ID is invalid")
		}
	}
	if query.PageSize == 0 {
		query.PageSize = 50
	}
	if query.PageSize < 1 || query.PageSize > 200 {
		return LogPage{}, invalidQueryf("request-log page size must be between 1 and 200")
	}
	filterDigest := logFilterDigest(query)
	var cursor *logCursor
	if query.Cursor != "" {
		decoded, err := codec.decode(query.Cursor)
		if err != nil {
			return LogPage{}, invalidQuery(err)
		}
		if decoded.NamespaceID != query.NamespaceID || decoded.QueryDigest != filterDigest {
			return LogPage{}, invalidQueryf("request-log cursor does not belong to this query")
		}
		cursor = &decoded
	}
	return q.listLogPage(ctx, query, codec, filterDigest, cursor)
}

func (q PostgresQueries) listLogPage(
	ctx context.Context,
	query LogQuery,
	codec *LogCursorCodec,
	filterDigest string,
	cursor *logCursor,
) (LogPage, error) {
	statement, args := rawLogPageQuery(query, cursor)
	rows, err := q.DB.QueryContext(ctx, statement, args...)
	if err != nil {
		return LogPage{}, fmt.Errorf("list request logs: %w", err)
	}
	defer rows.Close()
	items := make([]RequestLog, 0, query.PageSize+1)
	for rows.Next() {
		item, scanErr := scanRequestLog(rows)
		if scanErr != nil {
			return LogPage{}, scanErr
		}
		items = append(items, item)
	}
	if rowsErr := rows.Err(); rowsErr != nil {
		return LogPage{}, rowsErr
	}
	page := LogPage{Items: items}
	if len(items) > query.PageSize {
		last := items[query.PageSize-1]
		page.Items = items[:query.PageSize]
		page.NextCursor, err = codec.encode(logCursor{
			Version: 1, NamespaceID: query.NamespaceID, QueryDigest: filterDigest,
			OccurredAt: last.OccurredAt.UnixNano(), EventID: last.EventID,
		})
		if err != nil {
			return LogPage{}, err
		}
	}
	return page, nil
}

func rawLogPageQuery(query LogQuery, cursor *logCursor) (string, []any) {
	where, args := rawLogWhere(query)
	if cursor != nil {
		args = append(args, time.Unix(0, cursor.OccurredAt).UTC(), cursor.EventID)
		where += fmt.Sprintf(" AND (occurred_at, event_id) < ($%d, $%d::uuid)", len(args)-1, len(args))
	}
	args = append(args, query.PageSize+1)
	return fmt.Sprintf(`SELECT admission_id, event_id::text, COALESCE(external_request_id,''), occurred_at, protocol, path,
  status_code, COALESCE(error_code,''), usage_state, input_tokens::text, output_tokens::text,
  latency_ms, ttft_ms, COALESCE(api_key_id::text,''), COALESCE(user_id::text,''), COALESCE(team_id::text,''),
  COALESCE(entrypoint_id::text,''), COALESCE(recipe_id::text,''), costs, request_metadata
FROM usage_events WHERE %s
ORDER BY occurred_at DESC, event_id DESC LIMIT $%d`, where, len(args)), args
}

type rowScanner interface {
	Scan(...any) error
}

func scanRequestLog(row rowScanner) (RequestLog, error) {
	item, _, err := scanRequestLogWithMetadata(row)
	return item, err
}

func scanRequestLogWithMetadata(row rowScanner) (RequestLog, safeEventMetadata, error) {
	var item RequestLog
	var costsJSON, metadataJSON []byte
	if err := row.Scan(&item.AdmissionID, &item.EventID, &item.ExternalRequestID, &item.OccurredAt, &item.Protocol,
		&item.Path, &item.StatusCode, &item.ErrorCode, &item.UsageState, &item.InputTokens,
		&item.OutputTokens, &item.LatencyMilliseconds, &item.TTFTMilliseconds, &item.APIKeyID, &item.UserID,
		&item.TeamID, &item.EntrypointID, &item.RecipeID, &costsJSON, &metadataJSON); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("scan request log: %w", err)
	}
	var costs []storedCost
	if err := json.Unmarshal(costsJSON, &costs); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("%w: decode request-log costs", ErrLedgerCorrupt)
	}
	aggregates := make([]CostAggregate, 0, len(costs))
	for _, value := range costs {
		cost, err := parseCostAggregate(value.Currency, value.KnownNumerator, value.KnownDispatches, value.IncompleteDispatches)
		if err != nil {
			return RequestLog{}, safeEventMetadata{}, err
		}
		aggregates = append(aggregates, cost)
	}
	item.Costs = publicCosts(aggregates)
	var metadata safeEventMetadata
	if err := json.Unmarshal(metadataJSON, &metadata); err != nil {
		return RequestLog{}, safeEventMetadata{}, fmt.Errorf("%w: decode request-log metadata", ErrLedgerCorrupt)
	}
	item.CompletedAt = metadata.CompletedAt
	item.Stream = metadata.Stream
	item.ToolCall = metadata.ToolCall
	item.Metadata = metadata.Metadata
	return item, metadata, nil
}

func rawLogWhere(query LogQuery) (string, []any) {
	partitionStart := query.Start.UTC().Truncate(24 * time.Hour)
	partitionEnd := query.End.UTC().Truncate(24 * time.Hour)
	if !query.End.UTC().Equal(partitionEnd) {
		partitionEnd = partitionEnd.Add(24 * time.Hour)
	}
	args := []any{query.NamespaceID, query.Start, query.End, partitionStart, partitionEnd}
	clauses := []string{
		"namespace_id = $1", "occurred_at >= $2", "occurred_at < $3",
		"event_date >= $4::date", "event_date < $5::date",
		"event_kind IN ('actual','unknown')",
	}
	filters := []struct {
		column string
		value  string
		cast   string
	}{
		{"api_key_id", query.Filters.APIKeyID, "uuid"},
		{"user_id", query.Filters.UserID, "uuid"},
		{"team_id", query.Filters.TeamID, "uuid"},
		{"entrypoint_id", query.Filters.EntrypointID, "text"},
		{"recipe_id", query.Filters.RecipeID, "text"},
		{"protocol", query.Filters.Protocol, "text"},
		{"error_code", query.Filters.ErrorCode, "text"},
	}
	for _, filter := range filters {
		if filter.value == "" {
			continue
		}
		args = append(args, filter.value)
		clauses = append(clauses, fmt.Sprintf("%s = $%d::%s", filter.column, len(args), filter.cast))
	}
	if query.ExternalRequestID != "" {
		args = append(args, query.ExternalRequestID)
		clauses = append(clauses, fmt.Sprintf("external_request_id = $%d", len(args)))
	}
	if query.Filters.StatusCode != 0 {
		args = append(args, query.Filters.StatusCode)
		clauses = append(clauses, fmt.Sprintf("status_code = $%d", len(args)))
	}
	appendRawLogVisibility(&clauses, &args, query.Visibility)
	return strings.Join(clauses, " AND "), args
}

func appendRawLogVisibility(clauses *[]string, args *[]any, visibility QueryVisibility) {
	if visibility.All {
		return
	}
	parts := make([]string, 0, 3)
	for _, dimension := range []struct {
		column string
		values []string
	}{
		{"team_id", visibility.TeamIDs},
		{"user_id", visibility.UserIDs},
		{"api_key_id", visibility.APIKeyIDs},
	} {
		if len(dimension.values) == 0 {
			continue
		}
		*args = append(*args, pq.Array(dimension.values))
		parts = append(parts, fmt.Sprintf("%s = ANY($%d::uuid[])", dimension.column, len(*args)))
	}
	*clauses = append(*clauses, "("+strings.Join(parts, " OR ")+")")
}

func logFilterDigest(query LogQuery) string {
	payload, _ := json.Marshal(struct {
		Start             int64           `json:"start"`
		End               int64           `json:"end"`
		ExternalRequestID string          `json:"requestId,omitempty"`
		Filters           UsageFilters    `json:"filters"`
		Visibility        QueryVisibility `json:"visibility"`
	}{
		Start: query.Start.UnixNano(), End: query.End.UnixNano(),
		ExternalRequestID: query.ExternalRequestID,
		Filters:           query.Filters, Visibility: query.Visibility,
	})
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func (c *LogCursorCodec) encode(value logCursor) (string, error) {
	if c == nil || len(c.key) < sha256.Size {
		return "", fmt.Errorf("request-log cursor codec is closed")
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	mac := hmac.New(sha256.New, c.key)
	_, _ = mac.Write(payload)
	return base64.RawURLEncoding.EncodeToString(payload) + "." + base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (c *LogCursorCodec) decode(encoded string) (logCursor, error) {
	if c == nil || len(c.key) < sha256.Size {
		return logCursor{}, fmt.Errorf("request-log cursor codec is closed")
	}
	payloadPart, signaturePart, ok := strings.Cut(encoded, ".")
	if !ok || len(encoded) > 2048 {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	payload, err := base64.RawURLEncoding.DecodeString(payloadPart)
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != payloadPart {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	signature, err := base64.RawURLEncoding.DecodeString(signaturePart)
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != signaturePart {
		return logCursor{}, fmt.Errorf("request-log cursor is malformed")
	}
	mac := hmac.New(sha256.New, c.key)
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return logCursor{}, fmt.Errorf("request-log cursor signature is invalid")
	}
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	var value logCursor
	if err := decoder.Decode(&value); err != nil || value.Version != 1 || value.OccurredAt <= 0 {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if _, err := uuid.Parse(value.NamespaceID); err != nil {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if _, err := uuid.Parse(value.EventID); err != nil {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	if !isHexDigest(value.QueryDigest) {
		return logCursor{}, fmt.Errorf("request-log cursor is invalid")
	}
	return value, nil
}
