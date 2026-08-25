// Package auditlog exposes immutable Router Management audit events through
// bounded, keyset-paginated PostgreSQL queries. Mutation code remains in the
// owning domain repositories; this package is read-only.
package auditlog

import (
	"context"
	"crypto/hmac"
	"crypto/sha256"
	"database/sql"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/netip"
	"regexp"
	"strings"
	"time"

	"github.com/google/uuid"
)

var (
	ErrInvalidQuery = errors.New("invalid audit query")
	ErrCorrupt      = errors.New("audit log invariant violation")

	auditCodePattern = regexp.MustCompile(`^[a-z][a-z0-9._:-]{0,127}$`)
)

type Filters struct {
	ActorPrincipalID string
	Action           string
	ResourceType     string
	ResourceID       string
	Outcome          string
	RequestID        string
}

type Query struct {
	NamespaceID string
	Start       time.Time
	End         time.Time
	Filters     Filters
	PageSize    int
	Cursor      string
}

type Event struct {
	ID               string            `json:"id"`
	NamespaceID      string            `json:"namespaceId"`
	DesiredRevision  *int64            `json:"desiredRevision,omitempty"`
	ChainSequence    int64             `json:"chainSequence"`
	ActorPrincipalID string            `json:"actorPrincipalId,omitempty"`
	ActorChain       []string          `json:"actorChain"`
	Action           string            `json:"action"`
	ResourceType     string            `json:"resourceType"`
	ResourceID       string            `json:"resourceId,omitempty"`
	RequestID        string            `json:"requestId"`
	SourceIP         string            `json:"sourceIp,omitempty"`
	Outcome          string            `json:"outcome"`
	Reason           string            `json:"reason"`
	BeforeRevision   *int64            `json:"beforeRevision,omitempty"`
	AfterRevision    *int64            `json:"afterRevision,omitempty"`
	Details          map[string]string `json:"details"`
	PreviousHash     string            `json:"previousHash,omitempty"`
	EventHash        string            `json:"eventHash"`
	CreatedAt        time.Time         `json:"createdAt"`
}

type Page struct {
	Items      []Event
	NextCursor string
}

type CursorCodec struct{ key []byte }

func NewCursorCodec(key []byte) (*CursorCodec, error) {
	if len(key) < 32 {
		return nil, fmt.Errorf("audit cursor HMAC key must contain at least 32 bytes")
	}
	return &CursorCodec{key: append([]byte(nil), key...)}, nil
}

// Close erases the process-owned cursor signing material. The audit query
// service retains no deployment-root bytes after Management shutdown.
func (codec *CursorCodec) Close() {
	if codec == nil {
		return
	}
	for index := range codec.key {
		codec.key[index] = 0
	}
	codec.key = nil
}

type PostgresQueries struct{ DB *sql.DB }

func (queries PostgresQueries) List(
	ctx context.Context,
	query Query,
	codec *CursorCodec,
) (_ Page, returnErr error) {
	if queries.DB == nil || codec == nil {
		return Page{}, fmt.Errorf("audit database and cursor codec are required")
	}
	if query.PageSize == 0 {
		query.PageSize = 50
	}
	if err := validateQuery(query); err != nil {
		return Page{}, err
	}
	digest := queryDigest(query)
	var after *cursorValue
	if query.Cursor != "" {
		value, err := codec.decode(query.Cursor)
		if err != nil {
			return Page{}, err
		}
		if value.NamespaceID != query.NamespaceID || value.QueryDigest != digest {
			return Page{}, fmt.Errorf("%w: audit cursor does not belong to this query", ErrInvalidQuery)
		}
		after = &value
	}
	where, args := queryWhere(query)
	if after != nil {
		args = append(args, time.Unix(0, after.CreatedAt).UTC(), after.EventID)
		where += fmt.Sprintf(" AND (created_at, id) < ($%d, $%d::uuid)", len(args)-1, len(args))
	}
	args = append(args, query.PageSize+1)
	// #nosec G201 -- queryWhere emits only fixed column/cast clauses; every value remains a bind parameter.
	statement := fmt.Sprintf(`SELECT id::text, namespace_id::text, desired_revision,
  chain_sequence, COALESCE(actor_principal_id::text,''), actor_chain, action,
  resource_type, COALESCE(resource_id,''), request_id, COALESCE(source_ip::text,''),
  outcome, reason, before_revision, after_revision, details, previous_hash,
  event_hash, created_at
FROM access_audit_events
WHERE %s
ORDER BY created_at DESC, id DESC
LIMIT $%d`, where, len(args))
	rows, listErr := queries.DB.QueryContext(ctx, statement, args...)
	if listErr != nil {
		return Page{}, fmt.Errorf("list audit events: %w", listErr)
	}
	defer func() {
		returnErr = errors.Join(returnErr, rows.Close())
	}()
	items := make([]Event, 0, query.PageSize+1)
	for rows.Next() {
		item, err := scanEvent(rows)
		if err != nil {
			return Page{}, err
		}
		items = append(items, item)
	}
	if err := rows.Err(); err != nil {
		return Page{}, fmt.Errorf("iterate audit events: %w", err)
	}
	page := Page{Items: items}
	if len(items) > query.PageSize {
		last := items[query.PageSize-1]
		page.Items = items[:query.PageSize]
		page.NextCursor, listErr = codec.encode(cursorValue{
			Version: 1, NamespaceID: query.NamespaceID, QueryDigest: digest,
			CreatedAt: last.CreatedAt.UnixNano(), EventID: last.ID,
		})
		if listErr != nil {
			return Page{}, listErr
		}
	}
	return page, nil
}

type scanner interface{ Scan(...any) error }

func scanEvent(row scanner) (Event, error) {
	var event Event
	var desired, before, after sql.NullInt64
	var actorJSON, detailsJSON []byte
	var previousHash, eventHash []byte
	if err := row.Scan(
		&event.ID, &event.NamespaceID, &desired, &event.ChainSequence,
		&event.ActorPrincipalID, &actorJSON, &event.Action, &event.ResourceType,
		&event.ResourceID, &event.RequestID, &event.SourceIP, &event.Outcome,
		&event.Reason, &before, &after, &detailsJSON, &previousHash, &eventHash,
		&event.CreatedAt,
	); err != nil {
		return Event{}, fmt.Errorf("scan audit event: %w", err)
	}
	if desired.Valid {
		event.DesiredRevision = &desired.Int64
	}
	if before.Valid {
		event.BeforeRevision = &before.Int64
	}
	if after.Valid {
		event.AfterRevision = &after.Int64
	}
	if err := json.Unmarshal(actorJSON, &event.ActorChain); err != nil || event.ActorChain == nil {
		return Event{}, fmt.Errorf("%w: actor chain", ErrCorrupt)
	}
	if err := json.Unmarshal(detailsJSON, &event.Details); err != nil || event.Details == nil {
		return Event{}, fmt.Errorf("%w: details", ErrCorrupt)
	}
	event.PreviousHash = hex.EncodeToString(previousHash)
	event.EventHash = hex.EncodeToString(eventHash)
	if err := validateEvent(event); err != nil {
		return Event{}, err
	}
	return event, nil
}

func validateQuery(query Query) error {
	if _, err := uuid.Parse(query.NamespaceID); err != nil {
		return fmt.Errorf("%w: namespace ID", ErrInvalidQuery)
	}
	if query.Start.IsZero() || query.End.IsZero() || !query.Start.Before(query.End) || query.End.Sub(query.Start) > 5*366*24*time.Hour {
		return fmt.Errorf("%w: time range", ErrInvalidQuery)
	}
	if query.PageSize < 1 || query.PageSize > 200 {
		return fmt.Errorf("%w: page size", ErrInvalidQuery)
	}
	if query.Filters.ActorPrincipalID != "" {
		if _, err := uuid.Parse(query.Filters.ActorPrincipalID); err != nil {
			return fmt.Errorf("%w: actor principal ID", ErrInvalidQuery)
		}
	}
	for label, value := range map[string]string{
		"action": query.Filters.Action, "resource type": query.Filters.ResourceType,
		"outcome": query.Filters.Outcome,
	} {
		if value != "" && !auditCodePattern.MatchString(value) {
			return fmt.Errorf("%w: %s", ErrInvalidQuery, label)
		}
	}
	if query.Filters.Outcome != "" && query.Filters.Outcome != "allowed" && query.Filters.Outcome != "denied" && query.Filters.Outcome != "failed" {
		return fmt.Errorf("%w: outcome", ErrInvalidQuery)
	}
	for label, value := range map[string]string{
		"resource ID": query.Filters.ResourceID, "request ID": query.Filters.RequestID,
	} {
		if len(value) > 256 || strings.ContainsRune(value, '\x00') {
			return fmt.Errorf("%w: %s", ErrInvalidQuery, label)
		}
	}
	return nil
}

func validateEvent(event Event) error {
	if _, err := uuid.Parse(event.ID); err != nil {
		return fmt.Errorf("%w: event ID", ErrCorrupt)
	}
	if _, err := uuid.Parse(event.NamespaceID); err != nil || event.ChainSequence < 1 || len(event.EventHash) != sha256.Size*2 {
		return fmt.Errorf("%w: identity or chain", ErrCorrupt)
	}
	if event.PreviousHash != "" && len(event.PreviousHash) != sha256.Size*2 {
		return fmt.Errorf("%w: previous hash", ErrCorrupt)
	}
	if event.ActorPrincipalID != "" {
		if _, err := uuid.Parse(event.ActorPrincipalID); err != nil {
			return fmt.Errorf("%w: actor", ErrCorrupt)
		}
	}
	for _, actor := range event.ActorChain {
		if _, err := uuid.Parse(actor); err != nil {
			return fmt.Errorf("%w: actor chain", ErrCorrupt)
		}
	}
	if event.SourceIP != "" {
		if _, err := netip.ParseAddr(event.SourceIP); err != nil {
			return fmt.Errorf("%w: source IP", ErrCorrupt)
		}
	}
	if event.CreatedAt.IsZero() || event.Action == "" || event.ResourceType == "" || event.Outcome == "" || event.Reason == "" {
		return fmt.Errorf("%w: required event fields", ErrCorrupt)
	}
	return nil
}

func queryWhere(query Query) (string, []any) {
	args := []any{query.NamespaceID, query.Start, query.End}
	clauses := []string{"namespace_id = $1", "created_at >= $2", "created_at < $3"}
	for _, filter := range []struct {
		column string
		value  string
		cast   string
	}{
		{"actor_principal_id", query.Filters.ActorPrincipalID, "uuid"},
		{"action", query.Filters.Action, "text"},
		{"resource_type", query.Filters.ResourceType, "text"},
		{"resource_id", query.Filters.ResourceID, "text"},
		{"outcome", query.Filters.Outcome, "text"},
		{"request_id", query.Filters.RequestID, "text"},
	} {
		if filter.value == "" {
			continue
		}
		args = append(args, filter.value)
		clauses = append(clauses, fmt.Sprintf("%s = $%d::%s", filter.column, len(args), filter.cast))
	}
	return strings.Join(clauses, " AND "), args
}

type cursorValue struct {
	Version     int    `json:"v"`
	NamespaceID string `json:"n"`
	QueryDigest string `json:"q"`
	CreatedAt   int64  `json:"t"`
	EventID     string `json:"e"`
}

func queryDigest(query Query) string {
	payload, _ := json.Marshal(struct {
		Start   int64   `json:"start"`
		End     int64   `json:"end"`
		Filters Filters `json:"filters"`
	}{Start: query.Start.UnixNano(), End: query.End.UnixNano(), Filters: query.Filters})
	digest := sha256.Sum256(payload)
	return hex.EncodeToString(digest[:])
}

func (codec *CursorCodec) encode(value cursorValue) (string, error) {
	if codec == nil || len(codec.key) < sha256.Size {
		return "", fmt.Errorf("%w: cursor codec is closed", ErrInvalidQuery)
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	mac := hmac.New(sha256.New, codec.key)
	_, _ = mac.Write(payload)
	return base64.RawURLEncoding.EncodeToString(payload) + "." +
		base64.RawURLEncoding.EncodeToString(mac.Sum(nil)), nil
}

func (codec *CursorCodec) decode(encoded string) (cursorValue, error) {
	if codec == nil || len(codec.key) < sha256.Size {
		return cursorValue{}, fmt.Errorf("%w: cursor codec is closed", ErrInvalidQuery)
	}
	payloadPart, signaturePart, ok := strings.Cut(encoded, ".")
	if !ok || len(encoded) > 2048 {
		return cursorValue{}, fmt.Errorf("%w: malformed cursor", ErrInvalidQuery)
	}
	payload, err := base64.RawURLEncoding.DecodeString(payloadPart)
	if err != nil || base64.RawURLEncoding.EncodeToString(payload) != payloadPart {
		return cursorValue{}, fmt.Errorf("%w: malformed cursor", ErrInvalidQuery)
	}
	signature, err := base64.RawURLEncoding.DecodeString(signaturePart)
	if err != nil || len(signature) != sha256.Size ||
		base64.RawURLEncoding.EncodeToString(signature) != signaturePart {
		return cursorValue{}, fmt.Errorf("%w: malformed cursor", ErrInvalidQuery)
	}
	mac := hmac.New(sha256.New, codec.key)
	_, _ = mac.Write(payload)
	if !hmac.Equal(signature, mac.Sum(nil)) {
		return cursorValue{}, fmt.Errorf("%w: cursor signature", ErrInvalidQuery)
	}
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	var value cursorValue
	if err := decoder.Decode(&value); err != nil {
		return cursorValue{}, fmt.Errorf("%w: cursor payload", ErrInvalidQuery)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return cursorValue{}, fmt.Errorf("%w: cursor trailing data", ErrInvalidQuery)
	}
	if value.Version != 1 || value.CreatedAt <= 0 {
		return cursorValue{}, fmt.Errorf("%w: cursor payload", ErrInvalidQuery)
	}
	if _, err := uuid.Parse(value.NamespaceID); err != nil {
		return cursorValue{}, fmt.Errorf("%w: cursor namespace", ErrInvalidQuery)
	}
	if _, err := uuid.Parse(value.EventID); err != nil || len(value.QueryDigest) != sha256.Size*2 {
		return cursorValue{}, fmt.Errorf("%w: cursor identity", ErrInvalidQuery)
	}
	return value, nil
}
