package console

import (
	"context"
	"database/sql"
	"fmt"
	"strings"
)

// AppendAuditEvent appends an audit event to the immutable audit trail.
func (s *SQLiteStore) AppendAuditEvent(ctx context.Context, event *AuditEvent) error {
	if event == nil {
		return fmt.Errorf("audit event is required")
	}
	if event.Action == "" {
		return fmt.Errorf("action is required")
	}

	ensureID(&event.ID)
	if event.ActorType == "" {
		event.ActorType = PrincipalTypeUser
	}
	if event.Outcome == "" {
		event.Outcome = AuditOutcomeSuccess
	}
	defaultOccurredAt(&event.OccurredAt)

	metadata, err := metadataJSON(event.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode audit event metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_audit_events (
				id, actor_type, actor_id, action, target_type, target_id,
				outcome, message, metadata_json, occurred_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			event.ID,
			event.ActorType,
			nullableString(event.ActorID),
			event.Action,
			nullableString(event.TargetType),
			nullableString(event.TargetID),
			event.Outcome,
			nullableString(event.Message),
			metadata,
			event.OccurredAt,
		)
		return err
	})
}

// ListAuditEvents returns audit events matching the supplied filter.
func (s *SQLiteStore) ListAuditEvents(ctx context.Context, filter AuditEventFilter) ([]AuditEvent, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]AuditEvent, error) {
		query := `
			SELECT id, actor_type, actor_id, action, target_type, target_id,
			       outcome, message, metadata_json, occurred_at
			FROM console_audit_events
		`

		var clauses []string
		var args []interface{}
		if filter.ActorID != "" {
			clauses = append(clauses, "actor_id = ?")
			args = append(args, filter.ActorID)
		}
		if filter.Action != "" {
			clauses = append(clauses, "action = ?")
			args = append(args, filter.Action)
		}
		if filter.TargetType != "" {
			clauses = append(clauses, "target_type = ?")
			args = append(args, filter.TargetType)
		}
		if filter.TargetID != "" {
			clauses = append(clauses, "target_id = ?")
			args = append(args, filter.TargetID)
		}
		if filter.Outcome != "" {
			clauses = append(clauses, "outcome = ?")
			args = append(args, filter.Outcome)
		}
		if len(clauses) > 0 {
			query += " WHERE " + strings.Join(clauses, " AND ")
		}
		query += " ORDER BY occurred_at DESC LIMIT ?"
		args = append(args, normalizedLimit(filter.Limit))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		events, err := scanAuditEvents(rows)
		if err != nil {
			return nil, err
		}
		return events, nil
	})
}

func scanAuditEvents(rows *sql.Rows) ([]AuditEvent, error) {
	var events []AuditEvent
	for rows.Next() {
		event, err := scanAuditEvent(rows)
		if err != nil {
			return nil, err
		}
		events = append(events, event)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return events, nil
}

func scanAuditEvent(rows *sql.Rows) (AuditEvent, error) {
	var event AuditEvent
	var actorID sql.NullString
	var targetType sql.NullString
	var targetID sql.NullString
	var message sql.NullString
	var metadata string

	scanErr := rows.Scan(
		&event.ID,
		&event.ActorType,
		&actorID,
		&event.Action,
		&targetType,
		&targetID,
		&event.Outcome,
		&message,
		&metadata,
		&event.OccurredAt,
	)
	if scanErr != nil {
		return AuditEvent{}, scanErr
	}

	decodedMetadata, err := decodeMetadata(metadata)
	if err != nil {
		return AuditEvent{}, fmt.Errorf("failed to decode audit metadata: %w", err)
	}

	event.ActorID = scanOptionalString(actorID)
	event.TargetType = scanOptionalString(targetType)
	event.TargetID = scanOptionalString(targetID)
	event.Message = scanOptionalString(message)
	event.Metadata = decodedMetadata
	return event, nil
}
