package console

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

// SaveConfigRevision creates or updates a config revision.
func (s *SQLiteStore) SaveConfigRevision(ctx context.Context, revision *ConfigRevision) error {
	if revision == nil {
		return fmt.Errorf("config revision is required")
	}
	if len(revision.DocumentJSON) == 0 {
		return fmt.Errorf("document_json is required")
	}

	ensureID(&revision.ID)
	if revision.Status == "" {
		revision.Status = ConfigRevisionStatusDraft
	}
	ensureCreatedUpdated(&revision.CreatedAt, &revision.UpdatedAt)

	metadata, err := metadataJSON(revision.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode config revision metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_config_revisions (
				id, parent_revision_id, status, source, summary, document_json,
				runtime_config_yaml, created_by, activated_at, metadata_json,
				created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				parent_revision_id = excluded.parent_revision_id,
				status = excluded.status,
				source = excluded.source,
				summary = excluded.summary,
				document_json = excluded.document_json,
				runtime_config_yaml = excluded.runtime_config_yaml,
				created_by = excluded.created_by,
				activated_at = excluded.activated_at,
				metadata_json = excluded.metadata_json,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			revision.ID,
			nullableString(revision.ParentRevisionID),
			revision.Status,
			nullableString(revision.Source),
			nullableString(revision.Summary),
			string(revision.DocumentJSON),
			nullableString(revision.RuntimeConfigYAML),
			nullableString(revision.CreatedBy),
			nullTime(revision.ActivatedAt),
			metadata,
			revision.CreatedAt,
			revision.UpdatedAt,
		)
		return err
	})
}

// GetConfigRevision fetches a config revision by ID.
func (s *SQLiteStore) GetConfigRevision(ctx context.Context, id string) (*ConfigRevision, error) {
	if id == "" {
		return nil, fmt.Errorf("config revision id is required")
	}

	return withReadLock(ctx, s, func(ctx context.Context) (*ConfigRevision, error) {
		query := `
			SELECT id, parent_revision_id, status, source, summary, document_json,
			       runtime_config_yaml, created_by, activated_at, metadata_json,
			       created_at, updated_at
			FROM console_config_revisions
			WHERE id = ?
		`

		var revision ConfigRevision
		var parentID sql.NullString
		var source sql.NullString
		var summary sql.NullString
		var runtimeYAML sql.NullString
		var createdBy sql.NullString
		var activatedAt sql.NullTime
		var documentJSON string
		var metadata string

		err := s.db.QueryRowContext(ctx, query, id).Scan(
			&revision.ID,
			&parentID,
			&revision.Status,
			&source,
			&summary,
			&documentJSON,
			&runtimeYAML,
			&createdBy,
			&activatedAt,
			&metadata,
			&revision.CreatedAt,
			&revision.UpdatedAt,
		)
		if errors.Is(err, sql.ErrNoRows) {
			return nil, nil
		}
		if err != nil {
			return nil, err
		}

		revision.ParentRevisionID = scanOptionalString(parentID)
		revision.Source = scanOptionalString(source)
		revision.Summary = scanOptionalString(summary)
		revision.DocumentJSON = []byte(documentJSON)
		revision.RuntimeConfigYAML = scanOptionalString(runtimeYAML)
		revision.CreatedBy = scanOptionalString(createdBy)
		revision.ActivatedAt = scanOptionalTime(activatedAt)
		revision.Metadata, err = decodeMetadata(metadata)
		if err != nil {
			return nil, fmt.Errorf("failed to decode config revision metadata: %w", err)
		}

		return &revision, nil
	})
}

// ListConfigRevisions returns config revisions matching the supplied filter.
func (s *SQLiteStore) ListConfigRevisions(ctx context.Context, filter ConfigRevisionFilter) ([]ConfigRevision, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]ConfigRevision, error) {
		query := `
			SELECT id, parent_revision_id, status, source, summary, document_json,
			       runtime_config_yaml, created_by, activated_at, metadata_json,
			       created_at, updated_at
			FROM console_config_revisions
		`

		var clauses []string
		var args []interface{}
		if filter.Status != "" {
			clauses = append(clauses, "status = ?")
			args = append(args, filter.Status)
		}
		if filter.Source != "" {
			clauses = append(clauses, "source = ?")
			args = append(args, filter.Source)
		}
		if len(clauses) > 0 {
			query += " WHERE " + strings.Join(clauses, " AND ")
		}
		query += " ORDER BY created_at DESC LIMIT ?"
		args = append(args, normalizedLimit(filter.Limit))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		revisions, err := scanConfigRevisions(rows)
		if err != nil {
			return nil, err
		}
		return revisions, nil
	})
}

// SaveDeployEvent creates or updates a deploy event.
func (s *SQLiteStore) SaveDeployEvent(ctx context.Context, event *DeployEvent) error {
	if event == nil {
		return fmt.Errorf("deploy event is required")
	}
	if event.RevisionID == "" {
		return fmt.Errorf("revision_id is required")
	}

	ensureID(&event.ID)
	if event.Status == "" {
		event.Status = DeployEventStatusPending
	}
	ensureCreatedUpdated(&event.CreatedAt, &event.UpdatedAt)

	metadata, err := metadataJSON(event.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode deploy event metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_deploy_events (
				id, revision_id, status, trigger_source, message, runtime_target,
				rollback_revision_id, metadata_json, started_at, completed_at,
				created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				revision_id = excluded.revision_id,
				status = excluded.status,
				trigger_source = excluded.trigger_source,
				message = excluded.message,
				runtime_target = excluded.runtime_target,
				rollback_revision_id = excluded.rollback_revision_id,
				metadata_json = excluded.metadata_json,
				started_at = excluded.started_at,
				completed_at = excluded.completed_at,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			event.ID,
			event.RevisionID,
			event.Status,
			nullableString(event.TriggerSource),
			nullableString(event.Message),
			nullableString(event.RuntimeTarget),
			nullableString(event.RollbackRevisionID),
			metadata,
			nullTime(event.StartedAt),
			nullTime(event.CompletedAt),
			event.CreatedAt,
			event.UpdatedAt,
		)
		return err
	})
}

// ListDeployEvents returns deploy events matching the supplied filter.
func (s *SQLiteStore) ListDeployEvents(ctx context.Context, filter DeployEventFilter) ([]DeployEvent, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]DeployEvent, error) {
		query := `
			SELECT id, revision_id, status, trigger_source, message, runtime_target,
			       rollback_revision_id, metadata_json, started_at, completed_at,
			       created_at, updated_at
			FROM console_deploy_events
		`

		var clauses []string
		var args []interface{}
		if filter.RevisionID != "" {
			clauses = append(clauses, "revision_id = ?")
			args = append(args, filter.RevisionID)
		}
		if filter.Status != "" {
			clauses = append(clauses, "status = ?")
			args = append(args, filter.Status)
		}
		if len(clauses) > 0 {
			query += " WHERE " + strings.Join(clauses, " AND ")
		}
		query += " ORDER BY created_at DESC LIMIT ?"
		args = append(args, normalizedLimit(filter.Limit))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		events, err := scanDeployEvents(rows)
		if err != nil {
			return nil, err
		}
		return events, nil
	})
}

// SaveSecretRef creates or updates a secret reference.
func (s *SQLiteStore) SaveSecretRef(ctx context.Context, ref *SecretRef) error {
	if ref == nil {
		return fmt.Errorf("secret ref is required")
	}
	if ref.ExternalRef == "" {
		return fmt.Errorf("external_ref is required")
	}

	ensureID(&ref.ID)
	if ref.ScopeType == "" {
		ref.ScopeType = ScopeTypeGlobal
	}
	ensureCreatedUpdated(&ref.CreatedAt, &ref.UpdatedAt)

	metadata, err := metadataJSON(ref.Metadata)
	if err != nil {
		return fmt.Errorf("failed to encode secret ref metadata: %w", err)
	}

	return withWriteLock(ctx, s, func(ctx context.Context) error {
		query := `
			INSERT INTO console_secret_refs (
				id, scope_type, scope_id, provider, external_ref, version,
				redacted_label, last_rotated_at, metadata_json, created_at, updated_at
			) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
			ON CONFLICT(id) DO UPDATE SET
				scope_type = excluded.scope_type,
				scope_id = excluded.scope_id,
				provider = excluded.provider,
				external_ref = excluded.external_ref,
				version = excluded.version,
				redacted_label = excluded.redacted_label,
				last_rotated_at = excluded.last_rotated_at,
				metadata_json = excluded.metadata_json,
				updated_at = excluded.updated_at
		`

		_, err := s.db.ExecContext(
			ctx,
			query,
			ref.ID,
			ref.ScopeType,
			nullableString(ref.ScopeID),
			nullableString(ref.Provider),
			ref.ExternalRef,
			nullableString(ref.Version),
			nullableString(ref.RedactedLabel),
			nullTime(ref.LastRotatedAt),
			metadata,
			ref.CreatedAt,
			ref.UpdatedAt,
		)
		return err
	})
}

// ListSecretRefs returns secret references matching the supplied filter.
func (s *SQLiteStore) ListSecretRefs(ctx context.Context, filter SecretRefFilter) ([]SecretRef, error) {
	return withReadLock(ctx, s, func(ctx context.Context) ([]SecretRef, error) {
		query := `
			SELECT id, scope_type, scope_id, provider, external_ref, version,
			       redacted_label, last_rotated_at, metadata_json, created_at, updated_at
			FROM console_secret_refs
		`

		var clauses []string
		var args []interface{}
		if filter.ScopeType != "" {
			clauses = append(clauses, "scope_type = ?")
			args = append(args, filter.ScopeType)
		}
		if filter.ScopeID != "" {
			clauses = append(clauses, "scope_id = ?")
			args = append(args, filter.ScopeID)
		}
		if len(clauses) > 0 {
			query += " WHERE " + strings.Join(clauses, " AND ")
		}
		query += " ORDER BY created_at DESC LIMIT ?"
		args = append(args, normalizedLimit(filter.Limit))

		rows, err := s.db.QueryContext(ctx, query, args...)
		if err != nil {
			return nil, err
		}
		defer func() {
			_ = rows.Close()
		}()

		refs, err := scanSecretRefs(rows)
		if err != nil {
			return nil, err
		}
		return refs, nil
	})
}

func nullableString(value string) interface{} {
	if value == "" {
		return nil
	}
	return value
}

func scanConfigRevisions(rows *sql.Rows) ([]ConfigRevision, error) {
	var revisions []ConfigRevision
	for rows.Next() {
		revision, err := scanConfigRevision(rows)
		if err != nil {
			return nil, err
		}
		revisions = append(revisions, revision)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return revisions, nil
}

func scanConfigRevision(rows *sql.Rows) (ConfigRevision, error) {
	var revision ConfigRevision
	var parentID sql.NullString
	var source sql.NullString
	var summary sql.NullString
	var runtimeYAML sql.NullString
	var createdBy sql.NullString
	var activatedAt sql.NullTime
	var documentJSON string
	var metadata string

	scanErr := rows.Scan(
		&revision.ID,
		&parentID,
		&revision.Status,
		&source,
		&summary,
		&documentJSON,
		&runtimeYAML,
		&createdBy,
		&activatedAt,
		&metadata,
		&revision.CreatedAt,
		&revision.UpdatedAt,
	)
	if scanErr != nil {
		return ConfigRevision{}, scanErr
	}

	decodedMetadata, err := decodeMetadata(metadata)
	if err != nil {
		return ConfigRevision{}, fmt.Errorf("failed to decode config revision metadata: %w", err)
	}

	revision.ParentRevisionID = scanOptionalString(parentID)
	revision.Source = scanOptionalString(source)
	revision.Summary = scanOptionalString(summary)
	revision.DocumentJSON = []byte(documentJSON)
	revision.RuntimeConfigYAML = scanOptionalString(runtimeYAML)
	revision.CreatedBy = scanOptionalString(createdBy)
	revision.ActivatedAt = scanOptionalTime(activatedAt)
	revision.Metadata = decodedMetadata
	return revision, nil
}

func scanDeployEvents(rows *sql.Rows) ([]DeployEvent, error) {
	var events []DeployEvent
	for rows.Next() {
		event, err := scanDeployEvent(rows)
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

func scanDeployEvent(rows *sql.Rows) (DeployEvent, error) {
	var event DeployEvent
	var triggerSource sql.NullString
	var message sql.NullString
	var runtimeTarget sql.NullString
	var rollbackRevisionID sql.NullString
	var metadata string
	var startedAt sql.NullTime
	var completedAt sql.NullTime

	scanErr := rows.Scan(
		&event.ID,
		&event.RevisionID,
		&event.Status,
		&triggerSource,
		&message,
		&runtimeTarget,
		&rollbackRevisionID,
		&metadata,
		&startedAt,
		&completedAt,
		&event.CreatedAt,
		&event.UpdatedAt,
	)
	if scanErr != nil {
		return DeployEvent{}, scanErr
	}

	decodedMetadata, err := decodeMetadata(metadata)
	if err != nil {
		return DeployEvent{}, fmt.Errorf("failed to decode deploy event metadata: %w", err)
	}

	event.TriggerSource = scanOptionalString(triggerSource)
	event.Message = scanOptionalString(message)
	event.RuntimeTarget = scanOptionalString(runtimeTarget)
	event.RollbackRevisionID = scanOptionalString(rollbackRevisionID)
	event.StartedAt = scanOptionalTime(startedAt)
	event.CompletedAt = scanOptionalTime(completedAt)
	event.Metadata = decodedMetadata
	return event, nil
}

func scanSecretRefs(rows *sql.Rows) ([]SecretRef, error) {
	var refs []SecretRef
	for rows.Next() {
		ref, err := scanSecretRef(rows)
		if err != nil {
			return nil, err
		}
		refs = append(refs, ref)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	return refs, nil
}

func scanSecretRef(rows *sql.Rows) (SecretRef, error) {
	var ref SecretRef
	var scopeID sql.NullString
	var provider sql.NullString
	var version sql.NullString
	var redactedLabel sql.NullString
	var lastRotatedAt sql.NullTime
	var metadata string

	scanErr := rows.Scan(
		&ref.ID,
		&ref.ScopeType,
		&scopeID,
		&provider,
		&ref.ExternalRef,
		&version,
		&redactedLabel,
		&lastRotatedAt,
		&metadata,
		&ref.CreatedAt,
		&ref.UpdatedAt,
	)
	if scanErr != nil {
		return SecretRef{}, scanErr
	}

	decodedMetadata, err := decodeMetadata(metadata)
	if err != nil {
		return SecretRef{}, fmt.Errorf("failed to decode secret ref metadata: %w", err)
	}

	ref.ScopeID = scanOptionalString(scopeID)
	ref.Provider = scanOptionalString(provider)
	ref.Version = scanOptionalString(version)
	ref.RedactedLabel = scanOptionalString(redactedLabel)
	ref.LastRotatedAt = scanOptionalTime(lastRotatedAt)
	ref.Metadata = decodedMetadata
	return ref, nil
}
