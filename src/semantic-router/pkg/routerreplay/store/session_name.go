package store

import (
	"context"
	"fmt"
)

// RequestKindSessionTitle marks the side request a harness sends to name its
// session: Claude Code's auxiliary title request and the thread Codex starts
// for a title.
const RequestKindSessionTitle = "session_title"

// The other requests that are no turn of their agent: a Claude Code side
// request by the class Claude Code gives it, auxiliary or compaction, and a
// check a tool marks with x-vsr-probe.
const (
	RequestKindAuxiliary  = "auxiliary"
	RequestKindCompaction = "compaction"
	RequestKindProbe      = "probe"
)

// UpdateSessionName stores the session name a record's answer gave.
func (m *MemoryStore) UpdateSessionName(ctx context.Context, id string, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}
	rec.SessionName = name
	return nil
}

// UpdateSessionName stores the session name a record's answer gave.
func (r *RedisStore) UpdateSessionName(ctx context.Context, id string, name string) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		record.SessionName = name
		return true
	})
}

// UpdateSessionName stores the session name a record's answer gave.
func (m *MilvusStore) UpdateSessionName(ctx context.Context, id string, name string) error {
	return m.updateRecord(ctx, id, func(record *Record) { record.SessionName = name })
}

// UpdateSessionName stores the session name a record's answer gave.
func (q *QdrantStore) UpdateSessionName(ctx context.Context, id string, name string) error {
	return q.updateRecord(ctx, id, func(record *Record) { record.SessionName = name })
}

// UpdateSessionName stores the session name a record's answer gave in its
// routing metadata, beside its agent.
func (p *PostgresStore) UpdateSessionName(ctx context.Context, id string, name string) error {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET routing_metadata = COALESCE(routing_metadata, '{}'::jsonb) || jsonb_build_object('session_name', $2::text)
		WHERE id = $1
	`, p.tableName)
	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, sanitizePostgresText(name))
		if err != nil {
			return fmt.Errorf("failed to update session name: %w", err)
		}
		rows, err := result.RowsAffected()
		if err != nil {
			return err
		}
		if rows == 0 {
			return fmt.Errorf("record with ID %s not found", id)
		}
		return nil
	}
	if p.asyncWrites {
		return p.runAsyncAndWait(ctx, fn)
	}
	return fn()
}
