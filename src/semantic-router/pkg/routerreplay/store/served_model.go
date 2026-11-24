package store

import (
	"context"
	"fmt"
)

// ServedModel is the model whose answer a request got and the reasoning it
// was sent with. It differs from the routed model after a fallback candidate
// answered in place of a failed one.
type ServedModel struct {
	Model           string
	ReasoningMode   string
	ReasoningEffort string
}

func (s ServedModel) apply(record *Record) {
	record.SelectedModel = s.Model
	record.ReasoningMode = s.ReasoningMode
	record.ReasoningEffort = s.ReasoningEffort
}

// ServedModelWriter names the model that answered an existing record.
type ServedModelWriter interface {
	UpdateServedModel(ctx context.Context, id string, served ServedModel) error
}

// UpdateServedModel names the model that answered a record.
func (m *MemoryStore) UpdateServedModel(ctx context.Context, id string, served ServedModel) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	rec, ok := m.byID[id]
	if !ok {
		return fmt.Errorf("record with ID %s not found", id)
	}
	served.apply(rec)
	return nil
}

// UpdateServedModel names the model that answered a record.
func (r *RedisStore) UpdateServedModel(ctx context.Context, id string, served ServedModel) error {
	return r.updateRecord(ctx, id, func(record *Record) bool {
		served.apply(record)
		return true
	})
}

// UpdateServedModel names the model that answered a record.
func (m *MilvusStore) UpdateServedModel(ctx context.Context, id string, served ServedModel) error {
	return m.updateRecord(ctx, id, served.apply)
}

// UpdateServedModel names the model that answered a record.
func (q *QdrantStore) UpdateServedModel(ctx context.Context, id string, served ServedModel) error {
	return q.updateRecord(ctx, id, served.apply)
}

// UpdateServedModel names the model that answered a record. The effort lives
// in the routing metadata, as written by Add.
func (p *PostgresStore) UpdateServedModel(ctx context.Context, id string, served ServedModel) error {
	release, err := p.lifecycle.beginMutation()
	if err != nil {
		return err
	}
	defer release()
	//nolint:gosec // tableName is validated during store creation
	query := fmt.Sprintf(`
		UPDATE %s
		SET selected_model = $2,
		    reasoning_mode = $3,
		    routing_metadata = (COALESCE(routing_metadata, '{}'::jsonb) - 'reasoning_effort') ||
		        CASE WHEN $4::text = '' THEN '{}'::jsonb ELSE jsonb_build_object('reasoning_effort', $4::text) END
		WHERE id = $1
	`, p.tableName)
	fn := func() error {
		result, err := p.db.ExecContext(ctx, query, id, sanitizePostgresText(served.Model),
			sanitizePostgresText(served.ReasoningMode), sanitizePostgresText(served.ReasoningEffort))
		if err != nil {
			return fmt.Errorf("failed to update served model: %w", err)
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
