// Package postgres persists Router-native Agent state. PostgreSQL is the
// durable queue, event sequence, idempotency, and fencing authority; an
// optional notifier only accelerates wakeups and event delivery.
package postgres

import (
	"context"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

type Store struct {
	db       *sql.DB
	commands *managementcommand.Codec
}

func New(db *sql.DB, commands *managementcommand.Codec) (*Store, error) {
	if db == nil || commands == nil {
		return nil, fmt.Errorf("agent PostgreSQL database and Management command codec are required")
	}
	return &Store{db: db, commands: commands}, nil
}

func (store *Store) Ready(ctx context.Context) error {
	if store == nil || store.db == nil || store.commands == nil {
		return errors.New("agent PostgreSQL store is unavailable")
	}
	if err := store.db.PingContext(ctx); err != nil {
		return fmt.Errorf("agent PostgreSQL is unavailable: %w", err)
	}
	if err := commandpostgres.ValidateReferencedHMACVersions(ctx, store.db, store.commands); err != nil {
		return fmt.Errorf("agent idempotency keyring is not ready: %w", err)
	}
	return nil
}

func inTransaction[T any](ctx context.Context, store *Store, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return zero, fmt.Errorf("begin Agent transaction: %w", err)
	}
	value, err := operation(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, classifyWriteError(err)
	}
	return value, nil
}

func classifyWriteError(err error) error {
	if err == nil {
		return nil
	}
	var databaseError *pq.Error
	if errors.As(err, &databaseError) {
		switch databaseError.Code {
		case "23505", "40001", "40P01":
			return agentmanagement.ErrConflict
		case "23503":
			return fmt.Errorf("%w: referenced Agent resource is unavailable", agentmanagement.ErrConflict)
		case "23514", "22P02", "22003":
			return fmt.Errorf("%w: PostgreSQL rejected Agent state", agentmanagement.ErrInvalid)
		}
	}
	return err
}

func mapNotFound(err error) error {
	if errors.Is(err, sql.ErrNoRows) {
		return agentmanagement.ErrNotFound
	}
	return err
}

func parseDigest(value string) ([]byte, error) {
	if value == "" {
		return nil, nil
	}
	if !strings.HasPrefix(value, "sha256:") {
		return nil, agentmanagement.ErrInvalid
	}
	decoded, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	if err != nil || len(decoded) != 32 {
		return nil, agentmanagement.ErrInvalid
	}
	return decoded, nil
}

func nullableBytes(value []byte) any {
	if len(value) == 0 {
		return nil
	}
	return value
}

var _ agentmanagement.Store = (*Store)(nil)
