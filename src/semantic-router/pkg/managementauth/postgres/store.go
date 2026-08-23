// Package postgres persists authoritative global Router Management sessions.
// Namespace authority remains in current scoped role/link state and is not a
// column or filter on this store.
package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementauth"
)

type Store struct {
	db *sql.DB
}

func New(db *sql.DB) (*Store, error) {
	if db == nil {
		return nil, errors.New("management session PostgreSQL database is required")
	}
	return &Store{db: db}, nil
}

func inTransaction[T any](
	ctx context.Context,
	store *Store,
	isolation sql.IsolationLevel,
	operation func(*sql.Tx) (T, error),
) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: isolation})
	if err != nil {
		return zero, fmt.Errorf("begin management session transaction: %w", err)
	}
	value, err := operation(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit management session transaction: %w", err)
	}
	return value, nil
}

var _ managementauth.SessionRepository = (*Store)(nil)
