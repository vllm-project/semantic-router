// Package postgres persists managed routing desired state and immutable
// revisions in PostgreSQL. Every mutation appends audit and outbox records in
// the same transaction as the resource change.
package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

type Store struct {
	db                  *sql.DB
	validatePublication routingmanagement.PublicationValidator
}

func New(db *sql.DB, validatePublication routingmanagement.PublicationValidator) (*Store, error) {
	if db == nil || validatePublication == nil {
		return nil, fmt.Errorf("routing Management PostgreSQL database and publication validator are required")
	}
	return &Store{db: db, validatePublication: validatePublication}, nil
}

func inTransaction[T any](ctx context.Context, store *Store, fn func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelSerializable})
	if err != nil {
		return zero, fmt.Errorf("begin routing Management transaction: %w", err)
	}
	value, err := fn(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		if serializationFailure(err) {
			return zero, routingmanagement.ErrConflict
		}
		return zero, fmt.Errorf("commit routing Management transaction: %w", err)
	}
	return value, nil
}

func inReadTransaction[T any](ctx context.Context, store *Store, fn func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return zero, fmt.Errorf("begin routing Management read: %w", err)
	}
	value, err := fn(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit routing Management read: %w", err)
	}
	return value, nil
}

func serializationFailure(err error) bool {
	var postgresError *pq.Error
	return errors.As(err, &postgresError) && (postgresError.Code == "40001" || postgresError.Code == "40P01")
}

func classifyWriteError(err error) error {
	if err == nil {
		return nil
	}
	var postgresError *pq.Error
	if errors.As(err, &postgresError) {
		switch postgresError.Code {
		case "23505", "40001", "40P01":
			return routingmanagement.ErrConflict
		case "23503":
			return routingmanagement.ErrReferenced
		case "23514", "22P02", "22003":
			return fmt.Errorf("%w: PostgreSQL rejected canonical routing data", routingmanagement.ErrInvalid)
		}
	}
	return err
}

func (store *Store) NamespaceCurrency(ctx context.Context, namespaceID string) (string, error) {
	var currency string
	err := store.db.QueryRowContext(ctx, `SELECT billing_currency FROM access_namespaces
WHERE id = $1 AND status = 'active'`, namespaceID).Scan(&currency)
	if errors.Is(err, sql.ErrNoRows) {
		return "", routingmanagement.ErrNotFound
	}
	if err != nil {
		return "", fmt.Errorf("read routing namespace currency: %w", err)
	}
	return currency, nil
}
