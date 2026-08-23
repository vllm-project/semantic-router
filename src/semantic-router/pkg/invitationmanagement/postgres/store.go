package postgres

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/lib/pq"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	commandpostgres "github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand/postgres"
)

type Store struct {
	database *sql.DB
}

func New(database *sql.DB) (*Store, error) {
	if database == nil {
		return nil, invitationmanagement.ErrUnavailable
	}
	return &Store{database: database}, nil
}

func (store *Store) Ready(
	ctx context.Context,
	commands *managementcommand.Codec,
	invitationVersions []string,
	responseVersions []string,
) (returnErr error) {
	if store == nil || store.database == nil || commands == nil || len(invitationVersions) == 0 || len(responseVersions) == 0 {
		return invitationmanagement.ErrUnavailable
	}
	if err := commandpostgres.ValidateReferencedHMACVersions(ctx, store.database, commands); err != nil {
		return err
	}
	rows, queryContextErr := store.database.QueryContext(ctx, `SELECT DISTINCT pepper_version
FROM management_invitations
WHERE (status='pending' AND expires_at>clock_timestamp())
   OR (status='accepted' AND acceptance_result_expires_at>clock_timestamp())`)
	if queryContextErr != nil {
		return fmt.Errorf("read invitation token pepper versions: %w", queryContextErr)
	}
	defer func(openRows *sql.Rows) {
		returnErr = errors.Join(returnErr, openRows.Close())
	}(rows)
	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			return fmt.Errorf("scan invitation token pepper version: %w", err)
		}
		if !slices.Contains(invitationVersions, version) {
			return fmt.Errorf("invitation token pepper %q is unavailable", version)
		}
	}
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate invitation token pepper versions: %w", err)
	}
	if err := rows.Close(); err != nil {
		return fmt.Errorf("close invitation token pepper versions: %w", err)
	}
	rows, queryContextErr = store.database.QueryContext(ctx, `SELECT DISTINCT acceptance_response_kek_version
FROM management_invitations
WHERE status='accepted' AND acceptance_result_expires_at>clock_timestamp()
  AND acceptance_response_kek_version IS NOT NULL
UNION
SELECT DISTINCT response_kek_version FROM management_idempotency
WHERE expires_at>clock_timestamp() AND secret_response_expires_at>clock_timestamp()
  AND (endpoint LIKE '/management/v1/invitations%' OR endpoint='/management/v1/onboarding')`)
	if queryContextErr != nil {
		return fmt.Errorf("read invitation response KEK versions: %w", queryContextErr)
	}
	defer func(openRows *sql.Rows) {
		returnErr = errors.Join(returnErr, openRows.Close())
	}(rows)
	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			return fmt.Errorf("scan invitation response KEK version: %w", err)
		}
		if !slices.Contains(responseVersions, version) {
			return fmt.Errorf("invitation response KEK %q is unavailable", version)
		}
	}
	return rows.Err()
}

func inTransaction[T any](ctx context.Context, store *Store, isolation sql.IsolationLevel, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	for attempt := 0; attempt < 4; attempt++ {
		tx, err := store.database.BeginTx(ctx, &sql.TxOptions{Isolation: isolation})
		if err != nil {
			return zero, fmt.Errorf("begin invitation transaction: %w", err)
		}
		value, err := operation(tx)
		if err != nil {
			var committed *commitError
			if errors.As(err, &committed) {
				if commitErr := tx.Commit(); commitErr != nil {
					if retryableTransactionError(commitErr) && ctx.Err() == nil && attempt < 3 {
						continue
					}
					return zero, fmt.Errorf("commit invitation transaction: %w", commitErr)
				}
				return zero, committed.err
			}
			_ = tx.Rollback()
			if retryableTransactionError(err) && ctx.Err() == nil && attempt < 3 {
				continue
			}
			return zero, err
		}
		if err := tx.Commit(); err != nil {
			if retryableTransactionError(err) && ctx.Err() == nil && attempt < 3 {
				continue
			}
			return zero, fmt.Errorf("commit invitation transaction: %w", err)
		}
		return value, nil
	}
	return zero, invitationmanagement.ErrUnavailable
}

type commitError struct{ err error }

func (err *commitError) Error() string { return err.err.Error() }
func (err *commitError) Unwrap() error { return err.err }

func commitThen(err error) error { return &commitError{err: err} }

func retryableTransactionError(err error) bool {
	var postgresError *pq.Error
	return errors.As(err, &postgresError) && (postgresError.Code == "40001" || postgresError.Code == "40P01")
}

func mapCommandError(err error) error {
	if errors.Is(err, managementcommand.ErrConflict) {
		return invitationmanagement.ErrConflict
	}
	return err
}

func storedSecret(result managementcommand.StoredResult, resourceType string) (invitationmanagement.StoredSecret, error) {
	if result.Resource == nil || result.Secret == nil || result.Operation != nil || result.Resource.ResourceType != resourceType {
		return invitationmanagement.StoredSecret{}, invitationmanagement.ErrUnavailable
	}
	return invitationmanagement.StoredSecret{Result: *result.Resource, Secret: *result.Secret}, nil
}

func (store *Store) ReplaySecret(ctx context.Context, command managementcommand.Command) (invitationmanagement.StoredSecret, bool, error) {
	result, found, err := commandpostgres.Lookup(ctx, store.database, command)
	if err != nil || !found {
		return invitationmanagement.StoredSecret{}, false, mapCommandError(err)
	}
	if result.Resource == nil {
		return invitationmanagement.StoredSecret{}, false, invitationmanagement.ErrUnavailable
	}
	secret, err := storedSecret(result, result.Resource.ResourceType)
	return secret, true, err
}

func databaseNow(ctx context.Context, tx *sql.Tx) (time.Time, error) {
	var now time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&now); err != nil {
		return time.Time{}, fmt.Errorf("read PostgreSQL time: %w", err)
	}
	return now.UTC().Truncate(time.Microsecond), nil
}
