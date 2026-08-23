package outcomefeedback

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/google/uuid"
)

type PostgresRepository struct {
	database *sql.DB
}

func NewPostgresRepository(database *sql.DB) (*PostgresRepository, error) {
	if database == nil {
		return nil, errors.New("outcome PostgreSQL database is required")
	}
	return &PostgresRepository{database: database}, nil
}

func (repository *PostgresRepository) Record(
	ctx context.Context,
	caller Caller,
	idempotencyKey string,
	request Request,
) (Receipt, error) {
	if repository == nil || repository.database == nil {
		return Receipt{}, ErrUnavailable
	}
	if err := caller.Validate(); err != nil {
		return Receipt{}, err
	}
	if err := ValidateIdempotencyKey(idempotencyKey); err != nil {
		return Receipt{}, err
	}
	if err := request.Validate(); err != nil {
		return Receipt{}, err
	}
	requestDigest, recordErr := RequestDigest(request)
	if recordErr != nil {
		return Receipt{}, recordErr
	}
	idempotencyDigest := IdempotencyDigest(caller, request.ReplayID, idempotencyKey)

	transaction, recordErr := repository.database.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if recordErr != nil {
		return Receipt{}, fmt.Errorf("%w: begin outcome transaction", ErrUnavailable)
	}
	defer func() { _ = transaction.Rollback() }()

	replay, recordErr := loadOwnedReplay(ctx, transaction, caller, request)
	if recordErr != nil {
		return Receipt{}, recordErr
	}
	receiptID := uuid.NewString()
	inserted, recordErr := claimIdempotency(
		ctx, transaction, caller, request.ReplayID, idempotencyDigest[:], requestDigest[:], receiptID,
	)
	if recordErr != nil {
		return Receipt{}, recordErr
	}
	if !inserted {
		receipt, err := loadDuplicateReceipt(
			ctx, transaction, caller, request.ReplayID, idempotencyDigest[:], requestDigest[:],
		)
		if err != nil {
			return Receipt{}, err
		}
		if err := transaction.Commit(); err != nil {
			return Receipt{}, fmt.Errorf("%w: commit duplicate outcome receipt", ErrUnavailable)
		}
		return receipt, nil
	}

	revision, recordErr := advanceProjectionRevision(ctx, transaction, caller.NamespaceID)
	if recordErr != nil {
		return Receipt{}, recordErr
	}
	createdAt := time.Now().UTC()
	if err := insertOutcome(ctx, transaction, receiptID, caller, replay, request, requestDigest[:], revision, createdAt); err != nil {
		return Receipt{}, err
	}
	if err := enqueueProjection(ctx, transaction, receiptID, caller.NamespaceID, revision, createdAt); err != nil {
		return Receipt{}, err
	}
	if err := transaction.Commit(); err != nil {
		return Receipt{}, fmt.Errorf("%w: commit inference outcome", ErrUnavailable)
	}
	return Receipt{
		ID: receiptID, ReplayID: request.ReplayID,
		ProjectionRevision: revision, CreatedAt: createdAt,
	}, nil
}

func loadOwnedReplay(
	ctx context.Context,
	transaction *sql.Tx,
	caller Caller,
	request Request,
) (ReplayRecord, error) {
	var (
		record      ReplayRecord
		userID      sql.NullString
		teamID      sql.NullString
		routingJSON []byte
		modelsJSON  []byte
	)
	err := transaction.QueryRowContext(ctx, `SELECT
  namespace_id::text, replay_id, api_key_id::text, user_id::text, team_id::text,
  routing_context, served_models, created_at
FROM inference_replays
WHERE namespace_id=$1 AND replay_id=$2
FOR SHARE`, caller.NamespaceID, request.ReplayID).Scan(
		&record.NamespaceID, &record.ReplayID, &record.APIKeyID, &userID, &teamID,
		&routingJSON, &modelsJSON, &record.CreatedAt,
	)
	if errors.Is(err, sql.ErrNoRows) {
		return ReplayRecord{}, ErrNotFound
	}
	if err != nil {
		return ReplayRecord{}, fmt.Errorf("%w: read inference replay", ErrUnavailable)
	}
	record.UserID, record.TeamID = userID.String, teamID.String
	if err := decodeStrictJSON(routingJSON, &record.Routing); err != nil {
		return ReplayRecord{}, fmt.Errorf("%w: replay routing context is corrupt", ErrUnavailable)
	}
	if err := decodeStrictJSON(modelsJSON, &record.Models); err != nil {
		return ReplayRecord{}, fmt.Errorf("%w: replay served-model context is corrupt", ErrUnavailable)
	}
	if !record.Owns(caller) {
		return ReplayRecord{}, ErrNotFound
	}
	if request.Target == TargetModel && !record.Served(request.TargetRef, *request.TargetRevision) {
		return ReplayRecord{}, ErrNotFound
	}
	return record, nil
}

func claimIdempotency(
	ctx context.Context,
	transaction *sql.Tx,
	caller Caller,
	replayID string,
	idempotencyDigest []byte,
	requestDigest []byte,
	receiptID string,
) (bool, error) {
	result, err := transaction.ExecContext(ctx, `INSERT INTO inference_outcome_idempotency (
  namespace_id, api_key_id, replay_id, idempotency_digest, request_digest, receipt_id
) VALUES ($1,$2,$3,$4,$5,$6)
ON CONFLICT (namespace_id, api_key_id, replay_id, idempotency_digest) DO NOTHING`,
		caller.NamespaceID, caller.APIKeyID, replayID, idempotencyDigest, requestDigest, receiptID,
	)
	if err != nil {
		return false, fmt.Errorf("%w: claim outcome idempotency", ErrUnavailable)
	}
	rows, err := result.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("%w: inspect outcome idempotency claim", ErrUnavailable)
	}
	return rows == 1, nil
}

func loadDuplicateReceipt(
	ctx context.Context,
	transaction *sql.Tx,
	caller Caller,
	replayID string,
	idempotencyDigest []byte,
	requestDigest []byte,
) (Receipt, error) {
	var (
		storedDigest []byte
		receipt      Receipt
	)
	err := transaction.QueryRowContext(ctx, `SELECT i.request_digest, o.id::text,
  o.replay_id, o.projection_revision, o.created_at
FROM inference_outcome_idempotency i
JOIN inference_outcomes o
  ON o.namespace_id=i.namespace_id AND o.id=i.receipt_id
WHERE i.namespace_id=$1 AND i.api_key_id=$2 AND i.replay_id=$3
  AND i.idempotency_digest=$4
FOR SHARE`, caller.NamespaceID, caller.APIKeyID, replayID, idempotencyDigest).Scan(
		&storedDigest, &receipt.ID, &receipt.ReplayID, &receipt.ProjectionRevision, &receipt.CreatedAt,
	)
	if err != nil {
		return Receipt{}, fmt.Errorf("%w: read committed outcome receipt", ErrUnavailable)
	}
	if !bytes.Equal(storedDigest, requestDigest) {
		return Receipt{}, ErrIdempotencyConflict
	}
	receipt.Duplicate = true
	return receipt, nil
}

func advanceProjectionRevision(ctx context.Context, transaction *sql.Tx, namespaceID string) (int64, error) {
	var revision int64
	err := transaction.QueryRowContext(ctx, `INSERT INTO inference_outcome_projection_heads (
  namespace_id, desired_revision, applied_revision
) VALUES ($1,1,0)
ON CONFLICT (namespace_id) DO UPDATE SET
  desired_revision=inference_outcome_projection_heads.desired_revision+1,
  updated_at=clock_timestamp()
RETURNING desired_revision`, namespaceID).Scan(&revision)
	if err != nil {
		return 0, fmt.Errorf("%w: advance outcome projection revision", ErrUnavailable)
	}
	return revision, nil
}

func insertOutcome(
	ctx context.Context,
	transaction *sql.Tx,
	receiptID string,
	caller Caller,
	replay ReplayRecord,
	request Request,
	requestDigest []byte,
	revision int64,
	createdAt time.Time,
) error {
	metadataValue := request.Metadata
	if metadataValue == nil {
		metadataValue = map[string]string{}
	}
	metadata, err := json.Marshal(metadataValue)
	if err != nil {
		return fmt.Errorf("encode inference outcome metadata: %w", err)
	}
	var score any
	if request.Score != nil {
		score = strconv.FormatFloat(*request.Score, 'f', 9, 64)
	}
	var targetRevision any
	if request.TargetRevision != nil {
		targetRevision = *request.TargetRevision
	}
	targetModelID, targetModelName := "", ""
	if request.Target == TargetModel {
		for _, model := range replay.Models {
			if model.Revision == *request.TargetRevision && (model.ID == request.TargetRef || model.Name == request.TargetRef) {
				targetModelID, targetModelName = model.ID, model.Name
				break
			}
		}
	}
	_, err = transaction.ExecContext(ctx, `INSERT INTO inference_outcomes (
  id, namespace_id, replay_id, api_key_id, user_id, team_id, source,
  target, target_ref, target_revision, target_model_id, target_model_name,
  verdict, reason, score, metadata, request_digest, projection_revision, created_at
) VALUES (
  $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19
)`, receiptID, caller.NamespaceID, request.ReplayID, caller.APIKeyID,
		nullString(caller.UserID), nullString(caller.TeamID), string(caller.Source),
		string(request.Target), nullString(request.TargetRef), targetRevision,
		nullString(targetModelID), nullString(targetModelName), string(request.Verdict),
		nullString(request.Reason), score, metadata, requestDigest, revision, createdAt,
	)
	if err != nil {
		return fmt.Errorf("%w: insert inference outcome: %w", ErrUnavailable, err)
	}
	return nil
}

func enqueueProjection(
	ctx context.Context,
	transaction *sql.Tx,
	outcomeID string,
	namespaceID string,
	revision int64,
	createdAt time.Time,
) error {
	_, err := transaction.ExecContext(ctx, `INSERT INTO inference_outcome_projection_outbox (
  outcome_id, namespace_id, desired_revision, state, available_at, created_at
) VALUES ($1,$2,$3,'pending',$4,$4)`, outcomeID, namespaceID, revision, createdAt)
	if err != nil {
		return fmt.Errorf("%w: enqueue outcome learning projection", ErrUnavailable)
	}
	return nil
}

func decodeStrictJSON(value []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(value))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("trailing JSON")
		}
		return err
	}
	return nil
}

func nullString(value string) any {
	if value == "" {
		return nil
	}
	return value
}
