package postgres

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type outboxOperation string

const (
	outboxCreated           outboxOperation = "created"
	outboxUpdated           outboxOperation = "updated"
	outboxDeleted           outboxOperation = "deleted"
	outboxCredentialRotated outboxOperation = "credential_rotated"
	outboxCredentialRevoked outboxOperation = "credential_revoked"
)

const (
	lockNamespaceQuery  = `SELECT runtime_epoch FROM access_namespaces WHERE id = $1 FOR UPDATE`
	nextRevisionQuery   = `SELECT COALESCE(MAX(revision), 0) + 1 FROM policy_revisions WHERE namespace_id = $1`
	insertRevisionQuery = `INSERT INTO policy_revisions
  (namespace_id, revision, runtime_epoch, reason, actor_principal_id)
VALUES ($1, $2, $3, $4, $5)`
	insertOutboxQuery = `INSERT INTO policy_outbox
  (id, namespace_id, desired_revision, aggregate_type, aggregate_id, operation, payload)
VALUES ($1, $2, $3, $4, $5, $6, $7)`
)

type outboxPayload struct {
	AggregateRevision string            `json:"aggregateRevision"`
	References        map[string]string `json:"references,omitempty"`
}

type outboxMutation struct {
	AggregateType     string
	AggregateID       string
	AggregateRevision accesscontrol.Revision
	Operation         outboxOperation
	References        map[string]string
}

func appendMutationRecords(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	mutation outboxMutation,
	meta MutationMeta,
) (MutationReceipt, error) {
	aggregateRevisionValue, appendMutationRecordsErr := validateOutboxMutation(namespaceID, mutation, meta)
	if appendMutationRecordsErr != nil {
		return MutationReceipt{}, appendMutationRecordsErr
	}
	runtimeEpoch, desiredRevision, appendMutationRecordsErr := allocateDesiredRevision(ctx, tx, namespaceID)
	if appendMutationRecordsErr != nil {
		return MutationReceipt{}, appendMutationRecordsErr
	}
	if _, err := tx.ExecContext(ctx, insertRevisionQuery,
		namespaceID, desiredRevision, runtimeEpoch, meta.Reason, actorValue(meta.ActorPrincipalID)); err != nil {
		return MutationReceipt{}, fmt.Errorf("insert policy revision: %w", err)
	}
	payload, appendMutationRecordsErr := json.Marshal(outboxPayload{
		AggregateRevision: strconv.FormatInt(aggregateRevisionValue, 10),
		References:        mutation.References,
	})
	if appendMutationRecordsErr != nil {
		return MutationReceipt{}, fmt.Errorf("encode outbox payload: %w", appendMutationRecordsErr)
	}
	if _, err := tx.ExecContext(ctx, insertOutboxQuery,
		uuid.NewString(), namespaceID, desiredRevision, mutation.AggregateType,
		mutation.AggregateID, mutation.Operation, payload); err != nil {
		return MutationReceipt{}, fmt.Errorf("insert policy outbox: %w", err)
	}
	if err := appendAuditEvent(ctx, tx, namespaceID, mutation, meta, desiredRevision); err != nil {
		return MutationReceipt{}, err
	}
	return MutationReceipt{DesiredRevision: accesscontrol.Revision(desiredRevision)}, nil
}

func validateOutboxMutation(
	namespaceID accesscontrol.NamespaceID,
	mutation outboxMutation,
	meta MutationMeta,
) (int64, error) {
	if err := validateMutationMeta(meta); err != nil {
		return 0, err
	}
	if err := validateUUID("namespace id", string(namespaceID)); err != nil {
		return 0, err
	}
	if err := validateUUID("aggregate id", mutation.AggregateID); err != nil {
		return 0, err
	}
	return revisionAsInt64(mutation.AggregateRevision)
}

func allocateDesiredRevision(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
) (int64, int64, error) {
	var runtimeEpoch, desiredRevision int64
	if err := tx.QueryRowContext(ctx, lockNamespaceQuery, namespaceID).Scan(&runtimeEpoch); err != nil {
		if err == sql.ErrNoRows {
			return 0, 0, ErrNotFound
		}
		return 0, 0, fmt.Errorf("lock namespace for outbox: %w", err)
	}
	if runtimeEpoch <= 0 {
		return 0, 0, fmt.Errorf("namespace runtime epoch is invalid")
	}
	if err := tx.QueryRowContext(ctx, nextRevisionQuery, namespaceID).Scan(&desiredRevision); err != nil {
		return 0, 0, fmt.Errorf("allocate desired revision: %w", err)
	}
	if desiredRevision <= 0 {
		return 0, 0, fmt.Errorf("allocated desired revision is invalid")
	}
	return runtimeEpoch, desiredRevision, nil
}
