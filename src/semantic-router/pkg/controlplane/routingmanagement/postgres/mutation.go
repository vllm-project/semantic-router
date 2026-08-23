package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
)

type mutationRecord struct {
	resourceType     string
	resourceID       string
	resourceRevision int64
	action           string
	operation        string
	references       map[string]string
}

func replayResourceReceipt(
	stored managementcommand.StoredResult, resourceType, resourceID string,
) (routingmanagement.RevisionReceipt, error) {
	if stored.Resource == nil || stored.Resource.ResourceType != resourceType || stored.Resource.ResourceID != resourceID {
		return routingmanagement.RevisionReceipt{}, managementcommand.ErrConflict
	}
	return routingmanagement.RevisionReceipt{
		ResourceRevision: int64(stored.Resource.ResourceRevision), Replayed: true,
	}, nil
}

type auditDocument struct {
	EventID          string            `json:"eventId"`
	NamespaceID      string            `json:"namespaceId"`
	DesiredRevision  string            `json:"desiredRevision"`
	ChainSequence    string            `json:"chainSequence"`
	ActorPrincipalID string            `json:"actorPrincipalId"`
	ActorChain       []string          `json:"actorChain"`
	Action           string            `json:"action"`
	ResourceType     string            `json:"resourceType"`
	ResourceID       string            `json:"resourceId"`
	RequestID        string            `json:"requestId"`
	SourceIP         string            `json:"sourceIp"`
	Outcome          string            `json:"outcome"`
	Reason           string            `json:"reason"`
	BeforeRevision   *string           `json:"beforeRevision"`
	AfterRevision    string            `json:"afterRevision"`
	Details          map[string]string `json:"details"`
	PreviousHash     string            `json:"previousHash"`
	CreatedAt        string            `json:"createdAt"`
}

func appendMutation(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	mutation mutationRecord,
	meta routingmanagement.MutationContext,
	publish bool,
) (routingmanagement.RevisionReceipt, error) {
	if tx == nil || mutation.resourceRevision <= 0 || mutation.resourceID == "" ||
		strings.TrimSpace(meta.RequestID) == "" || strings.TrimSpace(meta.Reason) == "" {
		return routingmanagement.RevisionReceipt{}, fmt.Errorf("%w: mutation metadata is incomplete", routingmanagement.ErrInvalid)
	}
	var desiredRevision *int64
	if publish {
		allocated, err := appendPublication(ctx, tx, namespaceID, mutation, meta)
		if err != nil {
			return routingmanagement.RevisionReceipt{}, err
		}
		desiredRevision = &allocated
	}
	if err := appendAudit(ctx, tx, namespaceID, desiredRevision, mutation, meta); err != nil {
		return routingmanagement.RevisionReceipt{}, err
	}
	receipt := routingmanagement.RevisionReceipt{ResourceRevision: mutation.resourceRevision}
	if desiredRevision != nil {
		receipt.DesiredRevision = *desiredRevision
	}
	return receipt, nil
}

func appendPublication(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	mutation mutationRecord,
	meta routingmanagement.MutationContext,
) (int64, error) {
	var runtimeEpoch, desiredRevision int64
	if err := tx.QueryRowContext(ctx, `SELECT runtime_epoch FROM access_namespaces WHERE id = $1 FOR UPDATE`, namespaceID).Scan(&runtimeEpoch); err != nil {
		if err == sql.ErrNoRows {
			return 0, routingmanagement.ErrNotFound
		}
		return 0, fmt.Errorf("lock routing namespace: %w", err)
	}
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(revision), 0) + 1
FROM policy_revisions WHERE namespace_id = $1`, namespaceID).Scan(&desiredRevision); err != nil {
		return 0, fmt.Errorf("allocate routing desired revision: %w", err)
	}
	var actor any
	if meta.PrincipalID != "" {
		actor = meta.PrincipalID
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO policy_revisions
  (namespace_id, revision, runtime_epoch, reason, actor_principal_id)
VALUES ($1,$2,$3,$4,$5)`, namespaceID, desiredRevision, runtimeEpoch, meta.Reason, actor); err != nil {
		return 0, fmt.Errorf("insert routing desired revision: %w", err)
	}
	payload, err := json.Marshal(map[string]any{
		"aggregateRevision": mutation.resourceRevision,
		"references":        mutation.references,
	})
	if err != nil {
		return 0, fmt.Errorf("encode routing outbox payload: %w", err)
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO policy_outbox
  (id, namespace_id, desired_revision, aggregate_type, aggregate_id, operation, payload)
VALUES ($1,$2,$3,$4,$5,$6,$7)`, uuid.NewString(), namespaceID, desiredRevision,
		mutation.resourceType, mutation.resourceID, mutation.operation, payload); err != nil {
		return 0, fmt.Errorf("insert routing outbox: %w", err)
	}
	return desiredRevision, nil
}

func appendAudit(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID string,
	desiredRevision *int64,
	mutation mutationRecord,
	meta routingmanagement.MutationContext,
) error {
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_audit_heads (namespace_id)
VALUES ($1) ON CONFLICT (namespace_id) DO NOTHING`, namespaceID); err != nil {
		return fmt.Errorf("initialize routing audit head: %w", err)
	}
	var count int64
	var previous []byte
	if err := tx.QueryRowContext(ctx, `SELECT event_count, last_hash FROM access_audit_heads
WHERE namespace_id = $1 FOR UPDATE`, namespaceID).Scan(&count, &previous); err != nil {
		return fmt.Errorf("lock routing audit head: %w", err)
	}
	var createdAt time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&createdAt); err != nil {
		return fmt.Errorf("read routing audit time: %w", err)
	}
	createdAt = createdAt.UTC().Truncate(time.Microsecond)
	eventID := uuid.NewString()
	after := fmt.Sprintf("%d", mutation.resourceRevision)
	var before *string
	if mutation.operation != "created" {
		value := fmt.Sprintf("%d", mutation.resourceRevision-1)
		before = &value
	}
	desiredRevisionText := ""
	if desiredRevision != nil {
		desiredRevisionText = fmt.Sprintf("%d", *desiredRevision)
	}
	document := auditDocument{
		EventID: eventID, NamespaceID: namespaceID,
		DesiredRevision: desiredRevisionText, ChainSequence: fmt.Sprintf("%d", count+1),
		ActorPrincipalID: meta.PrincipalID, ActorChain: append([]string(nil), meta.ActorChain...),
		Action: mutation.action, ResourceType: mutation.resourceType, ResourceID: mutation.resourceID,
		RequestID: meta.RequestID, Outcome: "allowed", Reason: meta.Reason,
		BeforeRevision: before, AfterRevision: after, Details: cloneAuditDetails(mutation.references),
		PreviousHash: hex.EncodeToString(previous), CreatedAt: createdAt.Format(time.RFC3339Nano),
	}
	encoded, appendAuditErr := json.Marshal(document)
	if appendAuditErr != nil {
		return fmt.Errorf("encode routing audit document: %w", appendAuditErr)
	}
	hash := sha256.Sum256(encoded)
	actorChain, _ := json.Marshal(document.ActorChain)
	details, _ := json.Marshal(document.Details)
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_audit_events
  (id, namespace_id, desired_revision, chain_sequence, actor_principal_id, actor_chain,
   action, resource_type, resource_id, request_id, outcome, reason, before_revision,
   after_revision, details, previous_hash, event_hash, created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,'allowed',$11,$12,$13,$14,$15,$16,$17)`,
		eventID, namespaceID, nullableInt64(desiredRevision), count+1, actorValue(meta.PrincipalID), actorChain,
		mutation.action, mutation.resourceType, mutation.resourceID, meta.RequestID, meta.Reason,
		nullableRevision(before), mutation.resourceRevision, details, nullableBytes(previous), hash[:], createdAt); err != nil {
		return fmt.Errorf("insert routing audit event: %w", err)
	}
	result, appendAuditErr := tx.ExecContext(ctx, `UPDATE access_audit_heads
SET last_event_id=$2,last_hash=$3,event_count=event_count+1,updated_at=$4
WHERE namespace_id=$1 AND event_count=$5`, namespaceID, eventID, hash[:], createdAt, count)
	if appendAuditErr != nil {
		return fmt.Errorf("advance routing audit head: %w", appendAuditErr)
	}
	rows, _ := result.RowsAffected()
	if rows != 1 {
		return routingmanagement.ErrConflict
	}
	return nil
}

func cloneAuditDetails(source map[string]string) map[string]string {
	if len(source) == 0 {
		return map[string]string{}
	}
	result := make(map[string]string, len(source))
	for key, value := range source {
		result[key] = value
	}
	return result
}

func nullableInt64(value *int64) any {
	if value == nil {
		return nil
	}
	return *value
}

func actorValue(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func nullableRevision(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func nullableBytes(value []byte) any {
	if len(value) == 0 {
		return nil
	}
	return value
}
