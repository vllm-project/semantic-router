package postgres

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/invitationmanagement"
)

type auditDocument struct {
	EventID          string   `json:"eventId"`
	NamespaceID      string   `json:"namespaceId"`
	DesiredRevision  string   `json:"desiredRevision"`
	ChainSequence    string   `json:"chainSequence"`
	ActorPrincipalID string   `json:"actorPrincipalId"`
	ActorChain       []string `json:"actorChain"`
	Action           string   `json:"action"`
	ResourceType     string   `json:"resourceType"`
	ResourceID       string   `json:"resourceId"`
	RequestID        string   `json:"requestId"`
	SourceIP         string   `json:"sourceIp"`
	Outcome          string   `json:"outcome"`
	Reason           string   `json:"reason"`
	BeforeRevision   string   `json:"beforeRevision"`
	AfterRevision    string   `json:"afterRevision"`
	PreviousHash     string   `json:"previousHash"`
	CreatedAt        string   `json:"createdAt"`
}

func appendPublication(ctx context.Context, tx *sql.Tx, namespaceID, aggregateType, aggregateID string,
	aggregateRevision uint64, actor invitationmanagement.Actor,
) (int64, error) {
	var runtimeEpoch, desiredRevision int64
	if err := tx.QueryRowContext(ctx, `SELECT runtime_epoch FROM access_namespaces
WHERE id=$1 AND status='active' FOR UPDATE`, namespaceID).Scan(&runtimeEpoch); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return 0, invitationmanagement.ErrNotFound
		}
		return 0, fmt.Errorf("lock onboarding namespace: %w", err)
	}
	if err := tx.QueryRowContext(ctx, `SELECT COALESCE(MAX(revision),0)+1
FROM policy_revisions WHERE namespace_id=$1`, namespaceID).Scan(&desiredRevision); err != nil {
		return 0, fmt.Errorf("allocate onboarding desired revision: %w", err)
	}
	if runtimeEpoch <= 0 || desiredRevision <= 0 {
		return 0, invitationmanagement.ErrUnavailable
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO policy_revisions
  (namespace_id,revision,runtime_epoch,reason,actor_principal_id)
VALUES ($1,$2,$3,$4,$5)`, namespaceID, desiredRevision, runtimeEpoch, actor.Reason, actor.PrincipalID); err != nil {
		return 0, fmt.Errorf("insert onboarding policy revision: %w", err)
	}
	payload, _ := json.Marshal(map[string]any{"aggregateRevision": strconv.FormatUint(aggregateRevision, 10)})
	if _, err := tx.ExecContext(ctx, `INSERT INTO policy_outbox
  (id,namespace_id,desired_revision,aggregate_type,aggregate_id,operation,payload)
VALUES ($1,$2,$3,$4,$5,'created',$6)`, uuid.NewString(), namespaceID, desiredRevision,
		aggregateType, aggregateID, payload); err != nil {
		return 0, fmt.Errorf("insert onboarding policy outbox: %w", err)
	}
	return desiredRevision, nil
}

func appendAudit(ctx context.Context, tx *sql.Tx, namespaceID string, desiredRevision *int64,
	action, resourceType, resourceID string, before *uint64, after uint64, actor invitationmanagement.Actor,
) error {
	if after == 0 || namespaceID == "" || action == "" || resourceType == "" || resourceID == "" ||
		actor.PrincipalID == "" || actor.RequestID == "" || actor.Reason == "" {
		return invitationmanagement.ErrInvalidRequest
	}
	if _, err := tx.ExecContext(ctx, `INSERT INTO access_audit_heads (namespace_id)
VALUES ($1) ON CONFLICT (namespace_id) DO NOTHING`, namespaceID); err != nil {
		return fmt.Errorf("initialize invitation audit chain: %w", err)
	}
	var count int64
	var previousHash []byte
	if err := tx.QueryRowContext(ctx, `SELECT event_count,last_hash FROM access_audit_heads
WHERE namespace_id=$1 FOR UPDATE`, namespaceID).Scan(&count, &previousHash); err != nil {
		return fmt.Errorf("lock invitation audit chain: %w", err)
	}
	if count < 0 || count == math.MaxInt64 || (count == 0 && len(previousHash) != 0) ||
		(count > 0 && len(previousHash) != sha256.Size) {
		return invitationmanagement.ErrUnavailable
	}
	var createdAt time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&createdAt); err != nil {
		return fmt.Errorf("read invitation audit time: %w", err)
	}
	createdAt = createdAt.UTC().Truncate(time.Microsecond)
	document := auditDocument{
		EventID: uuid.NewString(), NamespaceID: namespaceID, ChainSequence: strconv.FormatInt(count+1, 10),
		ActorPrincipalID: actor.PrincipalID, ActorChain: append([]string(nil), actor.ActorChain...),
		Action: action, ResourceType: resourceType, ResourceID: resourceID, RequestID: actor.RequestID,
		Outcome: "allowed", Reason: actor.Reason, AfterRevision: strconv.FormatUint(after, 10),
		PreviousHash: hex.EncodeToString(previousHash), CreatedAt: createdAt.Format(time.RFC3339Nano),
	}
	if desiredRevision != nil {
		document.DesiredRevision = strconv.FormatInt(*desiredRevision, 10)
	}
	if before != nil {
		document.BeforeRevision = strconv.FormatUint(*before, 10)
	}
	if actor.SourceIP.IsValid() {
		document.SourceIP = actor.SourceIP.String()
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		return invitationmanagement.ErrUnavailable
	}
	hash := sha256.Sum256(encoded)
	actorChain, _ := json.Marshal(document.ActorChain)
	details, _ := json.Marshal(map[string]string{"document": string(encoded)})
	var source any
	if actor.SourceIP.IsValid() {
		source = actor.SourceIP.String()
	}
	var beforeValue any
	if before != nil {
		beforeValue = *before
	}
	var previous any
	if len(previousHash) != 0 {
		previous = previousHash
	}
	var desiredValue any
	if desiredRevision != nil {
		desiredValue = *desiredRevision
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO access_audit_events
  (id,namespace_id,desired_revision,chain_sequence,actor_principal_id,actor_chain,
   action,resource_type,resource_id,request_id,source_ip,outcome,reason,
   before_revision,after_revision,details,previous_hash,event_hash,created_at)
VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'allowed',$12,$13,$14,$15,$16,$17,$18)`,
		document.EventID, namespaceID, desiredValue, count+1, actor.PrincipalID, actorChain,
		action, resourceType, resourceID, actor.RequestID, source, actor.Reason,
		beforeValue, after, details, previous, hash[:], createdAt)
	if err != nil {
		return fmt.Errorf("insert invitation audit event: %w", err)
	}
	result, err := tx.ExecContext(ctx, `UPDATE access_audit_heads
SET last_event_id=$2,last_hash=$3,event_count=event_count+1,updated_at=$4
WHERE namespace_id=$1 AND event_count=$5`, namespaceID, document.EventID, hash[:], createdAt, count)
	if err != nil {
		return fmt.Errorf("advance invitation audit chain: %w", err)
	}
	rows, err := result.RowsAffected()
	if err != nil || rows != 1 {
		return invitationmanagement.ErrUnavailable
	}
	return nil
}
