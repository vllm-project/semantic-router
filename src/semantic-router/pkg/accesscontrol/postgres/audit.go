package postgres

import (
	"context"
	"crypto/sha256"
	"crypto/subtle"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

const (
	insertAuditHeadQuery = `INSERT INTO access_audit_heads (namespace_id)
VALUES ($1)
ON CONFLICT (namespace_id) DO NOTHING`
	lockAuditHeadQuery = `SELECT event_count, last_hash
FROM access_audit_heads
WHERE namespace_id = $1
FOR UPDATE`
	auditTimestampQuery   = `SELECT clock_timestamp()`
	insertAuditEventQuery = `INSERT INTO access_audit_events
  (id, namespace_id, desired_revision, chain_sequence,
   actor_principal_id, actor_chain, action, resource_type, resource_id,
   request_id, source_ip, outcome, reason, before_revision, after_revision,
   details, previous_hash, event_hash, created_at)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9,
        $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)`
	updateAuditHeadQuery = `UPDATE access_audit_heads
SET last_event_id = $2, last_hash = $3,
    event_count = event_count + 1, updated_at = $4
WHERE namespace_id = $1 AND event_count = $5`
)

type auditHashDocument struct {
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

type auditChainEntry struct {
	Document auditHashDocument
	Hash     []byte
}

func appendAuditEvent(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	mutation outboxMutation,
	meta MutationMeta,
	desiredRevision int64,
) error {
	if _, err := tx.ExecContext(ctx, insertAuditHeadQuery, namespaceID); err != nil {
		return fmt.Errorf("initialize audit chain head: %w", err)
	}
	var eventCount int64
	var previousHash []byte
	if err := tx.QueryRowContext(ctx, lockAuditHeadQuery, namespaceID).Scan(&eventCount, &previousHash); err != nil {
		if err == sql.ErrNoRows {
			return fmt.Errorf("audit chain head is missing: %w", ErrRevisionConflict)
		}
		return fmt.Errorf("lock audit chain head: %w", err)
	}
	if err := validateAuditHead(eventCount, previousHash); err != nil {
		return err
	}
	var createdAt time.Time
	if err := tx.QueryRowContext(ctx, auditTimestampQuery).Scan(&createdAt); err != nil {
		return fmt.Errorf("read audit timestamp: %w", err)
	}
	createdAt = createdAt.UTC().Truncate(time.Microsecond)
	eventID := uuid.NewString()
	sequence := eventCount + 1
	document, beforeRevision, afterRevision, appendAuditEventErr := newAuditHashDocument(
		eventID, namespaceID, desiredRevision, sequence, previousHash, mutation, meta, createdAt,
	)
	if appendAuditEventErr != nil {
		return appendAuditEventErr
	}
	eventHash, appendAuditEventErr := computeAuditEventHash(document)
	if appendAuditEventErr != nil {
		return appendAuditEventErr
	}
	actorChain, details, appendAuditEventErr := encodeAuditJSON(document)
	if appendAuditEventErr != nil {
		return appendAuditEventErr
	}
	if _, err := tx.ExecContext(ctx, insertAuditEventQuery,
		eventID, namespaceID, desiredRevision, sequence,
		actorValue(meta.ActorPrincipalID), actorChain, meta.Action,
		mutation.AggregateType, mutation.AggregateID, meta.RequestID,
		auditSourceIP(meta), "allowed", meta.Reason, beforeRevision, afterRevision,
		details, nullableBytes(previousHash), eventHash[:], createdAt,
	); err != nil {
		return fmt.Errorf("insert access audit event: %w", err)
	}
	result, appendAuditEventErr := tx.ExecContext(ctx, updateAuditHeadQuery,
		namespaceID, eventID, eventHash[:], createdAt, eventCount)
	if appendAuditEventErr != nil {
		return fmt.Errorf("advance audit chain head: %w", appendAuditEventErr)
	}
	if err := requireOneRow(result, ErrRevisionConflict); err != nil {
		return fmt.Errorf("advance audit chain head: %w", err)
	}
	return nil
}

// appendObservedAuditEvent records a security-sensitive read/action without
// inventing a policy revision or outbox item. Its before and after revisions
// are identical because the observed resource is not mutated.
func appendObservedAuditEvent(
	ctx context.Context,
	tx *sql.Tx,
	namespaceID accesscontrol.NamespaceID,
	resourceType, resourceID string,
	revision accesscontrol.Revision,
	meta MutationMeta,
) error {
	if err := validateMutationMeta(meta); err != nil {
		return err
	}
	if err := validateUUID("namespace id", string(namespaceID)); err != nil {
		return err
	}
	if err := validateUUID("resource id", resourceID); err != nil {
		return err
	}
	revisionValue, appendObservedAuditEventErr := revisionAsInt64(revision)
	if appendObservedAuditEventErr != nil {
		return appendObservedAuditEventErr
	}
	if _, err := tx.ExecContext(ctx, insertAuditHeadQuery, namespaceID); err != nil {
		return fmt.Errorf("initialize audit chain head: %w", err)
	}
	var eventCount int64
	var previousHash []byte
	if err := tx.QueryRowContext(ctx, lockAuditHeadQuery, namespaceID).Scan(&eventCount, &previousHash); err != nil {
		return fmt.Errorf("lock audit chain head: %w", err)
	}
	if err := validateAuditHead(eventCount, previousHash); err != nil {
		return err
	}
	var createdAt time.Time
	if err := tx.QueryRowContext(ctx, auditTimestampQuery).Scan(&createdAt); err != nil {
		return fmt.Errorf("read audit timestamp: %w", err)
	}
	createdAt = createdAt.UTC().Truncate(time.Microsecond)
	eventID, sequence := uuid.NewString(), eventCount+1
	revisionText := strconv.FormatInt(revisionValue, 10)
	details := make(map[string]string, len(meta.Details))
	for key, value := range meta.Details {
		details[key] = value
	}
	actors := make([]string, len(meta.ActorChain))
	for index, actor := range meta.ActorChain {
		actors[index] = string(actor)
	}
	document := auditHashDocument{
		EventID: eventID, NamespaceID: string(namespaceID), DesiredRevision: "",
		ChainSequence: strconv.FormatInt(sequence, 10), ActorPrincipalID: auditActor(meta),
		ActorChain: actors, Action: meta.Action, ResourceType: resourceType,
		ResourceID: resourceID, RequestID: meta.RequestID, SourceIP: auditSourceIPString(meta),
		Outcome: "allowed", Reason: meta.Reason, BeforeRevision: &revisionText,
		AfterRevision: revisionText, Details: details, PreviousHash: hex.EncodeToString(previousHash),
		CreatedAt: createdAt.Format(time.RFC3339Nano),
	}
	eventHash, appendObservedAuditEventErr := computeAuditEventHash(document)
	if appendObservedAuditEventErr != nil {
		return appendObservedAuditEventErr
	}
	actorChain, encodedDetails, appendObservedAuditEventErr := encodeAuditJSON(document)
	if appendObservedAuditEventErr != nil {
		return appendObservedAuditEventErr
	}
	if _, err := tx.ExecContext(ctx, insertAuditEventQuery,
		eventID, namespaceID, nil, sequence, actorValue(meta.ActorPrincipalID), actorChain,
		meta.Action, resourceType, resourceID, meta.RequestID, auditSourceIP(meta), "allowed",
		meta.Reason, revisionValue, revisionValue, encodedDetails, nullableBytes(previousHash), eventHash[:], createdAt,
	); err != nil {
		return fmt.Errorf("insert observed access audit event: %w", err)
	}
	result, appendObservedAuditEventErr := tx.ExecContext(ctx, updateAuditHeadQuery,
		namespaceID, eventID, eventHash[:], createdAt, eventCount)
	if appendObservedAuditEventErr != nil {
		return fmt.Errorf("advance audit chain head: %w", appendObservedAuditEventErr)
	}
	return requireOneRow(result, ErrRevisionConflict)
}

func encodeAuditJSON(document auditHashDocument) (string, string, error) {
	actorChain, err := json.Marshal(document.ActorChain)
	if err != nil {
		return "", "", fmt.Errorf("encode audit actor chain: %w", err)
	}
	details, err := json.Marshal(document.Details)
	if err != nil {
		return "", "", fmt.Errorf("encode audit details: %w", err)
	}
	return string(actorChain), string(details), nil
}

func validateAuditHead(eventCount int64, previousHash []byte) error {
	switch {
	case eventCount < 0 || eventCount == math.MaxInt64:
		return fmt.Errorf("audit chain event count is invalid")
	case eventCount == 0 && len(previousHash) != 0:
		return fmt.Errorf("empty audit chain has a previous hash")
	case eventCount > 0 && len(previousHash) != sha256.Size:
		return fmt.Errorf("audit chain previous hash is invalid")
	default:
		return nil
	}
}

func newAuditHashDocument(
	eventID string,
	namespaceID accesscontrol.NamespaceID,
	desiredRevision int64,
	sequence int64,
	previousHash []byte,
	mutation outboxMutation,
	meta MutationMeta,
	createdAt time.Time,
) (auditHashDocument, any, int64, error) {
	afterRevision, err := revisionAsInt64(mutation.AggregateRevision)
	if err != nil {
		return auditHashDocument{}, nil, 0, err
	}
	beforeRevision, beforeDocument, err := auditBeforeRevision(mutation.Operation, afterRevision)
	if err != nil {
		return auditHashDocument{}, nil, 0, err
	}
	details := make(map[string]string, len(meta.Details))
	for key, value := range meta.Details {
		details[key] = value
	}
	actors := make([]string, len(meta.ActorChain))
	for index, actor := range meta.ActorChain {
		actors[index] = string(actor)
	}
	document := auditHashDocument{
		EventID:          eventID,
		NamespaceID:      string(namespaceID),
		DesiredRevision:  strconv.FormatInt(desiredRevision, 10),
		ChainSequence:    strconv.FormatInt(sequence, 10),
		ActorPrincipalID: auditActor(meta),
		ActorChain:       actors,
		Action:           meta.Action,
		ResourceType:     mutation.AggregateType,
		ResourceID:       mutation.AggregateID,
		RequestID:        meta.RequestID,
		SourceIP:         auditSourceIPString(meta),
		Outcome:          "allowed",
		Reason:           meta.Reason,
		BeforeRevision:   beforeDocument,
		AfterRevision:    strconv.FormatInt(afterRevision, 10),
		Details:          details,
		PreviousHash:     hex.EncodeToString(previousHash),
		CreatedAt:        createdAt.Format(time.RFC3339Nano),
	}
	return document, beforeRevision, afterRevision, nil
}

func auditBeforeRevision(operation outboxOperation, after int64) (any, *string, error) {
	if operation == outboxCreated {
		return nil, nil, nil
	}
	if after <= 1 {
		return nil, nil, fmt.Errorf("non-create audit event requires a previous revision")
	}
	before := after - 1
	text := strconv.FormatInt(before, 10)
	return before, &text, nil
}

func computeAuditEventHash(document auditHashDocument) ([sha256.Size]byte, error) {
	encoded, err := json.Marshal(document)
	if err != nil {
		return [sha256.Size]byte{}, fmt.Errorf("encode audit hash document: %w", err)
	}
	return sha256.Sum256(encoded), nil
}

// verifyAuditChain validates a complete namespace chain in sequence order. The
// caller must load entries beginning at sequence one; this keeps the verifier
// explicit about its trust anchor instead of silently accepting a partial tail.
func verifyAuditChain(entries []auditChainEntry) error {
	var previousHash []byte
	var namespaceID string
	for index, entry := range entries {
		expectedSequence := int64(index + 1)
		sequence, err := strconv.ParseInt(entry.Document.ChainSequence, 10, 64)
		if err != nil || sequence != expectedSequence {
			return fmt.Errorf("audit chain sequence %d is invalid", expectedSequence)
		}
		if index == 0 {
			namespaceID = entry.Document.NamespaceID
		} else if entry.Document.NamespaceID != namespaceID {
			return fmt.Errorf("audit chain crosses namespaces at sequence %d", expectedSequence)
		}
		if entry.Document.PreviousHash != hex.EncodeToString(previousHash) {
			return fmt.Errorf("audit chain previous hash mismatch at sequence %d", expectedSequence)
		}
		computed, err := computeAuditEventHash(entry.Document)
		if err != nil {
			return err
		}
		if len(entry.Hash) != sha256.Size || subtle.ConstantTimeCompare(computed[:], entry.Hash) != 1 {
			return fmt.Errorf("audit chain event hash mismatch at sequence %d", expectedSequence)
		}
		previousHash = entry.Hash
	}
	return nil
}

func auditActor(meta MutationMeta) string {
	if meta.ActorPrincipalID == nil {
		return ""
	}
	return string(*meta.ActorPrincipalID)
}

func auditSourceIP(meta MutationMeta) any {
	if !meta.SourceIP.IsValid() {
		return nil
	}
	return meta.SourceIP.String()
}

func auditSourceIPString(meta MutationMeta) string {
	if !meta.SourceIP.IsValid() {
		return ""
	}
	return meta.SourceIP.String()
}
