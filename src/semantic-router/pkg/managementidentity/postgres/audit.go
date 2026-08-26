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
	"regexp"
	"strconv"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementidentity"
)

type auditMutation struct {
	NamespaceID    string
	Action         string
	ResourceType   string
	ResourceID     string
	BeforeRevision *uint64
	AfterRevision  uint64
	Actor          managementidentity.MutationActor
	ExternalActor  bool
}

type auditDocument struct {
	EventID          string   `json:"eventId"`
	NamespaceID      string   `json:"namespaceId"`
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

func appendAudit(ctx context.Context, tx *sql.Tx, mutation auditMutation) error {
	if err := validateAuditMutation(mutation); err != nil {
		return err
	}
	var sequence int64
	var previousHash []byte
	if mutation.NamespaceID != "" {
		if !canonicalUUID(mutation.NamespaceID) {
			return errors.New("management identity audit namespace is invalid")
		}
		if _, err := tx.ExecContext(ctx, `INSERT INTO access_audit_heads (namespace_id)
VALUES ($1) ON CONFLICT (namespace_id) DO NOTHING`, mutation.NamespaceID); err != nil {
			return fmt.Errorf("initialize Management identity audit chain: %w", err)
		}
		var count int64
		if err := tx.QueryRowContext(ctx, `SELECT event_count, last_hash
FROM access_audit_heads WHERE namespace_id = $1 FOR UPDATE`, mutation.NamespaceID).Scan(&count, &previousHash); err != nil {
			return fmt.Errorf("lock Management identity audit chain: %w", err)
		}
		if count < 0 || count == math.MaxInt64 || (count == 0 && len(previousHash) != 0) || (count > 0 && len(previousHash) != sha256.Size) {
			return errors.New("management identity audit chain is invalid")
		}
		sequence = count + 1
	}
	var createdAt time.Time
	if err := tx.QueryRowContext(ctx, `SELECT clock_timestamp()`).Scan(&createdAt); err != nil {
		return fmt.Errorf("read Management identity audit timestamp: %w", err)
	}
	createdAt = createdAt.UTC().Truncate(time.Microsecond)
	document := auditDocument{
		EventID: uuid.NewString(), NamespaceID: mutation.NamespaceID,
		ActorPrincipalID: mutation.Actor.PrincipalID, ActorChain: append([]string{}, mutation.Actor.ActorChain...),
		Action: mutation.Action, ResourceType: mutation.ResourceType, ResourceID: mutation.ResourceID,
		RequestID: mutation.Actor.RequestID, Outcome: "allowed", Reason: mutation.Actor.Reason,
		AfterRevision: strconv.FormatUint(mutation.AfterRevision, 10), PreviousHash: hex.EncodeToString(previousHash),
		CreatedAt: createdAt.Format(time.RFC3339Nano),
	}
	if sequence > 0 {
		document.ChainSequence = strconv.FormatInt(sequence, 10)
	}
	if mutation.BeforeRevision != nil {
		document.BeforeRevision = strconv.FormatUint(*mutation.BeforeRevision, 10)
	}
	if mutation.Actor.SourceIP.IsValid() {
		document.SourceIP = mutation.Actor.SourceIP.String()
	}
	encoded, err := json.Marshal(document)
	if err != nil {
		return fmt.Errorf("encode Management identity audit document: %w", err)
	}
	hash := sha256.Sum256(encoded)
	actorChain, _ := json.Marshal(document.ActorChain)
	details, _ := json.Marshal(map[string]string{"document": string(encoded)})
	var namespace, chainSequence, previous any
	if mutation.NamespaceID != "" {
		namespace, chainSequence = mutation.NamespaceID, sequence
		if len(previousHash) != 0 {
			previous = previousHash
		}
	}
	var before any
	if mutation.BeforeRevision != nil {
		before = *mutation.BeforeRevision
	}
	var source any
	if mutation.Actor.SourceIP.IsValid() {
		source = mutation.Actor.SourceIP.String()
	}
	var actorPrincipal any = mutation.Actor.PrincipalID
	if mutation.ExternalActor {
		actorPrincipal = nil
	}
	_, err = tx.ExecContext(ctx, `INSERT INTO access_audit_events
  (id, namespace_id, desired_revision, chain_sequence, actor_principal_id,
   actor_chain, action, resource_type, resource_id, request_id, source_ip,
   outcome, reason, before_revision, after_revision, details, previous_hash,
   event_hash, created_at)
VALUES ($1,$2,NULL,$3,$4,$5,$6,$7,$8,$9,$10,'allowed',$11,$12,$13,$14,$15,$16,$17)`,
		document.EventID, namespace, chainSequence, actorPrincipal,
		actorChain, mutation.Action, mutation.ResourceType, mutation.ResourceID,
		mutation.Actor.RequestID, source, mutation.Actor.Reason, before,
		mutation.AfterRevision, details, previous, hash[:], createdAt,
	)
	if err != nil {
		return fmt.Errorf("insert Management identity audit event: %w", err)
	}
	if mutation.NamespaceID != "" {
		result, err := tx.ExecContext(ctx, `UPDATE access_audit_heads
SET last_event_id=$2,last_hash=$3,event_count=event_count+1,updated_at=$4
WHERE namespace_id=$1 AND event_count=$5`, mutation.NamespaceID, document.EventID, hash[:], createdAt, sequence-1)
		if err != nil {
			return fmt.Errorf("advance Management identity audit head: %w", err)
		}
		count, err := result.RowsAffected()
		if err != nil || count != 1 {
			return errors.New("management identity audit head changed concurrently")
		}
	}
	return nil
}

func validateAuditMutation(mutation auditMutation) error {
	validActor := canonicalUUID(mutation.Actor.PrincipalID)
	if mutation.ExternalActor {
		validActor = mutation.Actor.PrincipalID == "" && len(mutation.Actor.ActorChain) == 0 &&
			!mutation.Actor.SourceIP.IsValid()
	}
	if mutation.AfterRevision == 0 || mutation.Action == "" || mutation.ResourceType == "" ||
		!canonicalAuditResourceID(mutation.ResourceID) || !validActor ||
		mutation.Actor.RequestID == "" || mutation.Actor.Reason == "" {
		return errors.New("management identity audit mutation is invalid")
	}
	return nil
}

var auditResourceIDPattern = regexp.MustCompile(`^(?:[a-z][a-z0-9_-]{2,127}|[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12})$`)

func canonicalAuditResourceID(value string) bool { return auditResourceIDPattern.MatchString(value) }
