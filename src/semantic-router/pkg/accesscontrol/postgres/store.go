package postgres

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"regexp"
	"strings"
	"unicode"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type Store struct {
	db *sql.DB
}

var auditActionPattern = regexp.MustCompile(`^[a-z][a-z0-9_.:-]{0,127}$`)

func New(db *sql.DB) (*Store, error) {
	if db == nil {
		return nil, fmt.Errorf("access-control PostgreSQL database is required")
	}
	return &Store{db: db}, nil
}

func inTransaction[T any](ctx context.Context, store *Store, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelReadCommitted})
	if err != nil {
		return zero, fmt.Errorf("begin access-control transaction: %w", err)
	}
	value, err := operation(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit access-control transaction: %w", err)
	}
	return value, nil
}

func inReadTransaction[T any](ctx context.Context, store *Store, operation func(*sql.Tx) (T, error)) (T, error) {
	var zero T
	tx, err := store.db.BeginTx(ctx, &sql.TxOptions{Isolation: sql.LevelRepeatableRead, ReadOnly: true})
	if err != nil {
		return zero, fmt.Errorf("begin access-control read transaction: %w", err)
	}
	value, err := operation(tx)
	if err != nil {
		_ = tx.Rollback()
		return zero, err
	}
	if err := tx.Commit(); err != nil {
		return zero, fmt.Errorf("commit access-control read transaction: %w", err)
	}
	return value, nil
}

func validateMutationMeta(meta MutationMeta) error {
	if err := validateMutationDescription(meta); err != nil {
		return err
	}
	if err := validateMutationActors(meta); err != nil {
		return err
	}
	if err := validateMutationSource(meta); err != nil {
		return err
	}
	return validateAuditDetails(meta.Details)
}

func validateMutationDescription(meta MutationMeta) error {
	if err := validateAuditText("mutation reason", meta.Reason, 512); err != nil {
		return err
	}
	if err := validateAuditText("request id", meta.RequestID, 256); err != nil {
		return err
	}
	if !auditActionPattern.MatchString(meta.Action) {
		return fmt.Errorf("audit action must use lowercase dotted action syntax")
	}
	return nil
}

func validateMutationActors(meta MutationMeta) error {
	if meta.ActorPrincipalID != nil {
		if err := validateUUID("actor principal", string(*meta.ActorPrincipalID)); err != nil {
			return err
		}
	}
	if len(meta.ActorChain) > 32 {
		return fmt.Errorf("audit actor chain exceeds 32 principals")
	}
	seenActors := make(map[accesscontrol.ManagementPrincipalID]struct{}, len(meta.ActorChain))
	for _, actor := range meta.ActorChain {
		if err := validateUUID("actor chain principal", string(actor)); err != nil {
			return err
		}
		if _, duplicate := seenActors[actor]; duplicate {
			return fmt.Errorf("audit actor chain contains duplicate principal %s", actor)
		}
		seenActors[actor] = struct{}{}
	}
	return nil
}

func validateMutationSource(meta MutationMeta) error {
	if meta.SourceIP.IsValid() && meta.SourceIP != meta.SourceIP.Unmap() {
		return fmt.Errorf("audit source IP must use canonical unmapped form")
	}
	return nil
}

func validateAuditDetails(details AuditDetails) error {
	if len(details) > 64 {
		return fmt.Errorf("audit details exceed 64 fields")
	}
	for key, value := range details {
		if err := validateAuditText("audit detail key", key, 64); err != nil {
			return err
		}
		if err := validateAuditText("audit detail value", value, 1024); err != nil {
			return err
		}
		if sensitiveAuditField(key) {
			return fmt.Errorf("audit detail %q may contain secret material", key)
		}
	}
	return nil
}

func validateAuditText(field, value string, maximum int) error {
	if strings.TrimSpace(value) == "" {
		return fmt.Errorf("%s is required", field)
	}
	if strings.TrimSpace(value) != value {
		return fmt.Errorf("%s must not have surrounding whitespace", field)
	}
	if len(value) > maximum {
		return fmt.Errorf("%s exceeds %d bytes", field, maximum)
	}
	for _, char := range value {
		if unicode.IsControl(char) {
			return fmt.Errorf("%s must not contain control characters", field)
		}
	}
	return nil
}

func sensitiveAuditField(field string) bool {
	normalized := strings.NewReplacer("_", "", "-", "", ".", "").Replace(strings.ToLower(field))
	for _, marker := range []string{"secret", "hmac", "ciphertext", "nonce", "pepper", "password"} {
		if strings.Contains(normalized, marker) {
			return true
		}
	}
	switch normalized {
	case "rawkey", "kek", "authorization":
		return true
	default:
		return false
	}
}

func validateUUID(field, value string) error {
	if _, err := uuid.Parse(value); err != nil {
		return fmt.Errorf("%s must be a UUID: %w", field, err)
	}
	return nil
}

func revisionAsInt64(revision accesscontrol.Revision) (int64, error) {
	if revision == 0 || revision > math.MaxInt64 {
		return 0, fmt.Errorf("revision must fit a positive PostgreSQL BIGINT")
	}
	return int64(revision), nil
}

func scanRevision(revision int64) (accesscontrol.Revision, error) {
	if revision <= 0 {
		return 0, fmt.Errorf("database returned invalid revision %d", revision)
	}
	return accesscontrol.Revision(revision), nil
}

func actorValue(actor *accesscontrol.ManagementPrincipalID) any {
	if actor == nil {
		return nil
	}
	return string(*actor)
}
