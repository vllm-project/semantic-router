package evaluationplane

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

const (
	lifecyclePolicySchemaVersion   = "evaluation-lifecycle-policy.v2"
	runLifecycleSchemaVersion      = "evaluation-run-lifecycle.v1"
	campaignLifecycleSchemaVersion = "evaluation-campaign-lifecycle.v1"
	lifecycleAuditSchemaVersion    = "evaluation-lifecycle-audit.v2"
	lifecyclePolicyRevision        = "evaluation-lifecycle-policy.2026-09-01"
	lifecycleFileName              = "lifecycle.json"
	lifecyclePolicyFileName        = "policy.json"

	RetentionEphemeral RetentionClass = "ephemeral"
	RetentionStandard  RetentionClass = "standard"
	RetentionProtected RetentionClass = "protected"
)

const (
	defaultOwnerQuotaBytes    = int64(64 * 1024 * 1024 * 1024)
	defaultStoreQuotaBytes    = int64(256 * 1024 * 1024 * 1024)
	defaultAuditQuotaBytes    = int64(64 * 1024 * 1024)
	defaultOwnerRunQuota      = 512
	defaultOwnerCampaignQuota = 512
	reservedRunBytes          = int64(20 * 1024 * 1024)
	maxLifecycleAuditCount    = uint64(1_000_000)
	maxLifecycleRecordSize    = int64(16 * 1024)
)

var (
	ErrForbidden = errors.New("evaluation lifecycle operation is forbidden")
	ErrQuota     = errors.New("evaluation lifecycle quota exceeded")
)

type RetentionClass string

type LifecycleLimits struct {
	MaxOwnerBytes     int64 `json:"max_owner_bytes"`
	MaxStoreBytes     int64 `json:"max_store_bytes"`
	MaxOwnerRuns      int   `json:"max_owner_runs"`
	MaxOwnerCampaigns int   `json:"max_owner_campaigns"`
	MaxAuditBytes     int64 `json:"max_audit_bytes"`
}

func DefaultLifecycleLimits() LifecycleLimits {
	return LifecycleLimits{
		MaxOwnerBytes:     defaultOwnerQuotaBytes,
		MaxStoreBytes:     defaultStoreQuotaBytes,
		MaxOwnerRuns:      defaultOwnerRunQuota,
		MaxOwnerCampaigns: defaultOwnerCampaignQuota,
		MaxAuditBytes:     defaultAuditQuotaBytes,
	}
}

func normalizeLifecycleLimits(limits LifecycleLimits) (LifecycleLimits, error) {
	if limits == (LifecycleLimits{}) {
		limits = DefaultLifecycleLimits()
	}
	if limits.MaxOwnerBytes < reservedRunBytes ||
		limits.MaxStoreBytes < reservedRunBytes+lifecycleCollectionReservedBytes ||
		limits.MaxOwnerBytes > limits.MaxStoreBytes || limits.MaxOwnerRuns < 1 ||
		limits.MaxOwnerRuns > 100_000 || limits.MaxOwnerCampaigns < 1 ||
		limits.MaxOwnerCampaigns > 100_000 || limits.MaxAuditBytes < maxLifecycleRecordSize ||
		limits.MaxAuditBytes > 1024*1024*1024 {
		return LifecycleLimits{}, fmt.Errorf("%w: evaluation lifecycle limits are invalid", ErrInvalid)
	}
	return limits, nil
}

type Actor struct {
	principalDigest string
	administrator   bool
}

func NewActor(principalID string, administrator bool) (Actor, error) {
	principalID = strings.TrimSpace(principalID)
	if principalID == "" || len(principalID) > 512 || strings.ContainsAny(principalID, "\x00\r\n") {
		return Actor{}, fmt.Errorf("%w: authenticated evaluation principal is invalid", ErrInvalid)
	}
	sum := sha256.Sum256([]byte("evaluation-principal.v1\x00" + principalID))
	return Actor{principalDigest: fmt.Sprintf("sha256:%x", sum), administrator: administrator}, nil
}

// PrincipalDigest exposes only the pseudonymous lifecycle identity. Actor's
// fields cannot be filled from a wire payload or external struct literal;
// authority enters through NewActor at the server authentication seam.
func (actor Actor) PrincipalDigest() string { return actor.principalDigest }

func SystemActor() Actor {
	actor, err := NewActor("vllm-sr:evaluation-system", true)
	if err != nil {
		panic(err)
	}
	return actor
}

func validateActor(actor Actor) error {
	if !digestPattern.MatchString(actor.principalDigest) {
		return fmt.Errorf("%w: evaluation actor identity is invalid", ErrInvalid)
	}
	return nil
}

type lifecycleStorePolicy struct {
	SchemaVersion    string          `json:"schema_version"`
	PolicyRevision   string          `json:"policy_revision"`
	Limits           LifecycleLimits `json:"limits"`
	ReservedRunBytes int64           `json:"reserved_run_bytes"`
	PolicyDigest     string          `json:"policy_digest"`
}

func newLifecycleStorePolicy(limits LifecycleLimits) lifecycleStorePolicy {
	policy := lifecycleStorePolicy{
		SchemaVersion: lifecyclePolicySchemaVersion, PolicyRevision: lifecyclePolicyRevision,
		Limits: limits, ReservedRunBytes: reservedRunBytes,
	}
	policy.PolicyDigest = lifecycleDigest(policy)
	return policy
}

type RunLifecycle struct {
	SchemaVersion        string         `json:"schema_version"`
	RunID                string         `json:"run_id"`
	OwnerPrincipalDigest string         `json:"owner_principal_digest"`
	RetentionClass       RetentionClass `json:"retention_class"`
	EvidenceHold         bool           `json:"evidence_hold"`
	DeleteAfter          *time.Time     `json:"delete_after,omitempty"`
	CreatedAt            time.Time      `json:"created_at"`
	UpdatedAt            time.Time      `json:"updated_at"`
	CreationAuditDigest  string         `json:"creation_audit_digest"`
	PolicyRevision       string         `json:"policy_revision"`
	PolicyDigest         string         `json:"policy_digest"`
}

type CampaignLifecycle struct {
	SchemaVersion        string         `json:"schema_version"`
	CampaignID           string         `json:"campaign_id"`
	OwnerPrincipalDigest string         `json:"owner_principal_digest"`
	RetentionClass       RetentionClass `json:"retention_class"`
	EvidenceHold         bool           `json:"evidence_hold"`
	DeleteAfter          *time.Time     `json:"delete_after,omitempty"`
	CreatedAt            time.Time      `json:"created_at"`
	UpdatedAt            time.Time      `json:"updated_at"`
	CreationAuditDigest  string         `json:"creation_audit_digest"`
	PolicyRevision       string         `json:"policy_revision"`
	PolicyDigest         string         `json:"policy_digest"`
}

func newRunLifecycle(run Run, actor Actor) RunLifecycle {
	createdAt := run.CreatedAt.UTC().Truncate(time.Microsecond)
	deleteAfter := createdAt.Add(30 * 24 * time.Hour)
	return RunLifecycle{
		SchemaVersion: runLifecycleSchemaVersion, RunID: run.ID,
		OwnerPrincipalDigest: actor.principalDigest,
		RetentionClass:       RetentionStandard, DeleteAfter: &deleteAfter,
		CreatedAt: createdAt, UpdatedAt: createdAt, PolicyRevision: lifecyclePolicyRevision,
	}
}

func validateRunLifecycle(run Run, lifecycle RunLifecycle) error {
	if lifecycle.SchemaVersion != runLifecycleSchemaVersion || lifecycle.RunID != run.ID ||
		!digestPattern.MatchString(lifecycle.OwnerPrincipalDigest) ||
		!digestPattern.MatchString(lifecycle.CreationAuditDigest) ||
		lifecycle.PolicyRevision != lifecyclePolicyRevision || lifecycle.CreatedAt.IsZero() ||
		lifecycle.UpdatedAt.Before(lifecycle.CreatedAt) || !lifecycle.CreatedAt.Equal(run.CreatedAt) ||
		lifecycle.PolicyDigest != lifecycleDigest(lifecycle) {
		return fmt.Errorf("%w: run lifecycle identity is invalid", ErrInvalid)
	}
	switch lifecycle.RetentionClass {
	case RetentionEphemeral, RetentionStandard:
		if lifecycle.DeleteAfter == nil || lifecycle.DeleteAfter.Before(lifecycle.CreatedAt) {
			return fmt.Errorf("%w: run lifecycle expiry is invalid", ErrInvalid)
		}
	case RetentionProtected:
		if lifecycle.DeleteAfter != nil {
			return fmt.Errorf("%w: protected run lifecycle cannot expire", ErrInvalid)
		}
	default:
		return fmt.Errorf("%w: run retention class is invalid", ErrInvalid)
	}
	return nil
}

func lifecycleDigest(value any) string {
	copyValue := value
	switch typed := value.(type) {
	case RunLifecycle:
		typed.PolicyDigest = ""
		copyValue = typed
	case CampaignLifecycle:
		typed.PolicyDigest = ""
		copyValue = typed
	case lifecycleStorePolicy:
		typed.PolicyDigest = ""
		copyValue = typed
	}
	encoded, err := json.Marshal(copyValue)
	if err != nil {
		panic(err)
	}
	return digestBytes(encoded)
}

type CampaignLifecycleView struct {
	SchemaVersion  string         `json:"schema_version"`
	CampaignID     string         `json:"campaign_id"`
	RetentionClass RetentionClass `json:"retention_class"`
	EvidenceHold   bool           `json:"evidence_hold"`
	DeleteAfter    *time.Time     `json:"delete_after,omitempty"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
}

type RunLifecycleView struct {
	SchemaVersion  string         `json:"schema_version"`
	RunID          string         `json:"run_id"`
	RetentionClass RetentionClass `json:"retention_class"`
	EvidenceHold   bool           `json:"evidence_hold"`
	DeleteAfter    *time.Time     `json:"delete_after,omitempty"`
	CreatedAt      time.Time      `json:"created_at"`
	UpdatedAt      time.Time      `json:"updated_at"`
}

func publicRunLifecycle(lifecycle RunLifecycle) RunLifecycleView {
	return RunLifecycleView{
		SchemaVersion: lifecycle.SchemaVersion, RunID: lifecycle.RunID,
		RetentionClass: lifecycle.RetentionClass, EvidenceHold: lifecycle.EvidenceHold,
		DeleteAfter: lifecycle.DeleteAfter, CreatedAt: lifecycle.CreatedAt, UpdatedAt: lifecycle.UpdatedAt,
	}
}

type UpdateLifecycleRequest struct {
	RetentionClass *RetentionClass `json:"retention_class,omitempty"`
	EvidenceHold   *bool           `json:"evidence_hold,omitempty"`
}
