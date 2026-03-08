// Package console defines the dashboard enterprise-console domain model.
package console

import (
	"encoding/json"
	"time"
)

// StoreBackendType identifies a concrete console store backend.
type StoreBackendType string

const (
	StoreBackendSQLite   StoreBackendType = "sqlite"
	StoreBackendPostgres StoreBackendType = "postgres"
)

// UserStatus tracks whether a console user is allowed to act.
type UserStatus string

const (
	UserStatusActive   UserStatus = "active"
	UserStatusDisabled UserStatus = "disabled"
)

// PrincipalType identifies the subject attached to a role binding.
type PrincipalType string

const (
	PrincipalTypeUser           PrincipalType = "user"
	PrincipalTypeGroup          PrincipalType = "group"
	PrincipalTypeServiceAccount PrincipalType = "service_account"
)

// ConsoleRole is the normalized RBAC role set for the dashboard.
type ConsoleRole string

const (
	ConsoleRoleViewer   ConsoleRole = "viewer"
	ConsoleRoleEditor   ConsoleRole = "editor"
	ConsoleRoleOperator ConsoleRole = "operator"
	ConsoleRoleAdmin    ConsoleRole = "admin"
)

// ScopeType defines the resource scope for a role or secret binding.
type ScopeType string

const (
	ScopeTypeGlobal      ScopeType = "global"
	ScopeTypeEnvironment ScopeType = "environment"
	ScopeTypeWorkspace   ScopeType = "workspace"
	ScopeTypeRevision    ScopeType = "config_revision"
)

// SessionStatus tracks whether a session is still valid.
type SessionStatus string

const (
	SessionStatusActive  SessionStatus = "active"
	SessionStatusRevoked SessionStatus = "revoked"
	SessionStatusExpired SessionStatus = "expired"
)

// ConfigRevisionStatus tracks the lifecycle stage of a console-owned config revision.
type ConfigRevisionStatus string

const (
	ConfigRevisionStatusDraft      ConfigRevisionStatus = "draft"
	ConfigRevisionStatusValidated  ConfigRevisionStatus = "validated"
	ConfigRevisionStatusActive     ConfigRevisionStatus = "active"
	ConfigRevisionStatusSuperseded ConfigRevisionStatus = "superseded"
	ConfigRevisionStatusRolledBack ConfigRevisionStatus = "rolled_back"
	ConfigRevisionStatusDeprecated ConfigRevisionStatus = "deprecated"
)

// DeployEventStatus tracks the lifecycle state of a rollout attempt.
type DeployEventStatus string

const (
	DeployEventStatusPending    DeployEventStatus = "pending"
	DeployEventStatusRunning    DeployEventStatus = "running"
	DeployEventStatusSucceeded  DeployEventStatus = "succeeded"
	DeployEventStatusFailed     DeployEventStatus = "failed"
	DeployEventStatusRolledBack DeployEventStatus = "rolled_back"
)

// AuditOutcome describes whether an audited action succeeded.
type AuditOutcome string

const (
	AuditOutcomeSuccess AuditOutcome = "success"
	AuditOutcomeFailure AuditOutcome = "failure"
)

// User is a first-class console identity.
type User struct {
	ID              string                 `json:"id"`
	Email           string                 `json:"email,omitempty"`
	DisplayName     string                 `json:"display_name,omitempty"`
	AuthProvider    string                 `json:"auth_provider,omitempty"`
	ExternalSubject string                 `json:"external_subject,omitempty"`
	Status          UserStatus             `json:"status"`
	LastLoginAt     *time.Time             `json:"last_login_at,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// RoleBinding grants a role to a principal over a specific scope.
type RoleBinding struct {
	ID            string                 `json:"id"`
	PrincipalType PrincipalType          `json:"principal_type"`
	PrincipalID   string                 `json:"principal_id"`
	Role          ConsoleRole            `json:"role"`
	ScopeType     ScopeType              `json:"scope_type"`
	ScopeID       string                 `json:"scope_id,omitempty"`
	GrantedBy     string                 `json:"granted_by,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
}

// Session is a dashboard-authenticated session boundary.
type Session struct {
	ID              string                 `json:"id"`
	UserID          string                 `json:"user_id"`
	AuthProvider    string                 `json:"auth_provider,omitempty"`
	ExternalSubject string                 `json:"external_subject,omitempty"`
	Status          SessionStatus          `json:"status"`
	ExpiresAt       *time.Time             `json:"expires_at,omitempty"`
	RevokedAt       *time.Time             `json:"revoked_at,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// ConfigRevision is the durable console source-of-truth record for a config change.
type ConfigRevision struct {
	ID                string                 `json:"id"`
	ParentRevisionID  string                 `json:"parent_revision_id,omitempty"`
	Status            ConfigRevisionStatus   `json:"status"`
	Source            string                 `json:"source,omitempty"`
	Summary           string                 `json:"summary,omitempty"`
	DocumentJSON      json.RawMessage        `json:"document_json"`
	RuntimeConfigYAML string                 `json:"runtime_config_yaml,omitempty"`
	CreatedBy         string                 `json:"created_by,omitempty"`
	ActivatedAt       *time.Time             `json:"activated_at,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

// DeployEvent records a single activation, rollback, or rollout attempt.
type DeployEvent struct {
	ID                 string                 `json:"id"`
	RevisionID         string                 `json:"revision_id"`
	Status             DeployEventStatus      `json:"status"`
	TriggerSource      string                 `json:"trigger_source,omitempty"`
	Message            string                 `json:"message,omitempty"`
	RuntimeTarget      string                 `json:"runtime_target,omitempty"`
	RollbackRevisionID string                 `json:"rollback_revision_id,omitempty"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
	StartedAt          *time.Time             `json:"started_at,omitempty"`
	CompletedAt        *time.Time             `json:"completed_at,omitempty"`
	CreatedAt          time.Time              `json:"created_at"`
	UpdatedAt          time.Time              `json:"updated_at"`
}

// SecretRef stores non-secret references used to resolve secret material at runtime.
type SecretRef struct {
	ID            string                 `json:"id"`
	ScopeType     ScopeType              `json:"scope_type"`
	ScopeID       string                 `json:"scope_id,omitempty"`
	Provider      string                 `json:"provider,omitempty"`
	ExternalRef   string                 `json:"external_ref"`
	Version       string                 `json:"version,omitempty"`
	RedactedLabel string                 `json:"redacted_label,omitempty"`
	LastRotatedAt *time.Time             `json:"last_rotated_at,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
}

// AuditEvent captures an append-only operational audit trail.
type AuditEvent struct {
	ID         string                 `json:"id"`
	ActorType  PrincipalType          `json:"actor_type"`
	ActorID    string                 `json:"actor_id,omitempty"`
	Action     string                 `json:"action"`
	TargetType string                 `json:"target_type,omitempty"`
	TargetID   string                 `json:"target_id,omitempty"`
	Outcome    AuditOutcome           `json:"outcome"`
	Message    string                 `json:"message,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
	OccurredAt time.Time              `json:"occurred_at"`
}
