package console

import (
	"context"
	"fmt"
)

// DefaultListLimit keeps store list calls bounded unless a caller overrides it.
const DefaultListLimit = 50

// StoreConfig selects a concrete console store backend.
type StoreConfig struct {
	Backend    StoreBackendType
	SQLitePath string
	DSN        string
}

// RoleBindingFilter constrains a role-binding list query.
type RoleBindingFilter struct {
	PrincipalType PrincipalType
	PrincipalID   string
	Role          ConsoleRole
	ScopeType     ScopeType
	ScopeID       string
	Limit         int
}

// SessionFilter constrains a session list query.
type SessionFilter struct {
	UserID string
	Status SessionStatus
	Limit  int
}

// ConfigRevisionFilter constrains a config revision list query.
type ConfigRevisionFilter struct {
	Status ConfigRevisionStatus
	Source string
	Limit  int
}

// DeployEventFilter constrains a deploy event list query.
type DeployEventFilter struct {
	RevisionID string
	Status     DeployEventStatus
	Limit      int
}

// SecretRefFilter constrains a secret-ref list query.
type SecretRefFilter struct {
	ScopeType ScopeType
	ScopeID   string
	Limit     int
}

// AuditEventFilter constrains an audit-event list query.
type AuditEventFilter struct {
	ActorID    string
	Action     string
	TargetType string
	TargetID   string
	Outcome    AuditOutcome
	Limit      int
}

// StoreLifecycle captures store-open or close behavior shared by every backend.
type StoreLifecycle interface {
	Close() error
}

// UserStore captures user persistence behavior.
type UserStore interface {
	SaveUser(ctx context.Context, user *User) error
	GetUser(ctx context.Context, id string) (*User, error)
}

// RoleBindingStore captures role-binding persistence behavior.
type RoleBindingStore interface {
	SaveRoleBinding(ctx context.Context, binding *RoleBinding) error
	ListRoleBindings(ctx context.Context, filter RoleBindingFilter) ([]RoleBinding, error)
}

// SessionStore captures session persistence behavior.
type SessionStore interface {
	SaveSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, id string) (*Session, error)
	ListSessions(ctx context.Context, filter SessionFilter) ([]Session, error)
}

// ConfigRevisionStore captures config-revision persistence behavior.
type ConfigRevisionStore interface {
	SaveConfigRevision(ctx context.Context, revision *ConfigRevision) error
	GetConfigRevision(ctx context.Context, id string) (*ConfigRevision, error)
	ListConfigRevisions(ctx context.Context, filter ConfigRevisionFilter) ([]ConfigRevision, error)
}

// DeployEventStore captures rollout-event persistence behavior.
type DeployEventStore interface {
	SaveDeployEvent(ctx context.Context, event *DeployEvent) error
	ListDeployEvents(ctx context.Context, filter DeployEventFilter) ([]DeployEvent, error)
}

// SecretRefStore captures secret-reference persistence behavior.
type SecretRefStore interface {
	SaveSecretRef(ctx context.Context, ref *SecretRef) error
	ListSecretRefs(ctx context.Context, filter SecretRefFilter) ([]SecretRef, error)
}

// AuditEventStore captures audit persistence behavior.
type AuditEventStore interface {
	AppendAuditEvent(ctx context.Context, event *AuditEvent) error
	ListAuditEvents(ctx context.Context, filter AuditEventFilter) ([]AuditEvent, error)
}

// Stores groups the smaller domain-specific store contracts for callers.
type Stores struct {
	Lifecycle    StoreLifecycle
	Users        UserStore
	RoleBindings RoleBindingStore
	Sessions     SessionStore
	Revisions    ConfigRevisionStore
	Deployments  DeployEventStore
	Secrets      SecretRefStore
	Audit        AuditEventStore
}

// OpenStore opens the configured console store backend.
func OpenStore(cfg StoreConfig) (*Stores, error) {
	backend := cfg.Backend
	if backend == "" {
		backend = StoreBackendSQLite
	}

	switch backend {
	case StoreBackendSQLite:
		if cfg.SQLitePath == "" {
			return nil, fmt.Errorf("sqlite_path is required for console store backend %q", backend)
		}
		store, err := NewSQLiteStore(cfg.SQLitePath)
		if err != nil {
			return nil, err
		}
		return NewStores(store), nil
	case StoreBackendPostgres:
		return nil, fmt.Errorf("console store backend %q is not implemented yet", backend)
	default:
		return nil, fmt.Errorf("unsupported console store backend %q", backend)
	}
}

// NewStores adapts a concrete store implementation into grouped domain contracts.
func NewStores(store *SQLiteStore) *Stores {
	return &Stores{
		Lifecycle:    store,
		Users:        store,
		RoleBindings: store,
		Sessions:     store,
		Revisions:    store,
		Deployments:  store,
		Secrets:      store,
		Audit:        store,
	}
}

func normalizedLimit(limit int) int {
	if limit <= 0 {
		return DefaultListLimit
	}
	return limit
}
