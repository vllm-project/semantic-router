package auth

import (
	"database/sql"
	"fmt"
	"strings"
)

const (
	RoleAdmin = "admin"
	RoleWrite = "write"
	RoleRead  = "read"
)

const (
	PermUsersManage    = "users.manage"
	PermUsersView      = "users.view"
	PermConfigRead     = "config.read"
	PermConfigWrite    = "config.write"
	PermConfigDeploy   = "config.deploy"
	PermEvalRead       = "evaluation.read"
	PermEvalWrite      = "evaluation.write"
	PermEvalRun        = "evaluation.run"
	PermTopologyRead   = "topology.read"
	PermLogsRead       = "logs.read"
	PermOpenClawRead   = "openclaw.read"
	PermOpenClaw       = "openclaw.manage"
	PermMlPipeline     = "mlpipeline.manage"
	PermFeedbackSubmit = "feedback.submit"
	PermReplayRead     = "replay.read"
	PermStatusRead     = "status.read"
)

var DefaultRolePermissions = map[string][]string{
	RoleAdmin: {PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMlPipeline, PermFeedbackSubmit, PermReplayRead, PermStatusRead},
	RoleWrite: {PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMlPipeline, PermFeedbackSubmit, PermReplayRead, PermStatusRead},
	RoleRead:  {PermConfigRead, PermTopologyRead, PermReplayRead},
}

var SupportedRoles = []string{RoleAdmin, RoleWrite, RoleRead}

var AllPermissions = []string{
	PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy,
	PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead,
	PermOpenClaw, PermMlPipeline,
	PermFeedbackSubmit, PermReplayRead, PermStatusRead,
}

func normalizeRole(raw string) (string, error) {
	role := strings.ToLower(strings.TrimSpace(raw))
	if role == "" {
		return "", nil
	}
	switch role {
	case RoleAdmin, RoleWrite, RoleRead:
		return role, nil
	default:
		return "", fmt.Errorf("role must be one of %s, %s, %s", RoleAdmin, RoleWrite, RoleRead)
	}
}

type User struct {
	ID          string   `json:"id"`
	Email       string   `json:"email"`
	Name        string   `json:"name"`
	Role        string   `json:"role"`
	Status      string   `json:"status"`
	CreatedAt   int64    `json:"createdAt"`
	UpdatedAt   int64    `json:"updatedAt"`
	LastLoginAt *int64   `json:"lastLoginAt,omitempty"`
	Permissions []string `json:"permissions,omitempty"`
}

func scanUser(row *sql.Row) (*User, error) {
	u := &User{}
	var lastLogin sql.NullInt64
	if err := row.Scan(&u.ID, &u.Email, &u.Name, &u.Role, &u.Status, &u.CreatedAt, &u.UpdatedAt, &lastLogin); err != nil {
		return nil, err
	}
	if lastLogin.Valid {
		t := lastLogin.Int64
		u.LastLoginAt = &t
	}
	return u, nil
}

func scanUserRows(rows *sql.Rows) (*User, error) {
	u := &User{}
	var lastLogin sql.NullInt64
	if err := rows.Scan(&u.ID, &u.Email, &u.Name, &u.Role, &u.Status, &u.CreatedAt, &u.UpdatedAt, &lastLogin); err != nil {
		return nil, err
	}
	if lastLogin.Valid {
		t := lastLogin.Int64
		u.LastLoginAt = &t
	}
	return u, nil
}

const createUsersSchema = `
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL DEFAULT 'read',
  status TEXT NOT NULL DEFAULT 'active',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  last_login_at INTEGER
);

CREATE TABLE IF NOT EXISTS dashboard_member_invitations (
  router_invitation_id TEXT PRIMARY KEY,
  router_namespace_id TEXT NOT NULL,
  router_revision INTEGER NOT NULL CHECK (router_revision > 0),
  email TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT '',
  token_digest TEXT NOT NULL UNIQUE,
  planned_subject_id TEXT NOT NULL UNIQUE,
  presentation_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (presentation_status IN ('pending', 'accepted', 'revoked')),
  expires_at INTEGER NOT NULL,
  accepted_at INTEGER,
  revoked_at INTEGER,
  created_at INTEGER NOT NULL,
  created_by TEXT NOT NULL,
  updated_at INTEGER NOT NULL,
  last_sent_at INTEGER,
  delivery_status TEXT NOT NULL DEFAULT 'not_requested',
  delivery_error TEXT NOT NULL DEFAULT '',
  FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS role_permissions (
  role TEXT NOT NULL,
  permission_key TEXT NOT NULL,
  allowed INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (role, permission_key)
);

CREATE TABLE IF NOT EXISTS user_permissions (
  user_id TEXT NOT NULL,
  permission_key TEXT NOT NULL,
  allowed INTEGER NOT NULL DEFAULT 1,
  PRIMARY KEY (user_id, permission_key),
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS user_audit_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  action TEXT NOT NULL,
  resource TEXT NOT NULL,
  method TEXT,
  path TEXT,
  ip TEXT,
  user_agent TEXT,
  status_code INTEGER,
  created_at INTEGER NOT NULL,
  extra_json TEXT,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  issued_at INTEGER NOT NULL,
  expires_at INTEGER NOT NULL,
  revoked_at INTEGER,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS dashboard_bootstrap_installation (
  singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
  user_id TEXT NOT NULL UNIQUE,
  session_id TEXT NOT NULL UNIQUE,
  source_issued_at INTEGER NOT NULL,
  source_expires_at INTEGER NOT NULL,
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL,
  FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON auth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_revoked_at ON auth_sessions(revoked_at);
CREATE INDEX IF NOT EXISTS idx_users_status_created_at ON users(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_dashboard_member_invitations_created_at ON dashboard_member_invitations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_member_invitations_email ON dashboard_member_invitations(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dashboard_member_pending_invite_email
  ON dashboard_member_invitations(email) WHERE presentation_status = 'pending';
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_created_at ON user_audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_user_id ON user_audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_user_created_at ON user_audit_logs(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_action_created_at ON user_audit_logs(action, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_resource_created_at ON user_audit_logs(resource, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_user_audit_logs_status_created_at ON user_audit_logs(status_code, created_at DESC);
`

const (
	InvitationPending  = "pending"
	InvitationAccepted = "accepted"
	InvitationRevoked  = "revoked"
	InvitationExpired  = "expired"
)

type DashboardMemberInvitation struct {
	ID              string `json:"id"`
	NamespaceID     string `json:"namespaceId"`
	Revision        uint64 `json:"revision"`
	Email           string `json:"email"`
	Name            string `json:"name"`
	Role            string `json:"role"`
	TeamID          string `json:"teamId,omitempty"`
	TeamRole        string `json:"teamRole,omitempty"`
	Status          string `json:"status"`
	ExpiresAt       int64  `json:"expiresAt"`
	AcceptedAt      *int64 `json:"acceptedAt,omitempty"`
	RevokedAt       *int64 `json:"revokedAt,omitempty"`
	CreatedAt       int64  `json:"createdAt"`
	CreatedBy       string `json:"createdBy,omitempty"`
	UpdatedAt       int64  `json:"updatedAt"`
	LastSentAt      *int64 `json:"lastSentAt,omitempty"`
	DeliveryStatus  string `json:"deliveryStatus"`
	DeliveryError   string `json:"deliveryError,omitempty"`
	InvitationToken string `json:"invitationToken,omitempty"`
	InvitationPath  string `json:"invitationPath,omitempty"`
}

// invitationPresentation is deliberately not an identity or authorization
// record. It links Router-owned invitation state to Dashboard delivery UX.
// Role grants, Team membership, policy bindings, and plaintext secrets never
// enter this table or type.
type invitationPresentation struct {
	RouterInvitationID string
	RouterNamespaceID  string
	RouterRevision     uint64
	Email              string
	Name               string
	TokenDigest        string
	PlannedSubjectID   string
	Status             string
	ExpiresAt          int64
	AcceptedAt         *int64
	RevokedAt          *int64
	CreatedAt          int64
	CreatedBy          string
	UpdatedAt          int64
	LastSentAt         *int64
	DeliveryStatus     string
	DeliveryError      string
}
