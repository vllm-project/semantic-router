package auth

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
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
	PermMcpRead        = "mcp.read"
	PermMcpManage      = "mcp.manage"
	PermToolsUse       = "tools.use"
	PermMlPipeline     = "mlpipeline.manage"
	PermFeedbackSubmit = "feedback.submit"
	PermReplayRead     = "replay.read"
	PermAccessRead     = "access.read"
	PermAccessManage   = "access.manage"
	PermAccessSelf     = "access.self"
	PermUsageRead      = "usage.read"
	PermUsageSelf      = "usage.self"
	PermAuditRead      = "audit.read"
	PermStatusRead     = "status.read"
)

var DefaultRolePermissions = map[string][]string{
	RoleAdmin: {PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline, PermFeedbackSubmit, PermReplayRead, PermAccessRead, PermAccessManage, PermAccessSelf, PermUsageRead, PermUsageSelf, PermAuditRead, PermStatusRead},
	RoleWrite: {PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline, PermFeedbackSubmit, PermReplayRead, PermAccessRead, PermAccessSelf, PermUsageRead, PermUsageSelf, PermAuditRead, PermStatusRead},
	// Read is the model-consumer surface: routing is inspectable while model
	// credentials, request logs, and usage remain scoped to the signed-in user.
	RoleRead: {PermConfigRead, PermEvalRead, PermTopologyRead, PermToolsUse, PermReplayRead, PermAccessSelf, PermUsageSelf},
}

var SupportedRoles = []string{RoleAdmin, RoleWrite, RoleRead}

var AllPermissions = []string{
	PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy,
	PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead,
	PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline,
	PermFeedbackSubmit, PermReplayRead,
	PermAccessRead, PermAccessManage, PermAccessSelf, PermUsageRead, PermUsageSelf, PermAuditRead,
	PermStatusRead,
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
  id TEXT PRIMARY KEY,
  email TEXT NOT NULL,
  name TEXT NOT NULL DEFAULT '',
  role TEXT NOT NULL,
  team_id TEXT NOT NULL DEFAULT '',
  team_role TEXT NOT NULL DEFAULT 'member',
  token_digest TEXT NOT NULL UNIQUE,
  status TEXT NOT NULL DEFAULT 'pending',
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

CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_expires_at ON auth_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_revoked_at ON auth_sessions(revoked_at);
CREATE INDEX IF NOT EXISTS idx_users_status_created_at ON users(status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_dashboard_member_invitations_created_at ON dashboard_member_invitations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_dashboard_member_invitations_email ON dashboard_member_invitations(email);
CREATE UNIQUE INDEX IF NOT EXISTS idx_dashboard_member_pending_invite_email
  ON dashboard_member_invitations(email) WHERE status = 'pending';
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
	CreatedBy       string `json:"createdBy"`
	UpdatedAt       int64  `json:"updatedAt"`
	LastSentAt      *int64 `json:"lastSentAt,omitempty"`
	DeliveryStatus  string `json:"deliveryStatus"`
	DeliveryError   string `json:"deliveryError,omitempty"`
	InvitationToken string `json:"invitationToken,omitempty"`
	InvitationPath  string `json:"invitationPath,omitempty"`
}

func (invitation DashboardMemberInvitation) effectiveStatus(now time.Time) string {
	if invitation.Status == InvitationPending && invitation.ExpiresAt <= now.Unix() {
		return InvitationExpired
	}
	return invitation.Status
}
