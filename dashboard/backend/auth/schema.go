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
	legacyRoleSuperAdmin = "super_admin"
	legacyRoleOperator   = "operator"
	legacyRoleUser       = "user"
	legacyRoleReadonly   = "readonly"
)

const (
	PermUsersManage  = "users.manage"
	PermUsersView    = "users.view"
	PermConfigRead   = "config.read"
	PermConfigWrite  = "config.write"
	PermConfigDeploy = "config.deploy"
	PermEvalRead     = "evaluation.read"
	PermEvalWrite    = "evaluation.write"
	PermEvalRun      = "evaluation.run"
	PermTopologyRead = "topology.read"
	PermLogsRead     = "logs.read"
	PermOpenClawRead = "openclaw.read"
	PermOpenClaw     = "openclaw.manage"
	PermMcpRead      = "mcp.read"
	PermMcpManage    = "mcp.manage"
	PermToolsUse     = "tools.use"
	PermMlPipeline   = "mlpipeline.manage"
)

var DefaultRolePermissions = map[string][]string{
	RoleAdmin: {PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline},
	RoleWrite: {PermConfigRead, PermConfigWrite, PermConfigDeploy, PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline},
	RoleRead:  {PermConfigRead, PermEvalRead, PermTopologyRead, PermLogsRead, PermOpenClawRead, PermMcpRead, PermToolsUse},
}

var SupportedRoles = []string{RoleAdmin, RoleWrite, RoleRead}

var legacyRoleAliases = map[string]string{
	legacyRoleSuperAdmin: RoleAdmin,
	legacyRoleOperator:   RoleWrite,
	legacyRoleUser:       RoleRead,
	legacyRoleReadonly:   RoleRead,
}

var AllPermissions = []string{
	PermUsersManage, PermUsersView, PermConfigRead, PermConfigWrite, PermConfigDeploy,
	PermEvalRead, PermEvalWrite, PermEvalRun, PermTopologyRead, PermLogsRead, PermOpenClawRead,
	PermOpenClaw, PermMcpRead, PermMcpManage, PermToolsUse, PermMlPipeline,
}

func normalizeRole(raw string) (string, error) {
	role := strings.ToLower(strings.TrimSpace(raw))
	if role == "" {
		return "", nil
	}
	if aliased, ok := legacyRoleAliases[role]; ok {
		role = aliased
	}

	switch role {
	case RoleAdmin, RoleWrite, RoleRead:
		return role, nil
	default:
		return "", fmt.Errorf("role must be one of %s, %s, %s", RoleAdmin, RoleWrite, RoleRead)
	}
}

func canonicalRole(raw string) string {
	role, err := normalizeRole(raw)
	if err != nil || role == "" {
		return strings.ToLower(strings.TrimSpace(raw))
	}
	return role
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
	u.Role = canonicalRole(u.Role)
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
	u.Role = canonicalRole(u.Role)
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
`
