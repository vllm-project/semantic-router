package auth

import (
	"context"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

// Mode identifies the dashboard authentication flow.
type Mode string

const (
	ModeBootstrap Mode = "bootstrap"
	ModeProxy     Mode = "proxy"
)

// Config controls dashboard authentication and session behavior.
type Config struct {
	Mode              Mode
	SessionCookieName string
	SessionTTL        time.Duration

	BootstrapUserID    string
	BootstrapEmail     string
	BootstrapName      string
	BootstrapRole      console.ConsoleRole
	BootstrapSubject   string
	BootstrapGrantedBy string

	ProxyUserHeader  string
	ProxyEmailHeader string
	ProxyNameHeader  string
	ProxyRolesHeader string
}

// Capabilities summarizes the product actions a session can perform.
type Capabilities struct {
	CanEditConfig    bool `json:"canEditConfig"`
	CanDeployConfig  bool `json:"canDeployConfig"`
	CanActivateSetup bool `json:"canActivateSetup"`
	CanRunEvaluation bool `json:"canRunEvaluation"`
	CanRunMLPipeline bool `json:"canRunMLPipeline"`
	CanAdminister    bool `json:"canAdminister"`
}

// RequestSession is the resolved identity attached to a request.
type RequestSession struct {
	Authenticated bool                  `json:"authenticated"`
	AuthMode      Mode                  `json:"authMode"`
	User          console.User          `json:"user"`
	Session       console.Session       `json:"session"`
	Roles         []console.ConsoleRole `json:"roles"`
	EffectiveRole console.ConsoleRole   `json:"effectiveRole"`
	Capabilities  Capabilities          `json:"capabilities"`
}

// Allows reports whether the request session satisfies the required role.
func (s *RequestSession) Allows(required console.ConsoleRole) bool {
	if s == nil || !s.Authenticated {
		return false
	}
	return roleRank(s.EffectiveRole) >= roleRank(required)
}

// CapabilitiesForRole projects the highest role into frontend-facing permissions.
func CapabilitiesForRole(role console.ConsoleRole) Capabilities {
	return Capabilities{
		CanEditConfig:    roleRank(role) >= roleRank(console.ConsoleRoleEditor),
		CanDeployConfig:  roleRank(role) >= roleRank(console.ConsoleRoleOperator),
		CanActivateSetup: roleRank(role) >= roleRank(console.ConsoleRoleOperator),
		CanRunEvaluation: roleRank(role) >= roleRank(console.ConsoleRoleOperator),
		CanRunMLPipeline: roleRank(role) >= roleRank(console.ConsoleRoleOperator),
		CanAdminister:    roleRank(role) >= roleRank(console.ConsoleRoleAdmin),
	}
}

type requestSessionContextKey struct{}

// WithRequestSession stores the resolved request session in context.
func WithRequestSession(ctx context.Context, session *RequestSession) context.Context {
	return context.WithValue(ctx, requestSessionContextKey{}, session)
}

// SessionFromContext returns the request session stored in context, if any.
func SessionFromContext(ctx context.Context) (*RequestSession, bool) {
	session, ok := ctx.Value(requestSessionContextKey{}).(*RequestSession)
	if !ok || session == nil {
		return nil, false
	}
	return session, true
}

// SessionFromRequest returns the request session stored on the request context, if any.
func SessionFromRequest(r *http.Request) (*RequestSession, bool) {
	if r == nil {
		return nil, false
	}
	return SessionFromContext(r.Context())
}

func highestRole(roles []console.ConsoleRole) console.ConsoleRole {
	highest := console.ConsoleRoleViewer
	for _, role := range roles {
		if roleRank(role) > roleRank(highest) {
			highest = role
		}
	}
	return highest
}

func roleRank(role console.ConsoleRole) int {
	switch role {
	case console.ConsoleRoleAdmin:
		return 4
	case console.ConsoleRoleOperator:
		return 3
	case console.ConsoleRoleEditor:
		return 2
	case console.ConsoleRoleViewer:
		return 1
	default:
		return 0
	}
}
