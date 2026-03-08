package auth

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

const (
	defaultSessionCookieName = "vllm_sr_session"
	defaultBootstrapUserID   = "local-admin"
	defaultBootstrapSubject  = "local-admin"
	defaultBootstrapName     = "Local Admin"
	defaultBootstrapRole     = console.ConsoleRoleAdmin
	defaultSessionTTL        = 12 * time.Hour
)

// Error describes a request-visible authentication failure.
type Error struct {
	StatusCode int
	Code       string
	Message    string
}

func (e *Error) Error() string {
	if e == nil {
		return ""
	}
	return e.Message
}

// Service resolves request sessions and roles for the dashboard backend.
type Service struct {
	cfg    Config
	stores *console.Stores
	now    func() time.Time
}

// New constructs an auth service around the console store seams.
func New(cfg Config, stores *console.Stores) (*Service, error) {
	if stores == nil || stores.Users == nil || stores.Sessions == nil || stores.RoleBindings == nil {
		return nil, fmt.Errorf("console identity stores are required")
	}

	if cfg.Mode == "" {
		cfg.Mode = ModeBootstrap
	}
	if cfg.SessionCookieName == "" {
		cfg.SessionCookieName = defaultSessionCookieName
	}
	if cfg.SessionTTL <= 0 {
		cfg.SessionTTL = defaultSessionTTL
	}
	if cfg.BootstrapUserID == "" {
		cfg.BootstrapUserID = defaultBootstrapUserID
	}
	if cfg.BootstrapSubject == "" {
		cfg.BootstrapSubject = defaultBootstrapSubject
	}
	if cfg.BootstrapName == "" {
		cfg.BootstrapName = defaultBootstrapName
	}
	if cfg.BootstrapRole == "" {
		cfg.BootstrapRole = defaultBootstrapRole
	}
	if cfg.BootstrapGrantedBy == "" {
		cfg.BootstrapGrantedBy = "dashboard-bootstrap"
	}
	if cfg.ProxyUserHeader == "" {
		cfg.ProxyUserHeader = "X-Forwarded-User"
	}
	if cfg.ProxyEmailHeader == "" {
		cfg.ProxyEmailHeader = "X-Forwarded-Email"
	}
	if cfg.ProxyNameHeader == "" {
		cfg.ProxyNameHeader = "X-Forwarded-Name"
	}
	if cfg.ProxyRolesHeader == "" {
		cfg.ProxyRolesHeader = "X-Forwarded-Roles"
	}

	switch cfg.Mode {
	case ModeBootstrap, ModeProxy:
	default:
		return nil, fmt.Errorf("unsupported dashboard auth mode %q", cfg.Mode)
	}
	if roleRank(cfg.BootstrapRole) == 0 {
		return nil, fmt.Errorf("unsupported bootstrap role %q", cfg.BootstrapRole)
	}

	return &Service{
		cfg:    cfg,
		stores: stores,
		now:    time.Now,
	}, nil
}

// Mode returns the configured authentication mode.
func (s *Service) Mode() Mode {
	if s == nil {
		return ""
	}
	return s.cfg.Mode
}

// SessionCookieName returns the cookie name used for dashboard sessions.
func (s *Service) SessionCookieName() string {
	if s == nil {
		return defaultSessionCookieName
	}
	return s.cfg.SessionCookieName
}

// ResolveSession loads or creates the request session and refreshes the session cookie.
func (s *Service) ResolveSession(w http.ResponseWriter, r *http.Request) (*RequestSession, error) {
	if s == nil {
		return nil, &Error{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "auth_service_unavailable",
			Message:    "Dashboard auth service is not configured.",
		}
	}

	switch s.cfg.Mode {
	case ModeBootstrap:
		return s.resolveBootstrapSession(w, r)
	case ModeProxy:
		return s.resolveProxySession(w, r)
	default:
		return nil, &Error{
			StatusCode: http.StatusServiceUnavailable,
			Code:       "auth_mode_invalid",
			Message:    "Dashboard auth mode is invalid.",
		}
	}
}

// RevokeRequestSession revokes the request session if one exists and clears the cookie.
func (s *Service) RevokeRequestSession(w http.ResponseWriter, r *http.Request) (string, error) {
	if s == nil {
		return "", nil
	}

	cookie, err := r.Cookie(s.cfg.SessionCookieName)
	if err != nil {
		s.clearSessionCookie(w, r)
		return "", nil
	}

	ctx := context.Background()
	session, err := s.stores.Sessions.GetSession(ctx, cookie.Value)
	if err != nil {
		return "", err
	}
	if session != nil && session.Status == console.SessionStatusActive {
		now := s.now().UTC()
		session.Status = console.SessionStatusRevoked
		session.RevokedAt = &now
		session.UpdatedAt = now
		if saveErr := s.stores.Sessions.SaveSession(ctx, session); saveErr != nil {
			return "", saveErr
		}
	}
	s.clearSessionCookie(w, r)
	return cookie.Value, nil
}

func (s *Service) resolveBootstrapSession(w http.ResponseWriter, r *http.Request) (*RequestSession, error) {
	ctx := context.Background()

	user, err := s.ensureUser(ctx, bootstrapUserKey(s.cfg.BootstrapUserID), "bootstrap", s.cfg.BootstrapSubject, s.cfg.BootstrapEmail, s.cfg.BootstrapName, map[string]interface{}{
		"auth_mode": "bootstrap",
	})
	if err != nil {
		return nil, err
	}
	err = s.ensureBootstrapRoleBinding(ctx, user.ID)
	if err != nil {
		return nil, err
	}

	session, err := s.resolveOrCreateSession(ctx, w, r, user, "bootstrap", s.cfg.BootstrapSubject, []console.ConsoleRole{s.cfg.BootstrapRole}, map[string]interface{}{
		"auth_mode": "bootstrap",
	})
	if err != nil {
		return nil, err
	}
	return buildRequestSession(ModeBootstrap, *user, *session, []console.ConsoleRole{s.cfg.BootstrapRole}), nil
}

func (s *Service) resolveProxySession(w http.ResponseWriter, r *http.Request) (*RequestSession, error) {
	ctx := context.Background()

	userHeader := strings.TrimSpace(r.Header.Get(s.cfg.ProxyUserHeader))
	emailHeader := strings.TrimSpace(r.Header.Get(s.cfg.ProxyEmailHeader))
	nameHeader := strings.TrimSpace(r.Header.Get(s.cfg.ProxyNameHeader))

	if userHeader == "" {
		existingSession, existingUser, existingRoles, err := s.loadExistingRequestSession(ctx, r)
		if err != nil {
			return nil, err
		}
		if existingSession != nil && existingUser != nil {
			return buildRequestSession(ModeProxy, *existingUser, *existingSession, existingRoles), nil
		}
		return nil, &Error{
			StatusCode: http.StatusUnauthorized,
			Code:       "identity_required",
			Message:    fmt.Sprintf("Proxy auth mode requires %s on incoming requests.", s.cfg.ProxyUserHeader),
		}
	}

	user, err := s.ensureUser(ctx, proxyUserKey(userHeader), "proxy", userHeader, emailHeader, coalesceString(nameHeader, userHeader), map[string]interface{}{
		"auth_mode": "proxy",
	})
	if err != nil {
		return nil, err
	}

	roles, err := s.resolveProxyRoles(ctx, user.ID, r.Header.Get(s.cfg.ProxyRolesHeader))
	if err != nil {
		return nil, err
	}
	session, err := s.resolveOrCreateSession(ctx, w, r, user, "proxy", userHeader, roles, map[string]interface{}{
		"auth_mode":   "proxy",
		"header_user": userHeader,
	})
	if err != nil {
		return nil, err
	}
	return buildRequestSession(ModeProxy, *user, *session, roles), nil
}

func (s *Service) ensureUser(
	ctx context.Context,
	userID string,
	authProvider string,
	subject string,
	email string,
	displayName string,
	metadata map[string]interface{},
) (*console.User, error) {
	existing, err := s.stores.Users.GetUser(ctx, userID)
	if err != nil {
		return nil, err
	}
	if existing != nil && existing.Status == console.UserStatusDisabled {
		return nil, &Error{
			StatusCode: http.StatusForbidden,
			Code:       "user_disabled",
			Message:    fmt.Sprintf("Console user %s is disabled.", existing.ID),
		}
	}

	now := s.now().UTC()
	user := &console.User{
		ID:              userID,
		Email:           email,
		DisplayName:     displayName,
		AuthProvider:    authProvider,
		ExternalSubject: subject,
		Status:          console.UserStatusActive,
		LastLoginAt:     &now,
		Metadata:        mergeMetadata(existingMetadata(existing), metadata),
		CreatedAt:       existingCreatedAt(existing),
		UpdatedAt:       now,
	}
	if existing != nil {
		user.Status = existing.Status
		user.CreatedAt = existing.CreatedAt
		if user.Email == "" {
			user.Email = existing.Email
		}
		if user.DisplayName == "" {
			user.DisplayName = existing.DisplayName
		}
	}
	if err := s.stores.Users.SaveUser(ctx, user); err != nil {
		return nil, err
	}
	return user, nil
}

func (s *Service) ensureBootstrapRoleBinding(ctx context.Context, userID string) error {
	binding := &console.RoleBinding{
		ID:            fmt.Sprintf("bootstrap:%s:%s", userID, s.cfg.BootstrapRole),
		PrincipalType: console.PrincipalTypeUser,
		PrincipalID:   userID,
		Role:          s.cfg.BootstrapRole,
		ScopeType:     console.ScopeTypeGlobal,
		GrantedBy:     s.cfg.BootstrapGrantedBy,
		Metadata: map[string]interface{}{
			"auth_mode": "bootstrap",
		},
	}
	return s.stores.RoleBindings.SaveRoleBinding(ctx, binding)
}

func (s *Service) resolveProxyRoles(ctx context.Context, userID string, rawHeader string) ([]console.ConsoleRole, error) {
	roles := parseRoles(rawHeader)
	if len(roles) > 0 {
		return roles, nil
	}

	bound, err := s.boundRoles(ctx, userID)
	if err != nil {
		return nil, err
	}
	if len(bound) > 0 {
		return bound, nil
	}
	return []console.ConsoleRole{console.ConsoleRoleViewer}, nil
}

func (s *Service) boundRoles(ctx context.Context, userID string) ([]console.ConsoleRole, error) {
	bindings, err := s.stores.RoleBindings.ListRoleBindings(ctx, console.RoleBindingFilter{
		PrincipalType: console.PrincipalTypeUser,
		PrincipalID:   userID,
		ScopeType:     console.ScopeTypeGlobal,
		Limit:         16,
	})
	if err != nil {
		return nil, err
	}

	seen := map[console.ConsoleRole]struct{}{}
	roles := make([]console.ConsoleRole, 0, len(bindings))
	for _, binding := range bindings {
		if roleRank(binding.Role) == 0 {
			continue
		}
		if _, ok := seen[binding.Role]; ok {
			continue
		}
		seen[binding.Role] = struct{}{}
		roles = append(roles, binding.Role)
	}
	return roles, nil
}

func (s *Service) loadExistingRequestSession(
	ctx context.Context,
	r *http.Request,
) (*console.Session, *console.User, []console.ConsoleRole, error) {
	session, err := s.validSessionFromRequest(ctx, r)
	if err != nil || session == nil {
		return nil, nil, nil, err
	}
	user, err := s.stores.Users.GetUser(ctx, session.UserID)
	if err != nil || user == nil {
		return nil, nil, nil, err
	}
	if user.Status == console.UserStatusDisabled {
		return nil, nil, nil, &Error{
			StatusCode: http.StatusForbidden,
			Code:       "user_disabled",
			Message:    fmt.Sprintf("Console user %s is disabled.", user.ID),
		}
	}

	roles := parseRolesMetadata(session.Metadata["roles"])
	if len(roles) == 0 {
		roles, err = s.boundRoles(ctx, user.ID)
		if err != nil {
			return nil, nil, nil, err
		}
	}
	if len(roles) == 0 {
		roles = []console.ConsoleRole{console.ConsoleRoleViewer}
	}
	return session, user, roles, nil
}

func (s *Service) resolveOrCreateSession(
	ctx context.Context,
	w http.ResponseWriter,
	r *http.Request,
	user *console.User,
	authProvider string,
	subject string,
	roles []console.ConsoleRole,
	metadata map[string]interface{},
) (*console.Session, error) {
	existing, err := s.validSessionFromRequest(ctx, r)
	if err != nil {
		return nil, err
	}

	now := s.now().UTC()
	sessionMetadata := mergeMetadata(metadata, map[string]interface{}{
		"roles":          roleStrings(roles),
		"effective_role": string(highestRole(roles)),
	})
	if existing != nil && existing.UserID == user.ID && existing.AuthProvider == authProvider {
		existing.ExternalSubject = subject
		existing.Status = console.SessionStatusActive
		existing.ExpiresAt = expiresAtPtr(now.Add(s.cfg.SessionTTL))
		existing.RevokedAt = nil
		existing.Metadata = mergeMetadata(existing.Metadata, sessionMetadata)
		existing.UpdatedAt = now
		if err := s.stores.Sessions.SaveSession(ctx, existing); err != nil {
			return nil, err
		}
		s.writeSessionCookie(w, r, existing.ID)
		return existing, nil
	}

	session := &console.Session{
		UserID:          user.ID,
		AuthProvider:    authProvider,
		ExternalSubject: subject,
		Status:          console.SessionStatusActive,
		ExpiresAt:       expiresAtPtr(now.Add(s.cfg.SessionTTL)),
		Metadata:        sessionMetadata,
	}
	if err := s.stores.Sessions.SaveSession(ctx, session); err != nil {
		return nil, err
	}
	s.writeSessionCookie(w, r, session.ID)
	return session, nil
}

func (s *Service) validSessionFromRequest(ctx context.Context, r *http.Request) (*console.Session, error) {
	cookie, err := r.Cookie(s.cfg.SessionCookieName)
	if err != nil || strings.TrimSpace(cookie.Value) == "" {
		return nil, nil
	}

	session, err := s.stores.Sessions.GetSession(ctx, cookie.Value)
	if err != nil || session == nil {
		return nil, err
	}
	if !sessionIsUsable(*session, s.now().UTC()) {
		s.expireSession(ctx, session)
		return nil, nil
	}
	return session, nil
}

func (s *Service) expireSession(ctx context.Context, session *console.Session) {
	if session == nil {
		return
	}
	now := s.now().UTC()
	session.Status = console.SessionStatusExpired
	session.RevokedAt = &now
	session.UpdatedAt = now
	_ = s.stores.Sessions.SaveSession(ctx, session)
}

func (s *Service) writeSessionCookie(w http.ResponseWriter, r *http.Request, sessionID string) {
	http.SetCookie(w, &http.Cookie{
		Name:     s.cfg.SessionCookieName,
		Value:    sessionID,
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestIsHTTPS(r),
		MaxAge:   int(s.cfg.SessionTTL.Seconds()),
	})
}

func (s *Service) clearSessionCookie(w http.ResponseWriter, r *http.Request) {
	http.SetCookie(w, &http.Cookie{
		Name:     s.cfg.SessionCookieName,
		Value:    "",
		Path:     "/",
		HttpOnly: true,
		SameSite: http.SameSiteLaxMode,
		Secure:   requestIsHTTPS(r),
		MaxAge:   -1,
		Expires:  time.Unix(0, 0),
	})
}

func buildRequestSession(
	mode Mode,
	user console.User,
	session console.Session,
	roles []console.ConsoleRole,
) *RequestSession {
	effectiveRole := highestRole(roles)
	return &RequestSession{
		Authenticated: true,
		AuthMode:      mode,
		User:          user,
		Session:       session,
		Roles:         roles,
		EffectiveRole: effectiveRole,
		Capabilities:  CapabilitiesForRole(effectiveRole),
	}
}
