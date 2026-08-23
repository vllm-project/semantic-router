package auth

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

type Service struct {
	store       *Store
	jwtSecret   []byte
	ttlDuration time.Duration
	now         func() time.Time

	invitationMailer      InvitationMailer
	invitationBaseURL     string
	invitationIssuerURL   string
	invitationAuthority   InvitationAuthority
	firstAdminProvisioner FirstAdminProvisioner

	// allowOpenBootstrap gates the public web-form bootstrap endpoint (off by default).
	allowOpenBootstrap bool
	// bootstrapMu serializes the check-then-create in BootstrapRegister so that two
	// concurrent requests cannot both pass the "no users yet" check and each create an
	// admin. The dashboard runs single-replica (enforced by the chart's replicaCount
	// guard), so a process-level mutex is sufficient; a multi-writer deployment would
	// need a transactional guard in the store instead.
	bootstrapMu sync.Mutex
}

// ErrBootstrapClosed is returned by BootstrapRegister when an admin already exists.
var ErrBootstrapClosed = errors.New("bootstrap is disabled")

type TokenClaims struct {
	UserID string `json:"userId"`
	Email  string `json:"email"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

func NewService(store *Store, secret string, ttlHours int) *Service {
	if ttlHours <= 0 {
		ttlHours = 12
	}
	if strings.TrimSpace(secret) == "" {
		b := make([]byte, 32)
		_, _ = rand.Read(b)
		secret = base64.RawStdEncoding.EncodeToString(b)
	}
	return &Service{
		store: store, jwtSecret: []byte(secret), ttlDuration: time.Duration(ttlHours) * time.Hour,
		now: time.Now,
	}
}

// AddAuditLog records one authenticated control-plane action without exposing
// the underlying auth store to other dashboard packages.
func (s *Service) AddAuditLog(ctx context.Context, entry AuditLog) error {
	if s == nil || s.store == nil {
		return nil
	}
	return s.store.AddAuditLog(ctx, entry)
}

// SetAllowOpenBootstrap toggles the public web-form bootstrap endpoint.
func (s *Service) SetAllowOpenBootstrap(v bool) { s.allowOpenBootstrap = v }

// OpenBootstrapEnabled reports whether the public web-form bootstrap endpoint is enabled.
func (s *Service) OpenBootstrapEnabled() bool { return s.allowOpenBootstrap }

// BootstrapRegister atomically creates the first admin. The "no users yet" check and
// the create run under bootstrapMu, so two concurrent requests cannot each create an
// admin (closing the time-of-check-to-time-of-use race in the public bootstrap path).
// Returns ErrBootstrapClosed if any user already exists.
func (s *Service) BootstrapRegister(
	ctx context.Context,
	email string,
	name string,
	password string,
	hash string,
) (*User, string, error) {
	s.bootstrapMu.Lock()
	defer s.bootstrapMu.Unlock()
	if s.firstAdminProvisioner == nil {
		ok, err := s.CanBootstrap(ctx)
		if err != nil {
			return nil, "", err
		}
		if !ok {
			return nil, "", ErrBootstrapClosed
		}
		user, err := s.store.CreateUser(ctx, email, defaultAdminName(name), hash, RoleAdmin, "active")
		if err != nil {
			return nil, "", err
		}
		token, err := s.issueTokenForContext(ctx, user)
		return user, token, err
	}

	now := s.now().UTC().Truncate(time.Second)
	pending, storedHash, created, err := s.store.PrepareBootstrapAdmin(
		ctx, strings.TrimSpace(email), defaultAdminName(name), hash, now, now.Add(s.ttlDuration),
	)
	if err != nil {
		return nil, "", err
	}
	if !created && !s.VerifyPassword(storedHash, password) {
		return nil, "", ErrBootstrapClosed
	}
	if provisionErr := s.firstAdminProvisioner.ProvisionFirstAdmin(ctx, FirstAdminIdentity{
		UserID: pending.User.ID, SessionID: pending.SessionID,
		Email: pending.User.Email, DisplayName: pending.User.Name,
		AuthenticatedAt: pending.SessionIssuedAt,
		ExpiresAt:       pending.SessionExpiresAt,
	}); provisionErr != nil {
		return nil, "", fmt.Errorf("provision Router administrator: %w", ErrFirstAdminProvisioningUnavailable)
	}
	token, session, err := s.prepareTokenForSession(
		pending.User, pending.SessionID, pending.SessionIssuedAt, pending.SessionExpiresAt,
	)
	if err != nil {
		return nil, "", err
	}
	user, err := s.store.CompleteBootstrapAdmin(
		ctx, pending.User.ID, session, s.now().UTC().Truncate(time.Second),
	)
	if err != nil {
		return nil, "", err
	}
	return user, token, nil
}

func (s *Service) HashPassword(password string) (string, error) {
	if err := ValidatePassword(password); err != nil {
		return "", err
	}
	h, err := bcrypt.GenerateFromPassword([]byte(password), 12)
	if err != nil {
		return "", err
	}
	return string(h), nil
}

func (s *Service) VerifyPassword(hash, password string) bool {
	if hash == "" {
		return false
	}
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)) == nil
}

func (s *Service) Login(ctx context.Context, email, password string) (string, *User, error) {
	id, _, _, _, status, _, _, _, hash, err := s.store.GetUserByEmail(ctx, email)
	if err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return "", nil, errors.New("invalid credentials")
		}
		return "", nil, err
	}
	if status != "active" {
		return "", nil, errors.New("user is not active")
	}
	if !s.VerifyPassword(hash, password) {
		return "", nil, errors.New("invalid credentials")
	}
	if updateErr := s.store.UpdateLoginTime(ctx, id); updateErr != nil {
		return "", nil, updateErr
	}
	u, err := s.store.GetUserByID(ctx, id)
	if err != nil {
		return "", nil, err
	}
	token, err := s.issueTokenForContext(ctx, u)
	if err != nil {
		return "", nil, err
	}
	return token, u, nil
}

func (s *Service) issueToken(user *User) (string, error) {
	return s.issueTokenForContext(context.Background(), user)
}

func (s *Service) issueTokenForContext(ctx context.Context, user *User) (string, error) {
	signed, session, err := s.prepareToken(user)
	if err != nil {
		return "", err
	}
	if err := s.store.CreateSession(ctx, session.ID, user.ID, session.IssuedAt, session.ExpiresAt); err != nil {
		return "", err
	}
	return signed, nil
}

func (s *Service) prepareToken(user *User) (string, localSessionDraft, error) {
	now := s.now().UTC().Truncate(time.Second)
	return s.prepareTokenAt(user, now, now.Add(s.ttlDuration))
}

func (s *Service) prepareTokenAt(
	user *User,
	issuedAt time.Time,
	expiresAt time.Time,
) (string, localSessionDraft, error) {
	return s.prepareTokenForSession(user, uuid.NewString(), issuedAt, expiresAt)
}

func (s *Service) prepareTokenForSession(
	user *User,
	sessionID string,
	issuedAt time.Time,
	expiresAt time.Time,
) (string, localSessionDraft, error) {
	issuedAt = issuedAt.UTC()
	expiresAt = expiresAt.UTC()
	if !uuidValid(sessionID) || issuedAt.IsZero() || !issuedAt.Before(expiresAt) {
		return "", localSessionDraft{}, errors.New("invalid dashboard session lifetime")
	}
	claims := TokenClaims{
		UserID: user.ID,
		Email:  user.Email,
		Role:   user.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			ID:        sessionID,
			ExpiresAt: jwt.NewNumericDate(expiresAt),
			IssuedAt:  jwt.NewNumericDate(issuedAt),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(s.jwtSecret)
	if err != nil {
		return "", localSessionDraft{}, err
	}
	return signed, localSessionDraft{ID: sessionID, IssuedAt: issuedAt.Unix(), ExpiresAt: expiresAt.Unix()}, nil
}

func (s *Service) ParseToken(raw string) (*TokenClaims, error) {
	t := &TokenClaims{}
	token, err := jwt.ParseWithClaims(raw, t, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method")
		}
		return s.jwtSecret, nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, errors.New("invalid token")
	}
	return t, nil
}

func (s *Service) ResolveSessionUser(ctx context.Context, claims *TokenClaims) (*User, map[string]bool, error) {
	if claims == nil || strings.TrimSpace(claims.UserID) == "" {
		return nil, nil, errors.New("invalid token")
	}

	user, err := s.store.GetUserByID(ctx, claims.UserID)
	if err != nil {
		return nil, nil, err
	}
	if user.Status != defaultUserStatusActive {
		return nil, nil, errors.New("user is not active")
	}
	if sessionErr := s.ensureSessionActive(ctx, claims); sessionErr != nil {
		return nil, nil, sessionErr
	}

	perms, err := s.store.GetEffectivePermissions(ctx, user.Role, user.ID)
	if err != nil {
		return nil, nil, err
	}
	return user, perms, nil
}

func (s *Service) ensureSessionActive(ctx context.Context, claims *TokenClaims) error {
	sessionID := strings.TrimSpace(claims.ID)
	if sessionID == "" {
		return nil
	}
	active, err := s.store.SessionActive(ctx, sessionID, claims.UserID, time.Now().Unix())
	if err != nil {
		return err
	}
	if !active {
		return errors.New("session is not active")
	}
	return nil
}

func (s *Service) RevokeToken(ctx context.Context, raw string) error {
	claims, err := s.ParseToken(raw)
	if err != nil {
		return nil
	}
	return s.store.RevokeSession(ctx, claims.ID)
}

func (s *Service) GetByID(ctx context.Context, id string) (*User, error) {
	return s.store.GetUserByID(ctx, id)
}

func (s *Service) EnsureBootstrapAdmin(ctx context.Context, email, password, name string) error {
	if strings.TrimSpace(email) == "" || strings.TrimSpace(password) == "" {
		return nil
	}
	n, _, _, _, _, _, _, _, _, err := s.store.GetUserByEmail(ctx, email)
	if err == nil && n != "" {
		return nil
	}
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return err
	}
	if err == nil {
		return nil
	}
	hash, err := s.HashPassword(password)
	if err != nil {
		return err
	}
	if _, err := s.store.CreateUser(ctx, email, defaultAdminName(name), hash, "admin", "active"); err != nil {
		return err
	}
	return nil
}

func (s *Service) CanBootstrap(ctx context.Context) (bool, error) {
	if s.firstAdminProvisioner != nil {
		return s.store.CanPrepareBootstrapAdmin(ctx)
	}
	count, err := s.store.CountUsers(ctx)
	if err != nil {
		return false, err
	}
	return count == 0, nil
}

// ConfigureFirstAdminProvisioner binds first Dashboard registration to the
// Router-owned Management authority. A nil provisioner retains the standalone
// Dashboard-only bootstrap used outside managed mode.
func (s *Service) ConfigureFirstAdminProvisioner(provisioner FirstAdminProvisioner) {
	s.firstAdminProvisioner = provisioner
}

func defaultAdminName(name string) string {
	if strings.TrimSpace(name) != "" {
		return strings.TrimSpace(name)
	}
	return "Admin"
}
