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

	// allowOpenBootstrap gates the public web-form bootstrap endpoint (off by default).
	allowOpenBootstrap bool
	// setupMode enables bootstrap during dashboard-first local install (trusted phase).
	setupMode bool
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
	return &Service{store: store, jwtSecret: []byte(secret), ttlDuration: time.Duration(ttlHours) * time.Hour}
}

// SetAllowOpenBootstrap toggles the public web-form bootstrap endpoint.
func (s *Service) SetAllowOpenBootstrap(v bool) { s.allowOpenBootstrap = v }

// SetSetupMode toggles dashboard-first setup mode for trusted first-run bootstrap.
func (s *Service) SetSetupMode(v bool) { s.setupMode = v }

// OpenBootstrapEnabled reports whether the public web-form bootstrap endpoint is enabled.
func (s *Service) OpenBootstrapEnabled() bool { return s.allowOpenBootstrap || s.setupMode }

// BootstrapRegister atomically creates the first admin. The "no users yet" check and
// the create run under bootstrapMu, so two concurrent requests cannot each create an
// admin (closing the time-of-check-to-time-of-use race in the public bootstrap path).
// Returns ErrBootstrapClosed if any user already exists.
func (s *Service) BootstrapRegister(ctx context.Context, email, name, hash string) (*User, error) {
	s.bootstrapMu.Lock()
	defer s.bootstrapMu.Unlock()
	ok, err := s.CanBootstrap(ctx)
	if err != nil {
		return nil, err
	}
	if !ok {
		return nil, ErrBootstrapClosed
	}
	return s.store.CreateUser(ctx, email, defaultAdminName(name), hash, RoleAdmin, "active")
}

func (s *Service) HashPassword(password string) (string, error) {
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
	id, e, n, role, status, _, _, _, hash, err := s.store.GetUserByEmail(ctx, email)
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
	u := &User{ID: id, Email: e, Name: n, Role: role, Status: status}
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
	now := time.Now()
	expiresAt := now.Add(s.ttlDuration)
	sessionID := uuid.NewString()
	claims := TokenClaims{
		UserID: user.ID,
		Email:  user.Email,
		Role:   user.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			ID:        sessionID,
			ExpiresAt: jwt.NewNumericDate(expiresAt),
			IssuedAt:  jwt.NewNumericDate(now),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	signed, err := token.SignedString(s.jwtSecret)
	if err != nil {
		return "", err
	}
	if err := s.store.CreateSession(ctx, sessionID, user.ID, now.Unix(), expiresAt.Unix()); err != nil {
		return "", err
	}
	return signed, nil
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
	count, err := s.store.CountUsers(ctx)
	if err != nil {
		return false, err
	}
	return count == 0, nil
}

func defaultAdminName(name string) string {
	if strings.TrimSpace(name) != "" {
		return strings.TrimSpace(name)
	}
	return "Admin"
}
