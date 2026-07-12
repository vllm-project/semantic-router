package auth

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

const dummyPasswordHash = passwordHashPrefix + "$2a$12$rbW/dA1D0Cq.EBraNzpl7efUNQegm9K3N5972H5RV4/cojTqK9jaO"

const emptyLoginAccountLimiterKey = "<empty-login-account>"

const (
	defaultJWTExpiryHours = 12
	maximumJWTExpiryHours = 7 * 24
)

var (
	ErrBootstrapClosed       = errors.New("bootstrap is disabled")
	ErrInvalidCredentials    = errors.New("invalid credentials")
	ErrCurrentPasswordFailed = errors.New("current password is invalid")
	ErrPasswordChanged       = errors.New("password changed concurrently")
	ErrPasswordWorkSaturated = errors.New("password verifier is busy")
	ErrLoginStateChanged     = errors.New("login state changed concurrently")
)

// LoginRateLimitError carries a safe Retry-After duration for the handler.
type LoginRateLimitError struct {
	RetryAfter time.Duration
}

func (e *LoginRateLimitError) Error() string { return "too many authentication attempts" }

type Service struct {
	store                  *Store
	jwtSecret              []byte
	ttlDuration            time.Duration
	policy                 *PasswordPolicy
	limiter                *LoginLimiter
	passwordChangeLimiter  *LoginLimiter
	verify                 func(hash, password string) bool
	loginPasswordWork      chan struct{}
	managementPasswordWork chan struct{}

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

type TokenClaims struct {
	UserID string `json:"userId"`
	Email  string `json:"email"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

func NewService(store *Store, secret string, ttlHours int) (*Service, error) {
	return newServiceWithEntropy(store, secret, ttlHours, rand.Reader)
}

func newServiceWithEntropy(
	store *Store,
	secret string,
	ttlHours int,
	entropy io.Reader,
) (*Service, error) {
	if ttlHours == 0 {
		ttlHours = defaultJWTExpiryHours
	}
	if ttlHours < 0 || ttlHours > maximumJWTExpiryHours {
		return nil, fmt.Errorf(
			"JWT expiry must be between 1 and %d hours",
			maximumJWTExpiryHours,
		)
	}
	if secret == "" {
		if entropy == nil {
			return nil, errors.New("generate JWT signing secret: entropy source is unavailable")
		}
		b := make([]byte, 32)
		if _, err := io.ReadFull(entropy, b); err != nil {
			return nil, fmt.Errorf("generate JWT signing secret: %w", err)
		}
		secret = base64.RawStdEncoding.EncodeToString(b)
	} else if err := validateConfiguredJWTSecret(secret); err != nil {
		return nil, err
	}
	service := &Service{
		store:                 store,
		jwtSecret:             []byte(secret),
		ttlDuration:           time.Duration(ttlHours) * time.Hour,
		policy:                NewPasswordPolicy(nil),
		limiter:               NewLoginLimiter(LoginLimiterConfig{}),
		passwordChangeLimiter: NewLoginLimiter(LoginLimiterConfig{}),
		loginPasswordWork: make(
			chan struct{},
			defaultLoginPasswordWorkConcurrency,
		),
		managementPasswordWork: make(
			chan struct{},
			defaultManagementPasswordWorkConcurrency,
		),
	}
	service.verify = service.VerifyPassword
	return service, nil
}

// SetAllowOpenBootstrap toggles the public web-form bootstrap endpoint.
func (s *Service) SetAllowOpenBootstrap(v bool) { s.allowOpenBootstrap = v }

// SetSetupMode toggles dashboard-first setup mode for trusted first-run bootstrap.
func (s *Service) SetSetupMode(v bool) { s.setupMode = v }

// SetPasswordPolicy installs the startup-validated password policy.
func (s *Service) SetPasswordPolicy(policy *PasswordPolicy) {
	if policy != nil {
		s.policy = policy
	}
}

// Close releases the authentication store owned by the service.
func (s *Service) Close() error {
	if s == nil || s.store == nil {
		return nil
	}
	return s.store.Close()
}

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

func (s *Service) Login(ctx context.Context, email, password string) (string, *User, error) {
	return s.LoginWithSource(ctx, email, password, "")
}

func (s *Service) LoginWithSource(
	ctx context.Context,
	email string,
	password string,
	source string,
) (string, *User, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	limiterAccount := email
	if limiterAccount == "" {
		limiterAccount = emptyLoginAccountLimiterKey
	}
	attempt, retryAfter := s.limiter.Reserve(limiterAccount, source)
	if attempt == nil {
		return "", nil, &LoginRateLimitError{RetryAfter: retryAfter}
	}
	defer attempt.Cancel()
	release, ok := acquirePasswordWork(s.loginPasswordWork)
	if !ok {
		return "", nil, &LoginRateLimitError{RetryAfter: time.Second}
	}
	defer release()

	passwordState, err := s.store.getUserPasswordStateByEmail(ctx, email)
	found := err == nil
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return "", nil, err
	}
	verificationHash := dummyPasswordHash
	if found {
		verificationHash = passwordState.hash
	}
	passwordMatches := s.verify(verificationHash, password)
	if !found || passwordState.user.Status != defaultUserStatusActive || !passwordMatches {
		attempt.Fail()
		return "", nil, ErrInvalidCredentials
	}
	upgradedHash := ""
	if passwordHashNeedsUpgrade(passwordState.hash) {
		upgraded, hashErr := hashVersionedPassword(password)
		if hashErr != nil {
			attempt.Finish()
			return "", nil, hashErr
		}
		upgradedHash = upgraded
	}
	u := passwordState.user
	issued, err := s.prepareToken(u)
	if err != nil {
		attempt.Finish()
		return "", nil, err
	}
	if err := s.store.CompleteLogin(
		ctx,
		u.ID,
		passwordState.hash,
		passwordState.authGeneration,
		upgradedHash,
		issued,
	); err != nil {
		if errors.Is(err, ErrLoginStateChanged) {
			attempt.Fail()
			return "", nil, ErrInvalidCredentials
		}
		attempt.Finish()
		return "", nil, err
	}
	attempt.Succeed()
	return issued.signed, u, nil
}

func (s *Service) issueToken(user *User) (string, error) {
	return s.issueTokenForContext(context.Background(), user)
}

func (s *Service) issueTokenForContext(ctx context.Context, user *User) (string, error) {
	issued, err := s.prepareToken(user)
	if err != nil {
		return "", err
	}
	if err := s.store.CreateSession(
		ctx,
		issued.sessionID,
		user.ID,
		issued.issuedAt.Unix(),
		issued.expiresAt.Unix(),
	); err != nil {
		return "", err
	}
	return issued.signed, nil
}

type issuedToken struct {
	signed    string
	sessionID string
	issuedAt  time.Time
	expiresAt time.Time
}

func (s *Service) prepareToken(user *User) (*issuedToken, error) {
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
		return nil, err
	}
	return &issuedToken{
		signed:    signed,
		sessionID: sessionID,
		issuedAt:  now,
		expiresAt: expiresAt,
	}, nil
}

func (s *Service) ParseToken(raw string) (*TokenClaims, error) {
	t := &TokenClaims{}
	token, err := jwt.ParseWithClaims(raw, t, func(token *jwt.Token) (interface{}, error) {
		if token.Method != jwt.SigningMethodHS256 {
			return nil, fmt.Errorf("unexpected signing method")
		}
		return s.jwtSecret, nil
	},
		jwt.WithValidMethods([]string{jwt.SigningMethodHS256.Alg()}),
		jwt.WithExpirationRequired(),
	)
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, errors.New("invalid token")
	}
	if strings.TrimSpace(t.ID) == "" {
		return nil, errors.New("token session id is required")
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
		return errors.New("token session id is required")
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
	email = strings.ToLower(strings.TrimSpace(email))
	if email == "" || strings.TrimSpace(password) == "" {
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
	hash, err := s.HashPasswordForUser(email, password)
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
