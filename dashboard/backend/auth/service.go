package auth

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"golang.org/x/crypto/bcrypt"
)

type Service struct {
	store       *Store
	jwtSecret   []byte
	ttlDuration time.Duration
}

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
	token, err := s.issueToken(u)
	if err != nil {
		return "", nil, err
	}
	return token, u, nil
}

func (s *Service) issueToken(user *User) (string, error) {
	claims := TokenClaims{
		UserID: user.ID,
		Email:  user.Email,
		Role:   user.Role,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(s.ttlDuration)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
		},
	}
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(s.jwtSecret)
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

	perms, err := s.store.GetEffectivePermissions(ctx, user.Role, user.ID)
	if err != nil {
		return nil, nil, err
	}
	return user, perms, nil
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
