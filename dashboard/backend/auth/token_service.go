package auth

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

type TokenClaims struct {
	UserID string `json:"userId"`
	Email  string `json:"email"`
	Role   string `json:"role"`
	jwt.RegisteredClaims
}

type issuedToken struct {
	signed    string
	sessionID string
	issuedAt  time.Time
	expiresAt time.Time
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
	if err := s.store.RevokeSession(ctx, claims.ID); err != nil {
		return err
	}
	s.invalidateSessionAuthorization(claims.ID)
	return nil
}
