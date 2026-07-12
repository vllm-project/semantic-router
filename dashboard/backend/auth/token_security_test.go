package auth

import (
	"context"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v5"
)

func TestParseTokenRejectsMissingSessionID(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	raw := signTokenForTest(t, svc, jwt.SigningMethodHS256, jwt.RegisteredClaims{
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
	})
	if _, err := svc.ParseToken(raw); err == nil {
		t.Fatal("ParseToken() accepted a token without jti")
	}
	if err := svc.ensureSessionActive(context.Background(), &TokenClaims{}); err == nil {
		t.Fatal("ensureSessionActive() accepted claims without jti")
	}
}

func TestParseTokenRejectsNonHS256HMACAlgorithm(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	raw := signTokenForTest(t, svc, jwt.SigningMethodHS512, jwt.RegisteredClaims{
		ID:        "session-id",
		ExpiresAt: jwt.NewNumericDate(time.Now().Add(time.Hour)),
	})
	if _, err := svc.ParseToken(raw); err == nil {
		t.Fatal("ParseToken() accepted an HS512 token")
	}
}

func TestParseTokenRequiresExpiration(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	raw := signTokenForTest(t, svc, jwt.SigningMethodHS256, jwt.RegisteredClaims{
		ID: "session-id",
	})
	if _, err := svc.ParseToken(raw); err == nil {
		t.Fatal("ParseToken() accepted a token without exp")
	}
}

func signTokenForTest(
	t *testing.T,
	svc *Service,
	method jwt.SigningMethod,
	registeredClaims jwt.RegisteredClaims,
) string {
	t.Helper()
	token := jwt.NewWithClaims(method, TokenClaims{
		UserID:           "user-id",
		Email:            "person@example.com",
		Role:             RoleRead,
		RegisteredClaims: registeredClaims,
	})
	raw, err := token.SignedString(svc.jwtSecret)
	if err != nil {
		t.Fatalf("SignedString() error = %v", err)
	}
	return raw
}
