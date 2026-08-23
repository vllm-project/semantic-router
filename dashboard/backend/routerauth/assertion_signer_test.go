package routerauth

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"os"
	"path/filepath"
	"testing"

	"github.com/golang-jwt/jwt/v5"
)

func TestLoadEd25519AssertionSignerPublishesOnlyPublicMaterial(t *testing.T) {
	t.Parallel()
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "dashboard-issuer.pem")
	if writeErr := os.WriteFile(path, pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: encoded}), 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}

	signer, err := LoadEd25519AssertionSigner(path, "dashboard-key-1")
	if err != nil {
		t.Fatalf("LoadEd25519AssertionSigner() error = %v", err)
	}
	jwk := signer.PublicJWK()
	if jwk.KeyType != "OKP" || jwk.Curve != "Ed25519" || jwk.Algorithm != "EdDSA" ||
		jwk.KeyID != "dashboard-key-1" || jwk.X == "" {
		t.Fatalf("public JWK = %+v", jwk)
	}

	signed, err := signer.Sign(jwt.MapClaims{"sub": "user-1"})
	if err != nil || signed == "" {
		t.Fatalf("Sign() token=%q error=%v", signed, err)
	}
	parsed, err := jwt.Parse(signed, func(token *jwt.Token) (any, error) {
		return private.Public(), nil
	}, jwt.WithValidMethods([]string{"EdDSA"}))
	if err != nil || parsed == nil || !parsed.Valid || parsed.Header["kid"] != "dashboard-key-1" {
		t.Fatalf("parsed assertion=%v error=%v", parsed, err)
	}
}

func TestLoadEd25519AssertionSignerRejectsNonEd25519PEM(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "invalid.pem")
	if err := os.WriteFile(path, []byte("not a private key"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadEd25519AssertionSigner(path, "dashboard-key-1"); err == nil {
		t.Fatal("LoadEd25519AssertionSigner() unexpectedly accepted invalid PEM")
	}
}
