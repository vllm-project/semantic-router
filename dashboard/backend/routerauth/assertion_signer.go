package routerauth

import (
	"crypto/ed25519"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"os"
	"strings"

	"github.com/golang-jwt/jwt/v5"
)

const dashboardAssertionAlgorithm = "EdDSA"

// AssertionSigner is the boundary between Dashboard identity exchange and key
// custody. Deployments may replace the file-backed signer with an HSM or KMS
// implementation without changing browser sessions or Router API clients.
type AssertionSigner interface {
	KeyID() string
	PublicJWK() PublicJWK
	Sign(jwt.Claims) (string, error)
}

// PublicJWK is the deliberately small Ed25519 key surface exposed by the
// Dashboard issuer. Private key material has no serializable representation.
type PublicJWK struct {
	KeyType   string `json:"kty"`
	Curve     string `json:"crv"`
	Use       string `json:"use"`
	Algorithm string `json:"alg"`
	KeyID     string `json:"kid"`
	X         string `json:"x"`
}

type fileAssertionSigner struct {
	keyID   string
	private ed25519.PrivateKey
	public  ed25519.PublicKey
}

// LoadEd25519AssertionSigner loads one PKCS#8 Ed25519 private key. The key is
// never generated implicitly: an ephemeral key would make Router trust and
// management-session recovery nondeterministic across Dashboard replicas.
func LoadEd25519AssertionSigner(path, keyID string) (AssertionSigner, error) {
	path = strings.TrimSpace(path)
	keyID = strings.TrimSpace(keyID)
	if path == "" || keyID == "" || len(keyID) > 128 || strings.ContainsAny(keyID, "\r\n\t ") {
		return nil, errors.New("dashboard assertion signing key file and key ID are required")
	}
	encoded, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read Dashboard assertion signing key: %w", err)
	}
	block, rest := pem.Decode(encoded)
	for index := range encoded {
		encoded[index] = 0
	}
	if block == nil || len(strings.TrimSpace(string(rest))) != 0 || block.Type != "PRIVATE KEY" {
		return nil, errors.New("dashboard assertion signing key must be one PKCS#8 PRIVATE KEY PEM block")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		for index := range block.Bytes {
			block.Bytes[index] = 0
		}
		return nil, fmt.Errorf("parse Dashboard assertion signing key: %w", err)
	}
	private, ok := parsed.(ed25519.PrivateKey)
	if !ok || len(private) != ed25519.PrivateKeySize {
		for index := range block.Bytes {
			block.Bytes[index] = 0
		}
		return nil, errors.New("dashboard assertion signing key must be Ed25519")
	}
	privateCopy := append(ed25519.PrivateKey(nil), private...)
	publicCopy := append(ed25519.PublicKey(nil), private.Public().(ed25519.PublicKey)...)
	for index := range block.Bytes {
		block.Bytes[index] = 0
	}
	return &fileAssertionSigner{keyID: keyID, private: privateCopy, public: publicCopy}, nil
}

func (signer *fileAssertionSigner) KeyID() string { return signer.keyID }

func (signer *fileAssertionSigner) PublicJWK() PublicJWK {
	return PublicJWK{
		KeyType: "OKP", Curve: "Ed25519", Use: "sig", Algorithm: dashboardAssertionAlgorithm,
		KeyID: signer.keyID, X: base64.RawURLEncoding.EncodeToString(signer.public),
	}
}

func (signer *fileAssertionSigner) Sign(claims jwt.Claims) (string, error) {
	if signer == nil || len(signer.private) != ed25519.PrivateKeySize {
		return "", errors.New("dashboard assertion signer is unavailable")
	}
	token := jwt.NewWithClaims(jwt.SigningMethodEdDSA, claims)
	token.Header["kid"] = signer.keyID
	return token.SignedString(signer.private)
}
