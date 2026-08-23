// Package managementauth implements Router-issued Management access tokens.
// External identity assertions are bootstrap evidence and are never accepted
// directly as Management bearer tokens.
package managementauth

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

const tokenType = "JWT"

var ErrInvalidToken = errors.New("invalid management access token")

type EvidenceKind string

const (
	EvidenceHuman    EvidenceKind = "human"
	EvidenceWorkload EvidenceKind = "workload"
)

type HumanEvidence struct {
	AuthenticationTime int64    `json:"auth_time"`
	AAL                string   `json:"aal"`
	AMR                []string `json:"amr"`
}

type WorkloadEvidence struct {
	Class           string `json:"class"`
	SourceAssuredAt int64  `json:"source_assured_at"`
}

// Claims intentionally contains no permissions or role list. Authorization is
// evaluated from the current applied role-binding projection on every request.
type Claims struct {
	Issuer         string            `json:"iss"`
	Subject        string            `json:"sub"`
	SessionID      string            `json:"sid"`
	TokenID        string            `json:"jti"`
	Audience       string            `json:"aud"`
	IssuedAt       int64             `json:"iat"`
	ExpiresAt      int64             `json:"exp"`
	AuthSourceKind string            `json:"auth_source_kind"`
	AuthSourceID   string            `json:"auth_source_id"`
	EvidenceKind   EvidenceKind      `json:"evidence_kind"`
	Human          *HumanEvidence    `json:"human,omitempty"`
	Workload       *WorkloadEvidence `json:"workload,omitempty"`
}

type tokenHeader struct {
	Algorithm string `json:"alg"`
	Type      string `json:"typ"`
	KeyID     string `json:"kid"`
}

type TokenCodec struct {
	Keyring  securitykeyring.Signing
	Issuer   string
	Audience string
	MaxSkew  time.Duration
}

func (c TokenCodec) Issue(claims Claims) (string, error) {
	if err := c.validateConfiguration(true); err != nil {
		return "", err
	}
	if err := c.validateClaims(claims, time.Unix(claims.IssuedAt, 0), false); err != nil {
		return "", err
	}
	header := tokenHeader{Algorithm: "EdDSA", Type: tokenType, KeyID: c.Keyring.ActiveVersion}
	headerJSON, err := json.Marshal(header)
	if err != nil {
		return "", fmt.Errorf("marshal token header: %w", err)
	}
	claimsJSON, err := json.Marshal(claims)
	if err != nil {
		return "", fmt.Errorf("marshal token claims: %w", err)
	}
	unsigned := encode(headerJSON) + "." + encode(claimsJSON)
	signature := ed25519.Sign(c.Keyring.Private[c.Keyring.ActiveVersion], []byte(unsigned))
	return unsigned + "." + encode(signature), nil
}

func (c TokenCodec) Verify(token string, now time.Time) (Claims, error) {
	if err := c.validateConfiguration(false); err != nil {
		return Claims{}, err
	}
	parts := strings.Split(token, ".")
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return Claims{}, ErrInvalidToken
	}
	headerBytes, decodeErr := decode(parts[0])
	if decodeErr != nil {
		return Claims{}, ErrInvalidToken
	}
	var header tokenHeader
	if err := decodeStrict(headerBytes, &header); err != nil || header.Algorithm != "EdDSA" || header.Type != tokenType || header.KeyID == "" {
		return Claims{}, ErrInvalidToken
	}
	publicKey, ok := c.Keyring.Public[header.KeyID]
	if !ok {
		return Claims{}, ErrInvalidToken
	}
	signature, decodeErr := decode(parts[2])
	if decodeErr != nil || len(signature) != ed25519.SignatureSize ||
		!ed25519.Verify(publicKey, []byte(parts[0]+"."+parts[1]), signature) {
		return Claims{}, ErrInvalidToken
	}
	claimsBytes, decodeErr := decode(parts[1])
	if decodeErr != nil {
		return Claims{}, ErrInvalidToken
	}
	var claims Claims
	if err := decodeStrict(claimsBytes, &claims); err != nil {
		return Claims{}, ErrInvalidToken
	}
	if err := c.validateClaims(claims, now, true); err != nil {
		return Claims{}, ErrInvalidToken
	}
	return claims, nil
}

func (c TokenCodec) validateConfiguration(requirePrivate bool) error {
	if c.Issuer == "" || c.Audience == "" || strings.TrimSpace(c.Issuer) != c.Issuer || strings.TrimSpace(c.Audience) != c.Audience {
		return errors.New("management token issuer and audience are required and canonical")
	}
	if c.MaxSkew < 0 || c.MaxSkew > time.Minute {
		return errors.New("management token maximum clock skew must be between 0 and 1m")
	}
	if c.Keyring.ActiveVersion == "" || len(c.Keyring.Public) == 0 {
		return errors.New("management token signing keyring is unavailable")
	}
	if requirePrivate && len(c.Keyring.Private[c.Keyring.ActiveVersion]) != ed25519.PrivateKeySize {
		return errors.New("active management token signing key is unavailable")
	}
	return nil
}

func (c TokenCodec) validateClaims(claims Claims, now time.Time, enforceTime bool) error {
	if claims.Issuer != c.Issuer || claims.Audience != c.Audience ||
		claims.Subject == "" || claims.SessionID == "" || claims.TokenID == "" ||
		claims.AuthSourceKind == "" || claims.AuthSourceID == "" {
		return ErrInvalidToken
	}
	if claims.IssuedAt <= 0 || claims.ExpiresAt <= claims.IssuedAt {
		return ErrInvalidToken
	}
	switch claims.EvidenceKind {
	case EvidenceHuman:
		if claims.Human == nil || claims.Workload != nil || claims.Human.AuthenticationTime <= 0 ||
			claims.Human.AAL == "" || len(claims.Human.AMR) == 0 {
			return ErrInvalidToken
		}
	case EvidenceWorkload:
		if claims.Workload == nil || claims.Human != nil || claims.Workload.Class == "" || claims.Workload.SourceAssuredAt <= 0 {
			return ErrInvalidToken
		}
	default:
		return ErrInvalidToken
	}
	if !enforceTime {
		return nil
	}
	skewSeconds := int64(c.MaxSkew / time.Second)
	nowSeconds := now.Unix()
	if claims.IssuedAt > nowSeconds+skewSeconds || claims.ExpiresAt <= nowSeconds-skewSeconds {
		return ErrInvalidToken
	}
	return nil
}

func encode(value []byte) string { return base64.RawURLEncoding.EncodeToString(value) }

func decode(value string) ([]byte, error) {
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil || base64.RawURLEncoding.EncodeToString(decoded) != value {
		return nil, ErrInvalidToken
	}
	return decoded, nil
}

func decodeStrict(payload []byte, destination any) error {
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(destination); err != nil {
		return err
	}
	var trailing any
	if err := decoder.Decode(&trailing); err != io.EOF {
		return ErrInvalidToken
	}
	return nil
}
