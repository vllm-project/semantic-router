package issuerverifier

import (
	"bytes"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"strings"
	"time"
)

const maximumAssertionBytes = 64 << 10

type parsedAssertion struct {
	unsigned  string
	signature []byte
	header    map[string]json.RawMessage
	claims    map[string]json.RawMessage
}

func parseAssertion(value string) (parsedAssertion, error) {
	if value == "" || len(value) > maximumAssertionBytes {
		return parsedAssertion{}, ErrDenied
	}
	parts := strings.Split(value, ".")
	if len(parts) != 3 {
		return parsedAssertion{}, ErrDenied
	}
	headerBytes, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil || base64.RawURLEncoding.EncodeToString(headerBytes) != parts[0] {
		return parsedAssertion{}, ErrDenied
	}
	claimsBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil || base64.RawURLEncoding.EncodeToString(claimsBytes) != parts[1] {
		return parsedAssertion{}, ErrDenied
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[2])
	if err != nil || len(signature) == 0 || base64.RawURLEncoding.EncodeToString(signature) != parts[2] {
		return parsedAssertion{}, ErrDenied
	}
	header, err := decodeRawObject(headerBytes)
	if err != nil {
		return parsedAssertion{}, ErrDenied
	}
	claims, err := decodeRawObject(claimsBytes)
	if err != nil {
		return parsedAssertion{}, ErrDenied
	}
	return parsedAssertion{
		unsigned: parts[0] + "." + parts[1], signature: signature,
		header: header, claims: claims,
	}, nil
}

func decodeRawObject(data []byte) (map[string]json.RawMessage, error) {
	decoder := json.NewDecoder(bytes.NewReader(data))
	token, tokenErr := decoder.Token()
	if tokenErr != nil || token != json.Delim('{') {
		return nil, errors.New("JSON object required")
	}
	result := make(map[string]json.RawMessage)
	for decoder.More() {
		nameToken, err := decoder.Token()
		name, ok := nameToken.(string)
		if err != nil || !ok || name == "" {
			return nil, errors.New("JSON object key is invalid")
		}
		if _, duplicate := result[name]; duplicate {
			return nil, errors.New("JSON object key is duplicated")
		}
		var raw json.RawMessage
		if err := decoder.Decode(&raw); err != nil {
			return nil, err
		}
		result[name] = append([]byte(nil), raw...)
	}
	if token, tokenErr = decoder.Token(); tokenErr != nil || token != json.Delim('}') {
		return nil, errors.New("JSON object is incomplete")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, errors.New("JSON object has trailing values")
	}
	return result, nil
}

func verifySignature(assertion parsedAssertion, key VerificationKey) error {
	digest := sha256.Sum256([]byte(assertion.unsigned))
	switch key.Algorithm {
	case "EdDSA":
		public, ok := key.PublicKey.(ed25519.PublicKey)
		if !ok || len(assertion.signature) != ed25519.SignatureSize ||
			!ed25519.Verify(public, []byte(assertion.unsigned), assertion.signature) {
			return ErrDenied
		}
	case "RS256":
		public, ok := key.PublicKey.(*rsa.PublicKey)
		if !ok || rsa.VerifyPKCS1v15(public, crypto.SHA256, digest[:], assertion.signature) != nil {
			return ErrDenied
		}
	case "PS256":
		public, ok := key.PublicKey.(*rsa.PublicKey)
		if !ok || rsa.VerifyPSS(public, crypto.SHA256, digest[:], assertion.signature,
			&rsa.PSSOptions{SaltLength: rsa.PSSSaltLengthEqualsHash}) != nil {
			return ErrDenied
		}
	case "ES256":
		public, ok := key.PublicKey.(*ecdsa.PublicKey)
		if !ok || len(assertion.signature) != 64 {
			return ErrDenied
		}
		r := new(big.Int).SetBytes(assertion.signature[:32])
		s := new(big.Int).SetBytes(assertion.signature[32:])
		if !ecdsa.Verify(public, digest[:], r, s) {
			return ErrDenied
		}
	default:
		return ErrDenied
	}
	return nil
}

func stringClaim(claims map[string]json.RawMessage, name string, required bool, limit int) (string, error) {
	raw, found := claims[name]
	if !found {
		if required {
			return "", ErrDenied
		}
		return "", nil
	}
	var value string
	if json.Unmarshal(raw, &value) != nil || !validClaimValue(value, limit) {
		return "", ErrDenied
	}
	return value, nil
}

func boolClaim(claims map[string]json.RawMessage, name string) (bool, bool, error) {
	raw, found := claims[name]
	if !found {
		return false, false, nil
	}
	var value bool
	if json.Unmarshal(raw, &value) != nil {
		return false, true, ErrDenied
	}
	return value, true, nil
}

func integerClaim(claims map[string]json.RawMessage, name string, required bool) (int64, error) {
	raw, found := claims[name]
	if !found {
		if required {
			return 0, ErrDenied
		}
		return 0, nil
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value json.Number
	if decoder.Decode(&value) != nil {
		return 0, ErrDenied
	}
	result, err := value.Int64()
	if err != nil || result <= 0 {
		return 0, ErrDenied
	}
	return result, nil
}

func audienceClaim(claims map[string]json.RawMessage) ([]string, error) {
	raw, found := claims["aud"]
	if !found {
		return nil, ErrDenied
	}
	var single string
	if json.Unmarshal(raw, &single) == nil {
		if !validClaimValue(single, 512) {
			return nil, ErrDenied
		}
		return []string{single}, nil
	}
	var values []string
	if json.Unmarshal(raw, &values) != nil || len(values) == 0 || len(values) > 16 {
		return nil, ErrDenied
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !validClaimValue(value, 512) {
			return nil, ErrDenied
		}
		if _, duplicate := seen[value]; duplicate {
			return nil, ErrDenied
		}
		seen[value] = struct{}{}
	}
	return values, nil
}

func stringArrayClaim(claims map[string]json.RawMessage, name string) ([]string, error) {
	raw, found := claims[name]
	if !found {
		return nil, ErrDenied
	}
	var values []string
	if json.Unmarshal(raw, &values) != nil || len(values) == 0 || len(values) > 16 {
		return nil, ErrDenied
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !validClaimValue(value, 64) {
			return nil, ErrDenied
		}
		if _, duplicate := seen[value]; duplicate {
			return nil, ErrDenied
		}
		seen[value] = struct{}{}
	}
	return values, nil
}

func allowedAudience(token, configured []string) bool {
	for _, candidate := range token {
		for _, allowed := range configured {
			if candidate == allowed {
				return true
			}
		}
	}
	return false
}

func validateAssertionTimes(claims map[string]json.RawMessage, now time.Time, skew, maximumLifetime time.Duration) (time.Time, time.Time, error) {
	issuedAt, err := integerClaim(claims, "iat", true)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	expiresAt, err := integerClaim(claims, "exp", true)
	if err != nil {
		return time.Time{}, time.Time{}, err
	}
	issued := time.Unix(issuedAt, 0).UTC()
	expires := time.Unix(expiresAt, 0).UTC()
	if issued.After(now.Add(skew)) || !expires.After(now.Add(-skew)) || !expires.After(issued) ||
		expires.Sub(issued) > maximumLifetime {
		return time.Time{}, time.Time{}, ErrDenied
	}
	if notBefore, err := integerClaim(claims, "nbf", false); err != nil {
		return time.Time{}, time.Time{}, err
	} else if notBefore > 0 && time.Unix(notBefore, 0).UTC().After(now.Add(skew)) {
		return time.Time{}, time.Time{}, ErrDenied
	}
	return issued, expires, nil
}

func claimName(issuer TrustedIssuer, logical, fallback string) string {
	if value := issuer.ClaimMapping[logical]; value != "" {
		return value
	}
	return fallback
}

func headerString(header map[string]json.RawMessage, name string) (string, error) {
	value, err := stringClaim(header, name, true, 128)
	if err != nil {
		return "", fmt.Errorf("%w: invalid assertion header", ErrDenied)
	}
	return value, nil
}
