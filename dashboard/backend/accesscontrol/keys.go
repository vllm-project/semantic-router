package accesscontrol

import (
	"crypto/aes"
	"crypto/cipher"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"strings"
)

const keyMarker = "vsr_"

func NewSecret() (secret, prefix string, err error) {
	raw := make([]byte, 32)
	if _, err = rand.Read(raw); err != nil {
		return "", "", err
	}
	secret = keyMarker + base64.RawURLEncoding.EncodeToString(raw)
	prefix = secret[:min(len(secret), 12)]
	return secret, prefix, nil
}

func DigestKey(secret, hmacSecret string) (string, error) {
	secret = strings.TrimSpace(secret)
	if !strings.HasPrefix(secret, keyMarker) || len(secret) < 24 {
		return "", errors.New("invalid API key")
	}
	if len(hmacSecret) < 32 {
		return "", errors.New("ACCESS_CONTROL_KEY_SECRET must contain at least 32 characters")
	}
	mac := hmac.New(sha256.New, []byte(hmacSecret))
	_, _ = mac.Write([]byte(secret))
	return hex.EncodeToString(mac.Sum(nil)), nil
}

func EncryptKeySecret(secret, masterSecret, keyID string) (string, error) {
	if len(masterSecret) < 32 {
		return "", errors.New("ACCESS_CONTROL_KEY_SECRET must contain at least 32 characters")
	}
	block, err := aes.NewCipher(deriveEncryptionKey(masterSecret))
	if err != nil {
		return "", err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	nonce := make([]byte, gcm.NonceSize())
	if _, err = rand.Read(nonce); err != nil {
		return "", err
	}
	sealed := gcm.Seal(nonce, nonce, []byte(secret), []byte(keyID))
	return base64.RawURLEncoding.EncodeToString(sealed), nil
}

func DecryptKeySecret(ciphertext, masterSecret, keyID string) (string, error) {
	if strings.TrimSpace(ciphertext) == "" {
		return "", errors.New("stored API key secret is invalid")
	}
	if len(masterSecret) < 32 {
		return "", errors.New("ACCESS_CONTROL_KEY_SECRET must contain at least 32 characters")
	}
	raw, err := base64.RawURLEncoding.DecodeString(ciphertext)
	if err != nil {
		return "", errors.New("stored API key secret is invalid")
	}
	block, err := aes.NewCipher(deriveEncryptionKey(masterSecret))
	if err != nil {
		return "", err
	}
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return "", err
	}
	if len(raw) < gcm.NonceSize() {
		return "", errors.New("stored API key secret is invalid")
	}
	plain, err := gcm.Open(nil, raw[:gcm.NonceSize()], raw[gcm.NonceSize():], []byte(keyID))
	if err != nil {
		return "", errors.New("stored API key secret could not be decrypted")
	}
	return string(plain), nil
}

func deriveEncryptionKey(masterSecret string) []byte {
	sum := sha256.Sum256([]byte("vllm-sr/access-key-encryption/v1\x00" + masterSecret))
	return sum[:]
}

func ModelAllowed(model string, patterns []string) bool {
	model = strings.TrimSpace(model)
	if model == "" {
		return false
	}
	for _, pattern := range patterns {
		pattern = strings.TrimSpace(pattern)
		switch {
		case pattern == "*", pattern == model:
			return true
		case strings.HasSuffix(pattern, "*") && strings.HasPrefix(model, strings.TrimSuffix(pattern, "*")):
			return true
		}
	}
	return false
}
