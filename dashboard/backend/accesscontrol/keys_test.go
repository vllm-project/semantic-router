package accesscontrol

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestNewSecretAndDigest(t *testing.T) {
	secret, prefix, err := NewSecret()
	require.NoError(t, err)
	require.NotEmpty(t, prefix)
	require.Contains(t, secret, prefix)
	digest, err := DigestKey(secret, "01234567890123456789012345678901")
	require.NoError(t, err)
	require.Len(t, digest, 64)
	require.NotContains(t, digest, secret)
}

func TestKeySecretEncryptionRoundTrip(t *testing.T) {
	const master = "01234567890123456789012345678901"
	secret, _, err := NewSecret()
	require.NoError(t, err)
	ciphertext, err := EncryptKeySecret(secret, master, "key-1")
	require.NoError(t, err)
	require.NotContains(t, ciphertext, secret)
	plain, err := DecryptKeySecret(ciphertext, master, "key-1")
	require.NoError(t, err)
	require.Equal(t, secret, plain)
	_, err = DecryptKeySecret(ciphertext, master, "key-2")
	require.Error(t, err)
}

func TestModelAllowed(t *testing.T) {
	tests := []struct {
		name     string
		model    string
		patterns []string
		allowed  bool
	}{
		{name: "exact", model: "vllm-sr/mom-v1-lite", patterns: []string{"vllm-sr/mom-v1-lite"}, allowed: true},
		{name: "prefix", model: "vllm-sr/mom-v1-ultra", patterns: []string{"vllm-sr/mom-*"}, allowed: true},
		{name: "wildcard", model: "anything", patterns: []string{"*"}, allowed: true},
		{name: "deny by default", model: "vllm-sr/mom-v1-vault", patterns: nil, allowed: false},
		{name: "different model", model: "vllm-sr/mom-v1-vault", patterns: []string{"vllm-sr/mom-v1-lite"}, allowed: false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			require.Equal(t, test.allowed, ModelAllowed(test.model, test.patterns))
		})
	}
}
