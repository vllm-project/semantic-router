package agentruntime

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscredential"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

var agentSecretAAD = []byte("vllm-sr/router-agent-secret/v1")

// SecretCodec keeps Tool Source and internal inference credentials in the
// Agent-specific control-plane encryption domain. The wire never exposes an
// encrypted or plaintext value; only the execution boundary may decrypt it.
type SecretCodec struct {
	keyring accesscredential.KEKKeyring
}

func NewSecretCodec(keys securitykeyring.Symmetric) (*SecretCodec, error) {
	keyring := accesscredential.KEKKeyring{
		ActiveVersion: keys.ActiveVersion,
		Keys:          cloneSecretKeys(keys.Keys),
	}
	if err := keyring.Validate(); err != nil {
		clearSecretKeys(keyring.Keys)
		return nil, fmt.Errorf("agent secret keyring: %w", err)
	}
	return &SecretCodec{keyring: keyring}, nil
}

func (codec *SecretCodec) Encrypt(_ context.Context, plaintext []byte) (agentmanagement.EncryptedSecret, error) {
	if codec == nil || len(plaintext) == 0 {
		return agentmanagement.EncryptedSecret{}, agentmanagement.ErrInvalid
	}
	envelope, err := codec.keyring.Seal(plaintext, agentSecretAAD)
	if err != nil {
		return agentmanagement.EncryptedSecret{}, err
	}
	return agentmanagement.EncryptedSecret{
		Ciphertext: envelope.Ciphertext, Nonce: envelope.Nonce, KEKVersion: envelope.KeyVersion,
	}, nil
}

func (codec *SecretCodec) Decrypt(_ context.Context, encrypted agentmanagement.EncryptedSecret) ([]byte, error) {
	if codec == nil {
		return nil, agentmanagement.ErrToolUnavailable
	}
	return codec.keyring.Open(accesscredential.Envelope{
		Ciphertext: encrypted.Ciphertext, Nonce: encrypted.Nonce, KeyVersion: encrypted.KEKVersion,
	}, agentSecretAAD)
}

func (codec *SecretCodec) Close() {
	if codec == nil {
		return
	}
	clearSecretKeys(codec.keyring.Keys)
	codec.keyring = accesscredential.KEKKeyring{}
}

func cloneSecretKeys(source map[string][]byte) map[string][]byte {
	result := make(map[string][]byte, len(source))
	for version, key := range source {
		result[version] = append([]byte(nil), key...)
	}
	return result
}

func clearSecretKeys(keys map[string][]byte) {
	for version, key := range keys {
		clear(key)
		delete(keys, version)
	}
}

var _ agentmanagement.SecretCodec = (*SecretCodec)(nil)
