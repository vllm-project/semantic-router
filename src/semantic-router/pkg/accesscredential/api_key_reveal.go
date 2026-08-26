package accesscredential

// APIKeyRevealAAD binds an encrypted API-key secret to exactly one namespace,
// logical key, credential version, and public key identifier. Every producer
// and consumer of revealable API-key credentials must use this same domain.
func APIKeyRevealAAD(namespaceID, keyID, credentialID, kid string) []byte {
	return []byte("vllm-sr/api-key-reveal/v1\x00" + namespaceID + "\x00" + keyID + "\x00" + credentialID + "\x00" + kid)
}
