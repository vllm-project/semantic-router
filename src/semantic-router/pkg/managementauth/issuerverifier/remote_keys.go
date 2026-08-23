package issuerverifier

import (
	"context"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rsa"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"math/big"
	"mime"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	maximumMetadataBytes = 1 << 20
	defaultKeyCacheTTL   = 5 * time.Minute
)

type RemoteKeySourceOptions struct {
	Transport http.RoundTripper
	CacheTTL  time.Duration
	Now       func() time.Time
}

type cachedKeys struct {
	set       KeySet
	expiresAt time.Time
}

type RemoteKeySource struct {
	client *http.Client
	ttl    time.Duration
	now    func() time.Time

	mu    sync.RWMutex
	cache map[string]cachedKeys
}

func NewRemoteKeySource(options RemoteKeySourceOptions) (*RemoteKeySource, error) {
	if options.Transport == nil {
		return nil, ErrUnavailable
	}
	ttl := options.CacheTTL
	if ttl == 0 {
		ttl = defaultKeyCacheTTL
	}
	if ttl < time.Minute || ttl > time.Hour {
		return nil, ErrUnavailable
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	client := &http.Client{
		Transport: options.Transport,
		Timeout:   15 * time.Second,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	return &RemoteKeySource{client: client, ttl: ttl, now: now, cache: make(map[string]cachedKeys)}, nil
}

func (source *RemoteKeySource) Keys(ctx context.Context, issuer TrustedIssuer) (KeySet, error) {
	if source == nil || source.client == nil || issuer.Validate() != nil {
		return KeySet{}, ErrUnavailable
	}
	identity := fmt.Sprintf("%s:%d:%s:%s", issuer.ID, issuer.Revision, issuer.DiscoveryURL, issuer.JWKSURL)
	now := source.now().UTC()
	source.mu.RLock()
	cached, found := source.cache[identity]
	source.mu.RUnlock()
	if found && now.Before(cached.expiresAt) {
		return cloneKeySet(cached.set), nil
	}
	set, err := source.fetch(ctx, issuer)
	if err != nil {
		return KeySet{}, err
	}
	source.mu.Lock()
	for key := range source.cache {
		if strings.HasPrefix(key, issuer.ID+":") && key != identity {
			delete(source.cache, key)
		}
	}
	source.cache[identity] = cachedKeys{set: cloneKeySet(set), expiresAt: now.Add(source.ttl)}
	source.mu.Unlock()
	return set, nil
}

func (source *RemoteKeySource) Invalidate(issuerID string) {
	if source == nil {
		return
	}
	source.mu.Lock()
	for key := range source.cache {
		if strings.HasPrefix(key, issuerID+":") {
			delete(source.cache, key)
		}
	}
	source.mu.Unlock()
}

// Refresh discards every cached revision for one issuer and fetches the exact
// supplied desired-state revision before an administrative refresh succeeds.
func (source *RemoteKeySource) Refresh(ctx context.Context, issuer TrustedIssuer) error {
	if source == nil {
		return ErrUnavailable
	}
	source.Invalidate(issuer.ID)
	_, err := source.Keys(ctx, issuer)
	return err
}

func (source *RemoteKeySource) Close() error {
	if source == nil {
		return nil
	}
	if closer, ok := source.client.Transport.(interface{ CloseIdleConnections() }); ok {
		closer.CloseIdleConnections()
	}
	source.mu.Lock()
	clear(source.cache)
	source.mu.Unlock()
	source.client = nil
	return nil
}

func (source *RemoteKeySource) fetch(ctx context.Context, issuer TrustedIssuer) (KeySet, error) {
	jwksURL := issuer.JWKSURL
	algorithms := []string(nil)
	if issuer.DiscoveryURL != "" {
		body, err := source.getJSON(ctx, issuer.DiscoveryURL)
		if err != nil {
			return KeySet{}, err
		}
		document, err := decodeRawObject(body)
		if err != nil {
			return KeySet{}, ErrUnavailable
		}
		discoveredIssuer, err := stringClaim(document, "issuer", true, 2048)
		if err != nil || discoveredIssuer != issuer.Issuer {
			return KeySet{}, ErrUnavailable
		}
		jwksURL, err = stringClaim(document, "jwks_uri", true, 2048)
		if err != nil || !canonicalHTTPSURL(jwksURL) {
			return KeySet{}, ErrUnavailable
		}
		algorithms, err = optionalStringArray(document, "id_token_signing_alg_values_supported", 16)
		if err != nil {
			return KeySet{}, ErrUnavailable
		}
	}
	body, err := source.getJSON(ctx, jwksURL)
	if err != nil {
		return KeySet{}, err
	}
	return parseJWKS(body, algorithms)
}

func (source *RemoteKeySource) getJSON(ctx context.Context, target string) ([]byte, error) {
	request, getJSONErr := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if getJSONErr != nil {
		return nil, ErrUnavailable
	}
	request.Header.Set("Accept", "application/json")
	response, getJSONErr := source.client.Do(request)
	if getJSONErr != nil {
		return nil, fmt.Errorf("%w: fetch issuer metadata", ErrUnavailable)
	}
	defer response.Body.Close()
	if response.StatusCode != http.StatusOK {
		return nil, ErrUnavailable
	}
	if contentType := response.Header.Get("Content-Type"); contentType != "" {
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil || (mediaType != "application/json" && mediaType != "application/jwk-set+json") {
			return nil, ErrUnavailable
		}
	}
	limited := io.LimitReader(response.Body, maximumMetadataBytes+1)
	body, getJSONErr := io.ReadAll(limited)
	if getJSONErr != nil || len(body) == 0 || len(body) > maximumMetadataBytes {
		return nil, ErrUnavailable
	}
	return body, nil
}

func parseJWKS(body []byte, discoveryAlgorithms []string) (KeySet, error) {
	document, err := decodeRawObject(body)
	if err != nil {
		return KeySet{}, ErrUnavailable
	}
	var rawKeys []json.RawMessage
	if raw, found := document["keys"]; !found || json.Unmarshal(raw, &rawKeys) != nil || len(rawKeys) == 0 || len(rawKeys) > 64 {
		return KeySet{}, ErrUnavailable
	}
	result := KeySet{Keys: make(map[string]VerificationKey, len(rawKeys))}
	for _, raw := range rawKeys {
		object, parseJWKSErr := decodeRawObject(raw)
		if parseJWKSErr != nil {
			return KeySet{}, ErrUnavailable
		}
		keyID, parseJWKSErr := stringClaim(object, "kid", true, 128)
		if parseJWKSErr != nil {
			return KeySet{}, ErrUnavailable
		}
		if _, duplicate := result.Keys[keyID]; duplicate {
			return KeySet{}, ErrUnavailable
		}
		algorithm, parseJWKSErr := stringClaim(object, "alg", true, 16)
		if parseJWKSErr != nil || !supportedAlgorithm(algorithm) ||
			(len(discoveryAlgorithms) > 0 && !slices.Contains(discoveryAlgorithms, algorithm)) {
			return KeySet{}, ErrUnavailable
		}
		if use, err := stringClaim(object, "use", false, 16); err != nil || (use != "" && use != "sig") {
			return KeySet{}, ErrUnavailable
		}
		if keyOperations, err := optionalStringArray(object, "key_ops", 8); err != nil ||
			(len(keyOperations) > 0 && !slices.Contains(keyOperations, "verify")) {
			return KeySet{}, ErrUnavailable
		}
		publicKey, parseJWKSErr := parseJWK(object, algorithm)
		if parseJWKSErr != nil {
			return KeySet{}, ErrUnavailable
		}
		result.Keys[keyID] = VerificationKey{Algorithm: algorithm, PublicKey: publicKey}
	}
	return result, nil
}

func parseJWK(object map[string]json.RawMessage, algorithm string) (any, error) {
	keyType, err := stringClaim(object, "kty", true, 16)
	if err != nil {
		return nil, err
	}
	switch {
	case algorithm == "EdDSA" && keyType == "OKP":
		curve, err := stringClaim(object, "crv", true, 16)
		encoded, decodeErr := stringClaim(object, "x", true, 128)
		if err != nil || decodeErr != nil || curve != "Ed25519" {
			return nil, ErrUnavailable
		}
		value, err := canonicalBase64URL(encoded, ed25519.PublicKeySize)
		if err != nil {
			return nil, err
		}
		return ed25519.PublicKey(value), nil
	case (algorithm == "RS256" || algorithm == "PS256") && keyType == "RSA":
		modulusText, err := stringClaim(object, "n", true, 2048)
		exponentText, exponentErr := stringClaim(object, "e", true, 16)
		if err != nil || exponentErr != nil {
			return nil, ErrUnavailable
		}
		modulus, err := canonicalBase64URL(modulusText, 0)
		if err != nil || len(modulus) < 256 || len(modulus) > 1024 {
			return nil, ErrUnavailable
		}
		exponentBytes, err := canonicalBase64URL(exponentText, 0)
		if err != nil || len(exponentBytes) == 0 || len(exponentBytes) > 4 {
			return nil, ErrUnavailable
		}
		exponent := 0
		for _, value := range exponentBytes {
			exponent = exponent<<8 | int(value)
		}
		if exponent < 3 || exponent%2 == 0 {
			return nil, ErrUnavailable
		}
		return &rsa.PublicKey{N: new(big.Int).SetBytes(modulus), E: exponent}, nil
	case algorithm == "ES256" && keyType == "EC":
		curve, err := stringClaim(object, "crv", true, 16)
		xText, xErr := stringClaim(object, "x", true, 128)
		yText, yErr := stringClaim(object, "y", true, 128)
		if err != nil || xErr != nil || yErr != nil || curve != "P-256" {
			return nil, ErrUnavailable
		}
		x, err := canonicalBase64URL(xText, 32)
		if err != nil {
			return nil, err
		}
		y, err := canonicalBase64URL(yText, 32)
		if err != nil {
			return nil, err
		}
		public := &ecdsa.PublicKey{Curve: elliptic.P256(), X: new(big.Int).SetBytes(x), Y: new(big.Int).SetBytes(y)}
		if !public.IsOnCurve(public.X, public.Y) {
			return nil, ErrUnavailable
		}
		return public, nil
	default:
		return nil, ErrUnavailable
	}
}

func canonicalBase64URL(value string, expectedLength int) ([]byte, error) {
	decoded, err := base64.RawURLEncoding.DecodeString(value)
	if err != nil || len(decoded) == 0 || (expectedLength > 0 && len(decoded) != expectedLength) ||
		base64.RawURLEncoding.EncodeToString(decoded) != value {
		return nil, ErrUnavailable
	}
	return decoded, nil
}

func optionalStringArray(object map[string]json.RawMessage, name string, maximum int) ([]string, error) {
	raw, found := object[name]
	if !found {
		return nil, nil
	}
	var values []string
	if json.Unmarshal(raw, &values) != nil || len(values) == 0 || len(values) > maximum {
		return nil, ErrUnavailable
	}
	seen := make(map[string]struct{}, len(values))
	for _, value := range values {
		if !validClaimValue(value, 64) {
			return nil, ErrUnavailable
		}
		if _, duplicate := seen[value]; duplicate {
			return nil, ErrUnavailable
		}
		seen[value] = struct{}{}
	}
	return values, nil
}

func supportedAlgorithm(value string) bool {
	return value == "EdDSA" || value == "RS256" || value == "PS256" || value == "ES256"
}

func cloneKeySet(source KeySet) KeySet {
	result := KeySet{Keys: make(map[string]VerificationKey, len(source.Keys))}
	for keyID, key := range source.Keys {
		result.Keys[keyID] = key
	}
	return result
}

var _ KeySource = (*RemoteKeySource)(nil)
