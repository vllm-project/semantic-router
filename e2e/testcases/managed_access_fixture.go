package testcases

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	managedAccessNamespace            = "vllm-semantic-router-system"
	managedAccessService              = "semantic-router-management"
	managedAccessServicePort          = "8080"
	managedAccessServerName           = "semantic-router-management.vllm-semantic-router-system.svc.cluster.local"
	managedAccessIdentitySecret       = "semantic-router-dashboard-e2e-dashboard"
	managedAccessAssertionKey         = "assertion-signing-key.pem"
	managedAccessCABundle             = "ca.crt"
	managedAccessIssuerID             = "10000000-0000-4000-8000-000000000011"
	managedAccessIssuerURL            = "https://semantic-router-dashboard-issuer.vllm-semantic-router-system.svc.cluster.local:9443"
	managedAccessAssertionKeyID       = "dashboard-e2e-v1"
	managedAccessAudience             = "vllm-sr-management"
	managedAccessManagementMediaType  = "application/vnd.vllm-semantic-router.management.v1+json"
	managedAccessManagementBasePath   = "/management/v1"
	managedAccessNamespaceHeader      = "VLLM-SR-Namespace"
	managedAccessIdempotencyHeader    = "Idempotency-Key"
	managedAccessFixtureBackendOrigin = "http://vllm-llama3-8b-instruct.default.svc.cluster.local:8000/v1"
	managedAccessFixtureBackendModel  = "base-model"
)

type managedAccessClient struct {
	baseURL string
	token   string
	client  *http.Client
	verbose bool
}

type managedAccessResponseError struct {
	method   string
	path     string
	expected []int
	status   int
	code     string
	body     string
}

func (err *managedAccessResponseError) Error() string {
	return fmt.Sprintf(
		"direct Management %s %s: expected %v, got %d: %s",
		err.method, err.path, err.expected, err.status, err.body,
	)
}

type managedAccessFixture struct {
	namespaceID    string
	keyID          string
	keyRevision    uint64
	secret         string
	authorizedName string
	hiddenName     string
}

type managedAccessResourceReference struct {
	Kind     string `json:"kind"`
	ID       string `json:"id"`
	Revision uint64 `json:"revision"`
}

type managedAccessMutationReceipt struct {
	Resource *managedAccessResourceReference `json:"resource,omitempty"`
}

type managedAccessIssuedKey struct {
	Data struct {
		KeyID    string `json:"keyId"`
		Status   string `json:"status"`
		Revision uint64 `json:"revision"`
	} `json:"data"`
	Secret string `json:"secret"`
}

type managedAccessDashboardClaims struct {
	UserID  string `json:"userId"`
	Email   string `json:"email"`
	ID      string `json:"jti"`
	Issued  int64  `json:"iat"`
	Expires int64  `json:"exp"`
}

func parseManagedAccessDashboardClaims(token string) (managedAccessDashboardClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return managedAccessDashboardClaims{}, fmt.Errorf("Dashboard session token is not a compact JWT")
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return managedAccessDashboardClaims{}, fmt.Errorf("decode Dashboard session claims: %w", err)
	}
	defer clear(payload)
	var claims managedAccessDashboardClaims
	if err := json.Unmarshal(payload, &claims); err != nil {
		return managedAccessDashboardClaims{}, fmt.Errorf("decode Dashboard session claims: %w", err)
	}
	if claims.UserID == "" || claims.Email == "" || claims.ID == "" || claims.Issued <= 0 ||
		claims.Expires <= claims.Issued || claims.Expires <= time.Now().UTC().Unix() {
		return managedAccessDashboardClaims{}, fmt.Errorf("Dashboard session claims are incomplete")
	}
	return claims, nil
}

func parseManagedAccessAssertionKey(encoded []byte) (ed25519.PrivateKey, error) {
	block, rest := pem.Decode(encoded)
	if block == nil || block.Type != "PRIVATE KEY" || strings.TrimSpace(string(rest)) != "" {
		return nil, fmt.Errorf("managed-access assertion key is not one PKCS#8 private key")
	}
	parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parse managed-access assertion key: %w", err)
	}
	privateKey, ok := parsed.(ed25519.PrivateKey)
	if !ok || len(privateKey) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("managed-access assertion key is not Ed25519")
	}
	return append(ed25519.PrivateKey(nil), privateKey...), nil
}

func signManagedAccessAssertion(
	privateKey ed25519.PrivateKey,
	principal managedAccessDashboardClaims,
	nonce string,
) (string, error) {
	now := time.Now().UTC()
	jti, err := managedAccessUUID()
	if err != nil {
		return "", fmt.Errorf("create managed-access assertion identity: %w", err)
	}
	header, err := managedAccessJWTPart(map[string]string{
		"alg": "EdDSA", "kid": managedAccessAssertionKeyID, "typ": "JWT",
	})
	if err != nil {
		return "", err
	}
	claims, err := managedAccessJWTPart(map[string]interface{}{
		"iss": managedAccessIssuerURL, "sub": principal.UserID, "aud": managedAccessAudience,
		"iat": now.Unix(), "exp": now.Add(time.Minute).Unix(), "jti": jti, "nonce": nonce,
		"source_session_exp": principal.Expires,
		"sid":                principal.ID, "auth_time": principal.Issued, "aal": "aal1", "amr": []string{"pwd"},
		"email": principal.Email, "email_verified": true,
	})
	if err != nil {
		return "", err
	}
	signingInput := header + "." + claims
	signature := ed25519.Sign(privateKey, []byte(signingInput))
	defer clear(signature)
	return signingInput + "." + base64.RawURLEncoding.EncodeToString(signature), nil
}

func managedAccessJWTPart(value interface{}) (string, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return "", fmt.Errorf("encode managed-access assertion: %w", err)
	}
	defer clear(encoded)
	return base64.RawURLEncoding.EncodeToString(encoded), nil
}

func managedAccessUUID() (string, error) {
	var value [16]byte
	if _, err := rand.Read(value[:]); err != nil {
		return "", err
	}
	value[6] = (value[6] & 0x0f) | 0x40
	value[8] = (value[8] & 0x3f) | 0x80
	encoded := hex.EncodeToString(value[:])
	return encoded[0:8] + "-" + encoded[8:12] + "-" + encoded[12:16] + "-" +
		encoded[16:20] + "-" + encoded[20:], nil
}

func (client *managedAccessClient) request(
	ctx context.Context,
	namespaceID string,
	method string,
	path string,
	idempotencyKey string,
	payload interface{},
	headers http.Header,
	expectedStatuses []int,
	result interface{},
) (int, http.Header, error) {
	encoded, err := encodeManagedAccessRequest(payload)
	if err != nil {
		return 0, nil, err
	}
	defer clear(encoded)
	requestURL := strings.TrimRight(client.baseURL, "/") + managedAccessManagementBasePath + path
	if client.verbose {
		fmt.Printf("[Management] %s %s\n", method, path)
	}
	request, err := newManagedAccessRequest(
		ctx, method, requestURL, client.token, namespaceID, idempotencyKey, encoded, payload != nil, headers,
	)
	if err != nil {
		return 0, nil, err
	}
	response, err := client.client.Do(request)
	if err != nil {
		return 0, nil, fmt.Errorf("direct Management request failed: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	return decodeManagedAccessResponse(response, method, path, expectedStatuses, result)
}

func encodeManagedAccessRequest(payload interface{}) ([]byte, error) {
	if payload == nil {
		return nil, nil
	}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal direct Management request: %w", err)
	}
	return encoded, nil
}

func newManagedAccessRequest(
	ctx context.Context,
	method string,
	requestURL string,
	token string,
	namespaceID string,
	idempotencyKey string,
	encoded []byte,
	hasPayload bool,
	headers http.Header,
) (*http.Request, error) {
	var requestBody io.Reader
	if hasPayload {
		requestBody = bytes.NewReader(encoded)
	}
	request, err := http.NewRequestWithContext(ctx, method, requestURL, requestBody)
	if err != nil {
		return nil, fmt.Errorf("create direct Management request: %w", err)
	}
	request.Header.Set("Accept", managedAccessManagementMediaType)
	setManagedAccessRequestHeaders(request, token, namespaceID, idempotencyKey, hasPayload, headers)
	return request, nil
}

func setManagedAccessRequestHeaders(
	request *http.Request,
	token string,
	namespaceID string,
	idempotencyKey string,
	hasPayload bool,
	headers http.Header,
) {
	if hasPayload {
		request.Header.Set("Content-Type", managedAccessManagementMediaType)
	}
	if token != "" {
		request.Header.Set("Authorization", "Bearer "+token)
	}
	if namespaceID != "" {
		request.Header.Set(managedAccessNamespaceHeader, namespaceID)
	}
	if idempotencyKey != "" {
		request.Header.Set(managedAccessIdempotencyHeader, idempotencyKey)
	}
	for name, values := range headers {
		for _, value := range values {
			request.Header.Add(name, value)
		}
	}
}

func decodeManagedAccessResponse(
	response *http.Response,
	method string,
	path string,
	expectedStatuses []int,
	result interface{},
) (int, http.Header, error) {
	body, err := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if err != nil {
		return response.StatusCode, response.Header.Clone(), fmt.Errorf("read direct Management response: %w", err)
	}
	defer clear(body)
	if !managedAccessExpectedStatus(response.StatusCode, expectedStatuses) {
		return response.StatusCode, response.Header.Clone(), &managedAccessResponseError{
			method: method, path: path, expected: append([]int(nil), expectedStatuses...),
			status: response.StatusCode, code: managedAccessResponseCode(body),
			body: truncateString(string(body), 300),
		}
	}
	if result != nil {
		if err := json.Unmarshal(body, result); err != nil {
			return response.StatusCode, response.Header.Clone(), fmt.Errorf("decode direct Management response: %w", err)
		}
	}
	return response.StatusCode, response.Header.Clone(), nil
}

func managedAccessResponseCode(body []byte) string {
	var envelope struct {
		Code  string `json:"code"`
		Error *struct {
			Code string `json:"code"`
		} `json:"error,omitempty"`
	}
	if err := json.Unmarshal(body, &envelope); err != nil {
		return ""
	}
	if envelope.Code != "" {
		return envelope.Code
	}
	if envelope.Error != nil {
		return envelope.Error.Code
	}
	return ""
}

func managedAccessExpectedStatus(status int, expected []int) bool {
	for _, candidate := range expected {
		if status == candidate {
			return true
		}
	}
	return false
}

func (client *managedAccessClient) mutation(
	ctx context.Context,
	namespaceID string,
	method string,
	path string,
	idempotencyKey string,
	payload interface{},
) (managedAccessResourceReference, error) {
	var receipt managedAccessMutationReceipt
	if _, _, err := client.request(
		ctx, namespaceID, method, path, idempotencyKey, payload, nil,
		[]int{http.StatusCreated}, &receipt,
	); err != nil {
		return managedAccessResourceReference{}, err
	}
	if receipt.Resource == nil || receipt.Resource.ID == "" || receipt.Resource.Revision == 0 {
		return managedAccessResourceReference{}, fmt.Errorf("direct Management mutation returned no resource identity")
	}
	return *receipt.Resource, nil
}

func publishManagedAccessEntrypoint(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	entrypointID string,
	initialRevision uint64,
) error {
	const maximumAttempts = 5
	etag := `"ep:` + strconv.FormatUint(initialRevision, 10) + `"`
	idempotencyKey := "managed-access-publish-" + entrypointID
	var lastErr error
	for attempt := 1; attempt <= maximumAttempts; attempt++ {
		status, _, err := client.request(
			ctx, namespaceID, http.MethodPost, "/routing/entrypoints/"+entrypointID+":publish",
			idempotencyKey, nil, http.Header{"If-Match": []string{etag}},
			[]int{http.StatusAccepted}, nil,
		)
		if err == nil {
			return nil
		}
		lastErr = err
		if status != http.StatusPreconditionFailed || attempt == maximumAttempts {
			break
		}
		_, headers, refreshErr := client.request(
			ctx, namespaceID, http.MethodGet, "/routing/entrypoints/"+entrypointID,
			"", nil, nil, []int{http.StatusOK}, nil,
		)
		if refreshErr != nil {
			return fmt.Errorf("refresh Entrypoint revision after publication conflict: %w", refreshErr)
		}
		etag = strings.TrimSpace(headers.Get("ETag"))
		if !validManagedAccessEntrypointETag(etag) {
			return fmt.Errorf("refresh Entrypoint revision returned invalid ETag %q", etag)
		}
	}
	return fmt.Errorf("publication did not converge after %d attempts: %w", maximumAttempts, lastErr)
}

func validManagedAccessEntrypointETag(value string) bool {
	if !strings.HasPrefix(value, `"ep:`) || !strings.HasSuffix(value, `"`) {
		return false
	}
	revision, err := strconv.ParseUint(strings.TrimSuffix(strings.TrimPrefix(value, `"ep:`), `"`), 10, 64)
	return err == nil && revision > 0
}

func waitManagedAccessEntrypoint(
	ctx context.Context,
	client *managedAccessClient,
	namespaceID string,
	entrypointID string,
) error {
	deadline := time.Now().Add(90 * time.Second)
	var lastStatus string
	for time.Now().Before(deadline) {
		var detail struct {
			Data struct {
				Status string `json:"status"`
			} `json:"data"`
		}
		_, _, err := client.request(
			ctx, namespaceID, http.MethodGet, "/routing/entrypoints/"+entrypointID,
			"", nil, nil, []int{http.StatusOK}, &detail,
		)
		if err == nil {
			lastStatus = detail.Data.Status
			if lastStatus == "active" {
				return nil
			}
		}
		timer := time.NewTimer(time.Second)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return fmt.Errorf("fixture Entrypoint %q did not become active (last status %q)", entrypointID, lastStatus)
}
