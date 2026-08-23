package providerdiscovery

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/providercredential"
)

const maximumDiscoveryResponseBytes = 4 << 20

type CredentialMetadataReader interface {
	GetProviderCredential(context.Context, accesscontrol.NamespaceID, string) (providercredential.Credential, error)
}

// Executor is the privileged boundary between a validated Provider Definition
// and outbound discovery I/O. It chooses no product behavior: the plan fixes
// origin/path/credential binding and the registry fixes the wire adapter.
type Executor struct {
	Registry           *Registry
	CredentialMetadata CredentialMetadataReader
	Credentials        CredentialResolver
	EgressPolicy       backendegress.Policy
	Transport          http.RoundTripper
	Claims             ClaimCodec
	ClaimTTL           time.Duration
	Now                func() time.Time
}

func (executor Executor) Execute(ctx context.Context, request ExecuteRequest) (Result, error) {
	if executor.Registry == nil || executor.Transport == nil {
		return Result{}, fmt.Errorf("%w: discovery registry and egress transport are required", ErrInvalidRequest)
	}
	if _, err := executor.EgressPolicy.AuthorizeOrigin(request.Plan.NormalizedOrigin); err != nil {
		return Result{}, fmt.Errorf("%w: discovery origin is denied", ErrInvalidRequest)
	}
	adapter, executeErr := executor.Registry.Adapter(request.Plan.DiscoveryAdapterID)
	if executeErr != nil {
		return Result{}, executeErr
	}
	if err := adapter.ValidateDiscovery(ctx, request.Plan); err != nil {
		return Result{}, fmt.Errorf("%w: %w", ErrInvalidRequest, err)
	}
	credential, credentialVersion, executeErr := executor.resolveCredential(ctx, request)
	if executeErr != nil {
		return Result{}, executeErr
	}
	query, executeErr := adapter.Query(request.Plan)
	if executeErr != nil {
		return Result{}, fmt.Errorf("%w: prepare adapter query: %w", ErrInvalidRequest, executeErr)
	}
	target, executeErr := discoveryURL(request.Plan.NormalizedOrigin, request.Plan.Path, query)
	if executeErr != nil {
		return Result{}, executeErr
	}
	httpRequest, executeErr := http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	if executeErr != nil {
		return Result{}, fmt.Errorf("%w: build discovery request", ErrInvalidRequest)
	}
	httpRequest.Header.Set("Accept", "application/json")
	for name, value := range request.Plan.Headers {
		httpRequest.Header.Set(name, value)
	}
	sensitiveHeaders, executeErr := applyCredential(httpRequest.Header, credential)
	if executeErr != nil {
		return Result{}, executeErr
	}
	defer scrubHeaders(httpRequest.Header, sensitiveHeaders)

	client := &http.Client{
		Transport: executor.Transport,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	response, executeErr := client.Do(httpRequest)
	if executeErr != nil {
		return Result{}, fmt.Errorf("%w: request failed", ErrUpstream)
	}
	defer response.Body.Close()
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		return Result{}, fmt.Errorf("%w: upstream returned status %d", ErrUpstream, response.StatusCode)
	}
	payload, executeErr := io.ReadAll(io.LimitReader(response.Body, maximumDiscoveryResponseBytes+1))
	if executeErr != nil {
		return Result{}, fmt.Errorf("%w: read response", ErrUpstream)
	}
	if len(payload) > maximumDiscoveryResponseBytes {
		return Result{}, fmt.Errorf("%w: response exceeds %d bytes", ErrInvalidResponse, maximumDiscoveryResponseBytes)
	}
	page, executeErr := adapter.Decode(request.Plan, bytes.NewReader(payload))
	if executeErr != nil {
		return Result{}, executeErr
	}
	if len(page.Models) > request.Plan.PageSize || page.HasMore && page.NextCursor == "" ||
		!page.HasMore && page.NextCursor != "" {
		return Result{}, fmt.Errorf("%w: adapter pagination contract is invalid", ErrInvalidResponse)
	}
	now := time.Now().UTC()
	if executor.Now != nil {
		now = executor.Now().UTC()
	}
	models, revision, expiresAt, executeErr := executor.Claims.Issue(
		request.Plan, request.AuthorityDigest, credentialVersion, page.Models, now, executor.ClaimTTL,
	)
	if executeErr != nil {
		return Result{}, executeErr
	}
	return Result{
		Models: models, NextCursor: page.NextCursor, HasMore: page.HasMore,
		CatalogRevision:   request.Plan.CatalogRevision,
		DiscoveryRevision: revision, ExpiresAt: expiresAt,
	}, nil
}

func (executor Executor) resolveCredential(
	ctx context.Context,
	request ExecuteRequest,
) (backendinvoker.Credential, string, error) {
	plan := request.Plan
	if plan.CredentialID == "" {
		return backendinvoker.Credential{}, "", nil
	}
	if executor.CredentialMetadata == nil || executor.Credentials == nil {
		return backendinvoker.Credential{}, "", fmt.Errorf("%w: credential services are unavailable", ErrCredentialMismatch)
	}
	metadata, resolveCredentialErr := executor.CredentialMetadata.GetProviderCredential(
		ctx, accesscontrol.NamespaceID(plan.NamespaceID), plan.CredentialID,
	)
	if resolveCredentialErr != nil {
		return backendinvoker.Credential{}, "", fmt.Errorf("%w: load credential metadata", ErrCredentialMismatch)
	}
	if err := metadata.Validate(); err != nil || metadata.NamespaceID != plan.NamespaceID ||
		metadata.Status != providercredential.StatusActive || metadata.ProviderID != plan.ProviderID ||
		metadata.CredentialMode != providercredential.Mode(plan.CredentialMode) ||
		metadata.CredentialAdapterID != plan.CredentialAdapterID ||
		metadata.NormalizedOrigin != plan.NormalizedOrigin {
		return backendinvoker.Credential{}, "", ErrCredentialMismatch
	}
	version, resolveCredentialErr := executor.Credentials.Pin(ctx, plan.CredentialID, plan.ProviderID, plan.NormalizedOrigin)
	if resolveCredentialErr != nil || strings.TrimSpace(version) == "" {
		return backendinvoker.Credential{}, "", fmt.Errorf("%w: pin active credential version", ErrCredentialMismatch)
	}
	credential, resolveCredentialErr := executor.Credentials.ResolvePinned(
		ctx, plan.CredentialID, version, plan.ProviderID, plan.NormalizedOrigin,
	)
	if resolveCredentialErr != nil || credential.Version != version {
		return backendinvoker.Credential{}, "", fmt.Errorf("%w: resolve pinned credential version", ErrCredentialMismatch)
	}
	return credential, version, nil
}

func discoveryURL(origin, path string, query url.Values) (string, error) {
	base, err := url.Parse(origin)
	if err != nil || base.Scheme == "" || base.Host == "" || base.RawQuery != "" || base.Fragment != "" {
		return "", fmt.Errorf("%w: discovery origin is invalid", ErrInvalidRequest)
	}
	if !strings.HasPrefix(path, "/") || strings.ContainsAny(path, "?#\\\r\n") {
		return "", fmt.Errorf("%w: discovery path is invalid", ErrInvalidRequest)
	}
	base.Path = strings.TrimRight(base.Path, "/") + path
	base.RawPath = ""
	base.RawQuery = query.Encode()
	return base.String(), nil
}

var forbiddenCredentialHeaders = map[string]struct{}{
	"connection": {}, "content-length": {}, "cookie": {}, "host": {},
	"proxy-authorization": {}, "set-cookie": {}, "te": {}, "trailer": {},
	"transfer-encoding": {}, "upgrade": {},
}

func applyCredential(headers http.Header, credential backendinvoker.Credential) ([]string, error) {
	if credential.Header == "" && credential.Secret == "" && credential.Version == "" {
		return nil, nil
	}
	name := http.CanonicalHeaderKey(credential.Header)
	if name == "" || name != credential.Header || strings.TrimSpace(credential.Secret) == "" ||
		strings.ContainsAny(credential.Prefix, "\r\n") {
		return nil, ErrCredentialMismatch
	}
	if _, forbidden := forbiddenCredentialHeaders[strings.ToLower(name)]; forbidden || headers.Values(name) != nil {
		return nil, ErrCredentialMismatch
	}
	sensitive := make([]string, 0, len(credential.Extra)+1)
	sensitive = append(sensitive, name)
	for extraName, values := range credential.Extra {
		canonical := http.CanonicalHeaderKey(extraName)
		if canonical == "" || canonical != extraName || len(values) == 0 {
			return nil, ErrCredentialMismatch
		}
		if _, forbidden := forbiddenCredentialHeaders[strings.ToLower(canonical)]; forbidden || headers.Values(canonical) != nil {
			return nil, ErrCredentialMismatch
		}
		for _, value := range values {
			if strings.ContainsAny(value, "\r\n") {
				return nil, ErrCredentialMismatch
			}
		}
		sensitive = append(sensitive, canonical)
	}
	headers.Set(name, credential.Prefix+credential.Secret)
	for extraName, values := range credential.Extra {
		for _, value := range values {
			headers.Add(extraName, value)
		}
	}
	return sensitive, nil
}

func scrubHeaders(headers http.Header, names []string) {
	for _, name := range names {
		headers.Del(name)
	}
}
