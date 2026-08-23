package agenttoolsource

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	mcpclient "github.com/mark3labs/mcp-go/client"
	mcptransport "github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendegress"
)

const (
	maximumDiscoveredTools = 256
	maximumRemoteResponse  = 2 << 20
	defaultRemoteTimeout   = 30 * time.Second
)

var (
	ErrDiscoveryFailed  = errors.New("remote Tool Source discovery failed")
	ErrInvocationFailed = errors.New("remote tool invocation failed")
)

type PinnedCredential struct {
	VersionID string
	Secret    []byte
}

type CredentialResolver interface {
	Resolve(context.Context, string, string, string) (PinnedCredential, error)
}

type ClientFactoryOptions struct {
	OperatorGuard backendegress.Guard
	Resolver      backendegress.Resolver
	Credentials   CredentialResolver
	Timeout       time.Duration
}

// ClientFactory creates one bounded, redirect-free streamable HTTP client per
// discovery or invocation. Source policy and immutable operator policy are
// independent guards; both must resolve to the same pinned dial target.
type ClientFactory struct {
	operatorGuard backendegress.Guard
	resolver      backendegress.Resolver
	credentials   CredentialResolver
	timeout       time.Duration
}

func NewClientFactory(options ClientFactoryOptions) (*ClientFactory, error) {
	if options.Credentials == nil {
		return nil, fmt.Errorf("remote Tool Source credential resolver is required")
	}
	timeout := options.Timeout
	if timeout == 0 {
		timeout = defaultRemoteTimeout
	}
	if timeout < time.Second || timeout > 10*time.Minute {
		return nil, fmt.Errorf("remote Tool Source timeout is invalid")
	}
	return &ClientFactory{
		operatorGuard: options.OperatorGuard, resolver: options.Resolver,
		credentials: options.Credentials, timeout: timeout,
	}, nil
}

// Discover returns source-qualified, fully compiled definitions and the exact
// credential version used. Descriptions and schemas remain untrusted data;
// callers persist an immutable digest and require separate operator approval.
func (factory *ClientFactory) Discover(
	ctx context.Context, source agentmanagement.ToolSource,
) ([]agentmanagement.ToolDefinition, string, error) {
	client, credential, closeClient, err := factory.open(ctx, source, "")
	if err != nil {
		return nil, "", ErrDiscoveryFailed
	}
	defer closeClient()
	result, err := client.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil || result == nil || len(result.Tools) > maximumDiscoveredTools {
		return nil, "", ErrDiscoveryFailed
	}
	definitions := make([]agentmanagement.ToolDefinition, 0, len(result.Tools))
	seen := make(map[string]struct{}, len(result.Tools))
	for _, remote := range result.Tools {
		definition, convertErr := remoteDefinition(source.ID, remote)
		if convertErr != nil {
			return nil, "", ErrDiscoveryFailed
		}
		if _, duplicate := seen[definition.Name]; duplicate {
			return nil, "", ErrDiscoveryFailed
		}
		seen[definition.Name] = struct{}{}
		definitions = append(definitions, definition)
	}
	return definitions, credential.VersionID, nil
}

func (factory *ClientFactory) Handler(
	source agentmanagement.ToolSource, credentialVersionID, upstreamName string,
) agentmanagement.ToolHandler {
	return &remoteToolHandler{
		factory: factory, source: source,
		credentialVersionID: credentialVersionID, upstreamName: upstreamName,
	}
}

type remoteToolHandler struct {
	factory             *ClientFactory
	source              agentmanagement.ToolSource
	credentialVersionID string
	upstreamName        string
}

func (handler *remoteToolHandler) ScrubInput(
	ctx context.Context, input json.RawMessage,
) (json.RawMessage, error) {
	credential, err := handler.factory.resolveCredential(
		ctx, handler.source, handler.credentialVersionID,
	)
	if err != nil {
		return nil, ErrInvocationFailed
	}
	defer clear(credential.Secret)
	return scrubRemoteToolPayload(input, credential.Secret)
}

func (handler *remoteToolHandler) Invoke(
	ctx context.Context, _ agentmanagement.ToolInvocationContext, input json.RawMessage,
) (agentmanagement.ToolResult, error) {
	client, credential, closeClient, invokeErr := handler.factory.open(
		ctx, handler.source, handler.credentialVersionID,
	)
	if invokeErr != nil {
		return agentmanagement.ToolResult{}, ErrInvocationFailed
	}
	defer closeClient()
	input, invokeErr = scrubRemoteToolPayload(input, credential.Secret)
	if invokeErr != nil {
		return agentmanagement.ToolResult{}, ErrInvocationFailed
	}
	var arguments map[string]any
	if err := json.Unmarshal(input, &arguments); err != nil {
		return agentmanagement.ToolResult{}, agentmanagement.ErrInvalid
	}
	result, invokeErr := client.CallTool(ctx, mcp.CallToolRequest{Params: mcp.CallToolParams{
		Name: handler.upstreamName, Arguments: arguments,
	}})
	if invokeErr != nil || result == nil || result.IsError {
		return agentmanagement.ToolResult{}, ErrInvocationFailed
	}
	value, invokeErr := boundedResult(result)
	if invokeErr != nil {
		return agentmanagement.ToolResult{}, ErrInvocationFailed
	}
	value, invokeErr = scrubRemoteToolPayload(value, credential.Secret)
	if invokeErr != nil {
		return agentmanagement.ToolResult{}, invokeErr
	}
	return agentmanagement.ToolResult{Value: value}, nil
}

func scrubRemoteToolPayload(value json.RawMessage, secret []byte) (json.RawMessage, error) {
	clean, err := agentmanagement.ScrubToolSecrets(value, secret)
	if err != nil {
		return nil, ErrInvocationFailed
	}
	return clean, nil
}

func (factory *ClientFactory) open(
	ctx context.Context, source agentmanagement.ToolSource, credentialVersionID string,
) (*mcpclient.Client, PinnedCredential, func(), error) {
	if factory == nil || source.Status != agentmanagement.StatusActive || source.Transport != "streamable_http" {
		return nil, PinnedCredential{}, nil, agentmanagement.ErrToolUnavailable
	}
	policy, err := (PolicyCompiler{}).Compile(source.EgressPolicy)
	if err != nil {
		return nil, PinnedCredential{}, nil, err
	}
	sourceGuard := backendegress.Guard{Policy: policy, Resolver: factory.resolver}
	transport, err := backendegress.NewTransport(backendegress.TransportOptions{
		Guard: sourceGuard, AdditionalGuards: []backendegress.Guard{factory.operatorGuard},
		DialTimeout: factory.timeout,
	})
	if err != nil {
		return nil, PinnedCredential{}, nil, err
	}
	credential, err := factory.resolveCredential(ctx, source, credentialVersionID)
	if err != nil {
		transport.CloseIdleConnections()
		return nil, PinnedCredential{}, nil, err
	}
	roundTripper := http.RoundTripper(transport)
	if len(credential.Secret) > 0 {
		roundTripper = &bearerRoundTripper{next: roundTripper, secret: credential.Secret}
	}
	roundTripper = &boundedRoundTripper{next: roundTripper, maximum: maximumRemoteResponse}
	httpClient := backendegress.NewHTTPClient(roundTripper, true)
	httpClient.Timeout = factory.timeout
	client, err := mcpclient.NewStreamableHttpClient(
		source.Endpoint,
		mcptransport.WithHTTPBasicClient(httpClient),
		mcptransport.WithHTTPTimeout(factory.timeout),
	)
	if err != nil {
		clear(credential.Secret)
		transport.CloseIdleConnections()
		return nil, PinnedCredential{}, nil, err
	}
	closeClient := func() {
		_ = client.Close()
		clear(credential.Secret)
		transport.CloseIdleConnections()
	}
	if err := client.Start(ctx); err != nil {
		closeClient()
		return nil, PinnedCredential{}, nil, err
	}
	initialize := mcp.InitializeRequest{}
	initialize.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initialize.Params.ClientInfo = mcp.Implementation{Name: "vllm-semantic-router", Version: "v0.4"}
	if _, err := client.Initialize(ctx, initialize); err != nil {
		closeClient()
		return nil, PinnedCredential{}, nil, err
	}
	return client, credential, closeClient, nil
}

func (factory *ClientFactory) resolveCredential(
	ctx context.Context, source agentmanagement.ToolSource, credentialVersionID string,
) (PinnedCredential, error) {
	if factory == nil || factory.credentials == nil {
		return PinnedCredential{}, agentmanagement.ErrToolUnavailable
	}
	if source.CredentialID == "" {
		if credentialVersionID != "" {
			return PinnedCredential{}, agentmanagement.ErrToolUnavailable
		}
		return PinnedCredential{}, nil
	}
	credential, err := factory.credentials.Resolve(
		ctx, source.NamespaceID, source.CredentialID, credentialVersionID,
	)
	if err != nil || len(credential.Secret) == 0 ||
		(credentialVersionID != "" && credential.VersionID != credentialVersionID) {
		clear(credential.Secret)
		return PinnedCredential{}, agentmanagement.ErrToolUnavailable
	}
	return credential, nil
}

func remoteDefinition(sourceID string, remote mcp.Tool) (agentmanagement.ToolDefinition, error) {
	if strings.TrimSpace(remote.Name) != remote.Name || remote.Name == "" {
		return agentmanagement.ToolDefinition{}, agentmanagement.ErrInvalid
	}
	input, err := remoteSchema(remote.RawInputSchema, remote.InputSchema)
	if err != nil {
		return agentmanagement.ToolDefinition{}, err
	}
	output, err := remoteSchema(remote.RawOutputSchema, remote.OutputSchema)
	if err != nil {
		return agentmanagement.ToolDefinition{}, err
	}
	if len(remote.RawOutputSchema) == 0 && remote.OutputSchema.Type == "" {
		output = json.RawMessage(`{"type":"object","additionalProperties":true}`)
	}
	class := agentmanagement.ToolExecute
	if remote.Annotations.ReadOnlyHint != nil && *remote.Annotations.ReadOnlyHint {
		class = agentmanagement.ToolRead
	} else if remote.Annotations.OpenWorldHint != nil && !*remote.Annotations.OpenWorldHint {
		class = agentmanagement.ToolWrite
	}
	idempotency := agentmanagement.ToolNotIdempotent
	if remote.Annotations.IdempotentHint != nil && *remote.Annotations.IdempotentHint {
		idempotency = agentmanagement.ToolInvocationIdempotent
	}
	definition := agentmanagement.ToolDefinition{
		Name: "remote." + sourceID + "." + remote.Name, Description: remote.Description,
		InputSchema: input, OutputSchema: output,
		RequiredPermissions: []accesscontrol.Permission{accesscontrol.PermissionToolInvoke},
		Class:               class, Idempotency: idempotency,
		TimeoutMilliseconds: int64(defaultRemoteTimeout / time.Millisecond),
	}
	return agentmanagement.CanonicalizeToolDefinition(definition)
}

func remoteSchema(raw json.RawMessage, structured any) (json.RawMessage, error) {
	if len(raw) > 0 {
		return append(json.RawMessage(nil), raw...), nil
	}
	encoded, err := json.Marshal(structured)
	if err != nil {
		return nil, err
	}
	return encoded, nil
}

func boundedResult(result *mcp.CallToolResult) (json.RawMessage, error) {
	var value any
	if result.StructuredContent != nil {
		value = result.StructuredContent
	} else {
		// This field name is an explicit trust label. The Agent prompt/runtime
		// treats its value as data and never as policy or approval instructions.
		value = map[string]any{"untrustedRemoteContent": result.Content}
	}
	encoded, err := json.Marshal(value)
	if err != nil || len(encoded) == 0 || len(encoded) > maximumRemoteResponse {
		return nil, ErrInvocationFailed
	}
	return encoded, nil
}

type bearerRoundTripper struct {
	next   http.RoundTripper
	secret []byte
}

func (transport *bearerRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	clone := request.Clone(request.Context())
	clone.Header = request.Header.Clone()
	clone.Header.Set("Authorization", "Bearer "+string(transport.secret))
	return transport.next.RoundTrip(clone)
}

type boundedRoundTripper struct {
	next    http.RoundTripper
	maximum int64
}

func (transport *boundedRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	response, err := transport.next.RoundTrip(request)
	if err != nil || response == nil || response.Body == nil {
		return response, err
	}
	if response.ContentLength > transport.maximum {
		_ = response.Body.Close()
		return nil, ErrInvocationFailed
	}
	response.Body = &boundedReadCloser{Reader: &hardLimitReader{
		reader: response.Body, remaining: transport.maximum,
	}, close: response.Body}
	return response, nil
}

type boundedReadCloser struct {
	io.Reader
	close io.Closer
}

func (reader *boundedReadCloser) Close() error { return reader.close.Close() }

type hardLimitReader struct {
	reader    io.Reader
	remaining int64
}

func (reader *hardLimitReader) Read(target []byte) (int, error) {
	if reader.remaining < 0 {
		return 0, ErrInvocationFailed
	}
	maximumRead := int64(len(target))
	if maximumRead > reader.remaining+1 {
		maximumRead = reader.remaining + 1
	}
	read, err := reader.reader.Read(target[:maximumRead])
	reader.remaining -= int64(read)
	if reader.remaining < 0 {
		return 0, ErrInvocationFailed
	}
	return read, err
}

var (
	_ agentmanagement.ToolSourceDiscoverer = (*ClientFactory)(nil)
	_ agentmanagement.ToolInputScrubber    = (*remoteToolHandler)(nil)
	_ agentmanagement.ToolHandler          = (*remoteToolHandler)(nil)
)
