package agentmanagement

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
)

type ToolClass string

const (
	ToolRead    ToolClass = "read"
	ToolWrite   ToolClass = "write"
	ToolExecute ToolClass = "execute"
)

type ToolIdempotency string

const (
	ToolNotIdempotent        ToolIdempotency = "none"
	ToolInvocationIdempotent ToolIdempotency = "invocation"
)

// ToolDefinition is safe to expose to an Agent. It contains schemas and
// authorization metadata, never an implementation or credential.
type ToolDefinition struct {
	Name                string                     `json:"name"`
	Description         string                     `json:"description"`
	InputSchema         json.RawMessage            `json:"inputSchema"`
	OutputSchema        json.RawMessage            `json:"outputSchema"`
	RequiredPermissions []accesscontrol.Permission `json:"requiredPermissions"`
	Class               ToolClass                  `json:"class"`
	Idempotency         ToolIdempotency            `json:"idempotency"`
	TimeoutMilliseconds int64                      `json:"timeoutMilliseconds"`
}

type ToolInvocationContext struct {
	NamespaceID      string
	PrincipalID      string
	SessionID        string
	TurnID           string
	InvocationID     string
	AuthorityDigest  string
	RegistryRevision string
	Target           Target
	Origin           ToolOrigin
}

type ToolResult struct {
	Value      json.RawMessage `json:"value,omitempty"`
	ArtifactID string          `json:"artifactId,omitempty"`
}

type ToolHandler interface {
	Invoke(context.Context, ToolInvocationContext, json.RawMessage) (ToolResult, error)
}

// ToolInputScrubber is implemented by transports that inject a plaintext
// credential. It removes exact transport-owned secrets before the Router
// computes an invocation digest or writes the input to durable state.
type ToolInputScrubber interface {
	ScrubInput(context.Context, json.RawMessage) (json.RawMessage, error)
}

type ToolHandlerFunc func(context.Context, ToolInvocationContext, json.RawMessage) (ToolResult, error)

func (function ToolHandlerFunc) Invoke(
	ctx context.Context, invocation ToolInvocationContext, input json.RawMessage,
) (ToolResult, error) {
	return function(ctx, invocation, input)
}

type RegisteredTool struct {
	Definition ToolDefinition
	Handler    ToolHandler
	Origin     ToolOrigin
}

type ToolOriginKind string

const (
	ToolOriginRouter ToolOriginKind = "router"
	ToolOriginRemote ToolOriginKind = "remote"
)

// ToolOrigin is registry-only trust metadata. It participates in the pinned
// registry digest but is not model-supplied input and cannot be changed by a
// tool result.
type ToolOrigin struct {
	Kind                    ToolOriginKind `json:"kind"`
	SourceID                string         `json:"sourceId,omitempty"`
	SourceRevision          int64          `json:"sourceRevision,omitempty"`
	DiscoveryDigest         string         `json:"discoveryDigest,omitempty"`
	ApprovedDiscoveryDigest string         `json:"approvedDiscoveryDigest,omitempty"`
	CredentialVersionID     string         `json:"credentialVersionId,omitempty"`
}

type ToolAuthorizer interface {
	AuthorizeTool(context.Context, ToolInvocationContext, ToolDefinition) (ToolInvocationContext, error)
}

// ToolRegistry is immutable. A turn holds one pointer and one content digest,
// so later integration discovery cannot change the tools mid-plan.
type ToolRegistry struct {
	revision   string
	ordered    []ToolDefinition
	byName     map[string]registeredTool
	authorizer ToolAuthorizer
}

type registeredTool struct {
	RegisteredTool
	input  *compiledSchema
	output *compiledSchema
}

func NewToolRegistry(tools []RegisteredTool, authorizer ToolAuthorizer) (*ToolRegistry, error) {
	if len(tools) == 0 || authorizer == nil {
		return nil, fmt.Errorf("%w: Tool Registry requires tools and an authorizer", ErrInvalid)
	}
	copyTools := append([]RegisteredTool(nil), tools...)
	sort.Slice(copyTools, func(i, j int) bool {
		return copyTools[i].Definition.Name < copyTools[j].Definition.Name
	})
	registry := &ToolRegistry{
		ordered: make([]ToolDefinition, 0, len(copyTools)),
		byName:  make(map[string]registeredTool, len(copyTools)), authorizer: authorizer,
	}
	type digestEntry struct {
		Definition ToolDefinition `json:"definition"`
		Origin     ToolOrigin     `json:"origin"`
	}
	digestEntries := make([]digestEntry, 0, len(copyTools))
	for _, candidate := range copyTools {
		definition, inputSchema, outputSchema, err := canonicalizeToolDefinition(candidate.Definition)
		if err != nil {
			return nil, err
		}
		if candidate.Handler == nil {
			return nil, fmt.Errorf("%w: tool %q has no handler", ErrInvalid, candidate.Definition.Name)
		}
		origin, err := canonicalToolOrigin(candidate.Origin, definition.Name)
		if err != nil {
			return nil, err
		}
		candidate.Origin = origin
		if _, duplicate := registry.byName[definition.Name]; duplicate {
			return nil, fmt.Errorf("%w: duplicate tool %q", ErrInvalid, definition.Name)
		}
		candidate.Definition = definition
		registry.byName[definition.Name] = registeredTool{
			RegisteredTool: candidate, input: inputSchema, output: outputSchema,
		}
		registry.ordered = append(registry.ordered, cloneToolDefinition(definition))
		digestEntries = append(digestEntries, digestEntry{Definition: definition, Origin: origin})
	}
	canonical, err := json.Marshal(digestEntries)
	if err != nil {
		return nil, fmt.Errorf("%w: encode Tool Registry", ErrInvalid)
	}
	digest := sha256.Sum256(canonical)
	registry.revision = "sha256:" + hex.EncodeToString(digest[:])
	return registry, nil
}

func (definition ToolDefinition) Validate() error {
	if !canonicalToolName(definition.Name) || strings.TrimSpace(definition.Description) == "" ||
		len(definition.Description) > 1024 || len(definition.RequiredPermissions) == 0 ||
		definition.TimeoutMilliseconds < 100 || definition.TimeoutMilliseconds > int64((10*time.Minute)/time.Millisecond) {
		return fmt.Errorf("%w: invalid tool definition %q", ErrInvalid, definition.Name)
	}
	if definition.Class != ToolRead && definition.Class != ToolWrite && definition.Class != ToolExecute {
		return fmt.Errorf("%w: invalid tool class for %q", ErrInvalid, definition.Name)
	}
	if definition.Idempotency != ToolNotIdempotent && definition.Idempotency != ToolInvocationIdempotent {
		return fmt.Errorf("%w: invalid tool idempotency for %q", ErrInvalid, definition.Name)
	}
	for _, permission := range definition.RequiredPermissions {
		if !permission.Valid() {
			return fmt.Errorf("%w: tool %q contains an unknown permission", ErrInvalid, definition.Name)
		}
	}
	return nil
}

func canonicalizeToolDefinition(
	definition ToolDefinition,
) (ToolDefinition, *compiledSchema, *compiledSchema, error) {
	definition = cloneToolDefinition(definition)
	definition.Name = strings.TrimSpace(definition.Name)
	definition.Description = sanitizeToolText(definition.Description)
	sort.Slice(definition.RequiredPermissions, func(i, j int) bool {
		return definition.RequiredPermissions[i] < definition.RequiredPermissions[j]
	})
	permissions := definition.RequiredPermissions[:0]
	for _, permission := range definition.RequiredPermissions {
		if len(permissions) == 0 || permissions[len(permissions)-1] != permission {
			permissions = append(permissions, permission)
		}
	}
	definition.RequiredPermissions = permissions
	if err := definition.Validate(); err != nil {
		return ToolDefinition{}, nil, nil, err
	}
	input, canonicalInput, err := compileToolSchema(definition.InputSchema)
	if err != nil {
		return ToolDefinition{}, nil, nil, fmt.Errorf("tool %q input schema: %w", definition.Name, err)
	}
	output, canonicalOutput, err := compileToolSchema(definition.OutputSchema)
	if err != nil {
		return ToolDefinition{}, nil, nil, fmt.Errorf("tool %q output schema: %w", definition.Name, err)
	}
	definition.InputSchema = canonicalInput
	definition.OutputSchema = canonicalOutput
	return definition, input, output, nil
}

// CanonicalizeToolDefinition compiles the complete input/output schema using
// the same bounded offline validator used at invocation time. Discovery uses
// it before a definition is persisted or presented for approval.
func CanonicalizeToolDefinition(definition ToolDefinition) (ToolDefinition, error) {
	canonical, _, _, err := canonicalizeToolDefinition(definition)
	return canonical, err
}

func canonicalToolOrigin(origin ToolOrigin, toolName string) (ToolOrigin, error) {
	if origin.Kind == "" {
		origin.Kind = ToolOriginRouter
	}
	if origin.Kind == ToolOriginRouter {
		if origin.SourceID != "" || origin.SourceRevision != 0 || origin.DiscoveryDigest != "" || origin.ApprovedDiscoveryDigest != "" ||
			origin.CredentialVersionID != "" {
			return ToolOrigin{}, fmt.Errorf("%w: Router tool %q has remote trust metadata", ErrInvalid, toolName)
		}
		return origin, nil
	}
	if origin.Kind != ToolOriginRemote || uuid.Validate(origin.SourceID) != nil ||
		origin.SourceRevision < 1 ||
		!validSHA256Digest(origin.DiscoveryDigest) || origin.DiscoveryDigest != origin.ApprovedDiscoveryDigest ||
		(origin.CredentialVersionID != "" && uuid.Validate(origin.CredentialVersionID) != nil) ||
		!strings.HasPrefix(toolName, "remote."+origin.SourceID+".") {
		return ToolOrigin{}, fmt.Errorf("%w: remote tool %q is not source-qualified and approved", ErrDenied, toolName)
	}
	return origin, nil
}

func sanitizeToolText(value string) string {
	value = strings.Map(func(character rune) rune {
		if character < 0x20 || character == 0x7f {
			return ' '
		}
		return character
	}, value)
	return strings.TrimSpace(value)
}

func validSHA256Digest(value string) bool {
	if len(value) != len("sha256:")+sha256.Size*2 || !strings.HasPrefix(value, "sha256:") {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, "sha256:"))
	return err == nil
}

func (registry *ToolRegistry) Revision() string {
	if registry == nil {
		return ""
	}
	return registry.revision
}

func (registry *ToolRegistry) Manifest(createdAt, expiresAt time.Time) RegistryManifest {
	if registry == nil {
		return RegistryManifest{}
	}
	tools := make([]ToolManifest, 0, len(registry.ordered))
	for _, definition := range registry.ordered {
		registered := registry.byName[definition.Name]
		tools = append(tools, ToolManifest{
			Definition: cloneToolDefinition(definition), Origin: registered.Origin,
		})
	}
	return RegistryManifest{
		Revision: registry.revision, Tools: tools,
		CreatedAt: createdAt.UTC(), ExpiresAt: expiresAt.UTC(),
	}
}

func (registry *ToolRegistry) Definitions(policy ToolPolicy) []ToolDefinition {
	if registry == nil {
		return nil
	}
	result := make([]ToolDefinition, 0, len(registry.ordered))
	for _, definition := range registry.ordered {
		registered := registry.byName[definition.Name]
		if policy.allowsRegistered(registered) {
			result = append(result, cloneToolDefinition(definition))
		}
	}
	return result
}

// AllDefinitions returns the immutable Management catalog. Profile policy is
// applied only when a session is created/executed; an authorized operator must
// be able to discover exact remote mutating names before granting them.
func (registry *ToolRegistry) AllDefinitions() []ToolDefinition {
	if registry == nil {
		return nil
	}
	result := make([]ToolDefinition, len(registry.ordered))
	for index := range registry.ordered {
		result[index] = cloneToolDefinition(registry.ordered[index])
	}
	return result
}

func (registry *ToolRegistry) Origin(name string) (ToolOrigin, bool) {
	if registry == nil {
		return ToolOrigin{}, false
	}
	registered, found := registry.byName[name]
	return registered.Origin, found
}

// Definition returns the exact immutable definition visible through policy.
// Runtime orchestration uses this instead of reconstructing execution metadata
// from model-supplied tool-call fields.
func (registry *ToolRegistry) Definition(name string, policy ToolPolicy) (ToolDefinition, ToolOrigin, bool) {
	if registry == nil {
		return ToolDefinition{}, ToolOrigin{}, false
	}
	registered, found := registry.byName[name]
	if !found || !policy.allowsRegistered(registered) {
		return ToolDefinition{}, ToolOrigin{}, false
	}
	return cloneToolDefinition(registered.Definition), registered.Origin, true
}

// ScrubInvocationInput runs the immutable handler's exact-secret boundary
// before BeginInvocation. Invocation authorization still occurs in Invoke;
// this method only guarantees that an unauthorized or failed call cannot put
// a transport credential into its durable input row first.
func (registry *ToolRegistry) ScrubInvocationInput(
	ctx context.Context,
	pinnedRevision string,
	policy ToolPolicy,
	name string,
	input json.RawMessage,
) (json.RawMessage, error) {
	if registry == nil || pinnedRevision == "" || pinnedRevision != registry.revision {
		return nil, fmt.Errorf("%w: Tool Registry revision changed", ErrConflict)
	}
	registered, found := registry.byName[name]
	if !found || !policy.allowsRegistered(registered) {
		return nil, ErrToolUnavailable
	}
	if err := registered.input.validateRaw(input, 1<<20); err != nil {
		return nil, err
	}
	clean := append(json.RawMessage(nil), input...)
	if scrubber, ok := registered.Handler.(ToolInputScrubber); ok {
		var err error
		clean, err = scrubber.ScrubInput(ctx, clean)
		if err != nil {
			return nil, err
		}
	} else {
		var err error
		clean, err = sanitizeTranscriptObject(clean, maximumInlineToolResultBytes)
		if err != nil {
			return nil, err
		}
	}
	if err := registered.input.validateRaw(clean, 1<<20); err != nil {
		return nil, err
	}
	return clean, nil
}

func (registry *ToolRegistry) Invoke(
	ctx context.Context,
	pinnedRevision string,
	policy ToolPolicy,
	invocation ToolInvocationContext,
	name string,
	input json.RawMessage,
) (ToolResult, error) {
	if registry == nil || pinnedRevision == "" || pinnedRevision != registry.revision {
		return ToolResult{}, fmt.Errorf("%w: Tool Registry revision changed", ErrConflict)
	}
	registered, found := registry.byName[name]
	if !found || !policy.allowsRegistered(registered) {
		return ToolResult{}, ErrToolUnavailable
	}
	if err := registered.input.validateRaw(input, 1<<20); err != nil {
		return ToolResult{}, err
	}
	invocation.RegistryRevision = registry.revision
	invocation.Origin = registered.Origin
	authorized, err := registry.authorizer.AuthorizeTool(ctx, invocation, registered.Definition)
	if err != nil {
		return ToolResult{}, err
	}
	if authorized.NamespaceID != invocation.NamespaceID || authorized.PrincipalID != invocation.PrincipalID ||
		authorized.SessionID != invocation.SessionID || authorized.TurnID != invocation.TurnID ||
		authorized.InvocationID != invocation.InvocationID || authorized.Target != invocation.Target ||
		authorized.RegistryRevision != invocation.RegistryRevision || authorized.Origin != invocation.Origin ||
		authorized.AuthorityDigest == "" {
		return ToolResult{}, fmt.Errorf("%w: Tool authorization context is invalid", ErrDenied)
	}
	timeout := time.Duration(registered.Definition.TimeoutMilliseconds) * time.Millisecond
	callContext, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	result, err := registered.Handler.Invoke(callContext, authorized, append(json.RawMessage(nil), input...))
	if err != nil {
		return ToolResult{}, err
	}
	// Durable invocation rows and transcript events share one 64 KiB safe
	// representation. Larger output must first be persisted as an Artifact and
	// returned as a bounded JSON reference; it may never leave a side-effecting
	// invocation stuck between execution and event commit.
	if len(result.Value) == 0 || len(result.Value) > maximumInlineToolResultBytes ||
		(result.ArtifactID != "" && uuid.Validate(result.ArtifactID) != nil) {
		return ToolResult{}, fmt.Errorf("tool %q returned an invalid bounded result", name)
	}
	if err := registered.output.validateRaw(result.Value, 1<<20); err != nil {
		return ToolResult{}, fmt.Errorf("tool %q output: %w", name, err)
	}
	return result, nil
}

func (policy ToolPolicy) allowsRegistered(tool registeredTool) bool {
	if !policy.Allows(tool.Definition.Name) {
		return false
	}
	if tool.Origin.Kind != ToolOriginRemote || tool.Definition.Class == ToolRead {
		return true
	}
	// A wildcard never grants a newly discovered remote mutating tool. The
	// exact source-qualified name must be present in the Profile revision.
	for _, allowed := range policy.Allow {
		if allowed == tool.Definition.Name {
			return true
		}
	}
	return false
}

func (policy ToolPolicy) Allows(name string) bool {
	allowed := false
	for _, pattern := range policy.Allow {
		if toolPatternMatches(pattern, name) {
			allowed = true
			break
		}
	}
	if !allowed {
		return false
	}
	for _, pattern := range policy.Deny {
		if toolPatternMatches(pattern, name) {
			return false
		}
	}
	return true
}

func toolPatternMatches(pattern, name string) bool {
	if pattern == name {
		return true
	}
	return strings.HasSuffix(pattern, ".*") && strings.HasPrefix(name, strings.TrimSuffix(pattern, "*"))
}

func cloneToolDefinition(source ToolDefinition) ToolDefinition {
	result := source
	result.InputSchema = append(json.RawMessage(nil), source.InputSchema...)
	result.OutputSchema = append(json.RawMessage(nil), source.OutputSchema...)
	result.RequiredPermissions = append([]accesscontrol.Permission(nil), source.RequiredPermissions...)
	return result
}

func canonicalToolName(value string) bool {
	if len(value) < 3 || len(value) > 128 || value[0] < 'a' || value[0] > 'z' {
		return false
	}
	for _, character := range value[1:] {
		if (character < 'a' || character > 'z') && (character < '0' || character > '9') &&
			character != '_' && character != '-' && character != '.' {
			return false
		}
	}
	return true
}
