package agentmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/securitykeyring"
)

type RegistrySource interface {
	Current(context.Context, string) (*ToolRegistry, error)
	Load(context.Context, string, string) (*ToolRegistry, error)
}

type ServiceOptions struct {
	Store            Store
	Notifier         TurnNotifier
	SessionAuthority SessionAuthority
	TargetVisibility TargetVisibility
	Definitions      DefinitionValidator
	SourcePolicies   ToolSourcePolicyValidator
	ToolSources      ToolSourceDiscoverer
	Registries       RegistrySource
	SecretCodec      SecretCodec
	CommandCodec     *managementcommand.Codec
	CursorKeyring    securitykeyring.Symmetric
	SessionTTL       time.Duration
	Now              func() time.Time
}

type Service struct {
	store            Store
	notifier         TurnNotifier
	sessionAuthority SessionAuthority
	targetVisibility TargetVisibility
	definitions      DefinitionValidator
	sourcePolicies   ToolSourcePolicyValidator
	toolSources      ToolSourceDiscoverer
	registries       RegistrySource
	secretCodec      SecretCodec
	commandCodec     *managementcommand.Codec
	codec            signedCodec
	sessionTTL       time.Duration
	now              func() time.Time
	closeOnce        sync.Once
}

const agentResourceCommandTTL = 7 * 24 * time.Hour

const (
	agentProfileResourceType        = "agent_profile"
	agentSkillResourceType          = "agent_skill"
	agentToolCredentialResourceType = "agent_tool_credential"
	agentToolSourceResourceType     = "agent_tool_source"
)

func NewService(options ServiceOptions) (*Service, error) {
	if options.Store == nil || options.SessionAuthority == nil || options.TargetVisibility == nil || options.Definitions == nil ||
		options.SourcePolicies == nil || options.ToolSources == nil ||
		options.Registries == nil || options.SecretCodec == nil || options.CommandCodec == nil {
		return nil, fmt.Errorf("%w: Agent service dependencies are incomplete", ErrInvalid)
	}
	codec, err := newSignedCodec(options.CursorKeyring)
	if err != nil {
		return nil, err
	}
	now := options.Now
	if now == nil {
		now = time.Now
	}
	sessionTTL := options.SessionTTL
	if sessionTTL == 0 {
		sessionTTL = 8 * time.Hour
	}
	if sessionTTL < 5*time.Minute || sessionTTL > 24*time.Hour {
		codec.close()
		return nil, fmt.Errorf("%w: Agent session TTL is outside the supported range", ErrInvalid)
	}
	return &Service{
		store: options.Store, notifier: options.Notifier, sessionAuthority: options.SessionAuthority,
		targetVisibility: options.TargetVisibility,
		definitions:      options.Definitions, sourcePolicies: options.SourcePolicies, toolSources: options.ToolSources,
		registries:  options.Registries,
		secretCodec: options.SecretCodec, commandCodec: options.CommandCodec,
		codec: codec, sessionTTL: sessionTTL, now: now,
	}, nil
}

func (service *Service) Close() {
	if service != nil {
		service.closeOnce.Do(func() { service.codec.close() })
	}
}

func (service *Service) Ready(ctx context.Context) error {
	if service == nil || service.store == nil || service.registries == nil {
		return errors.New("agent service is unavailable")
	}
	return service.store.Ready(ctx)
}

func (service *Service) ListProfiles(
	ctx context.Context, namespaceID string, request PageRequest, access AccessContext,
) (Page[Profile], error) {
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Page[Profile]{}, err
	}
	query, err := service.listQuery(namespaceID, "profiles", request)
	if err != nil {
		return Page[Profile]{}, err
	}
	result, err := service.store.ListProfiles(ctx, namespaceID, query)
	if err != nil {
		return Page[Profile]{}, err
	}
	for index := range result.Items {
		if err := service.filterProfileTarget(ctx, namespaceID, access.PrincipalID, &result.Items[index]); err != nil {
			return Page[Profile]{}, err
		}
	}
	return makePage(service, namespaceID, "profiles", query,
		request.PageSize, result.Items, result.HasMore, nil)
}

func (service *Service) GetProfile(
	ctx context.Context, namespaceID, id string, access AccessContext,
) (Profile, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil {
		return Profile{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Profile{}, err
	}
	profile, err := service.store.GetProfile(ctx, namespaceID, id)
	if err != nil {
		return Profile{}, err
	}
	if err := service.filterProfileTarget(ctx, namespaceID, access.PrincipalID, &profile); err != nil {
		return Profile{}, err
	}
	return profile, nil
}

func (service *Service) filterProfileTarget(
	ctx context.Context, namespaceID, principalID string, profile *Profile,
) error {
	if profile == nil || profile.DefaultTarget == nil {
		return nil
	}
	visible, err := service.targetVisibility.CanDiscover(ctx, namespaceID, principalID, *profile.DefaultTarget)
	if err != nil {
		return err
	}
	if !visible {
		// Do not disclose either the request-facing identifier or whether it
		// still exists. The Profile remains usable with an explicit target.
		profile.DefaultTarget = nil
	}
	return nil
}

func (service *Service) CreateProfile(
	ctx context.Context, namespaceID, idempotencyKey string, input ProfileInput, mutation MutationContext,
) (ResourceMutationResult, error) {
	input, err := NormalizeProfileInput(input)
	if err != nil || uuid.Validate(namespaceID) != nil || uuid.Validate(mutation.PrincipalID) != nil {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-profiles", idempotencyKey, input, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentProfileResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	registry, err := service.registries.Current(ctx, namespaceID)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if err := service.definitions.ValidateProfile(ctx, namespaceID, input, registry); err != nil {
		return ResourceMutationResult{}, err
	}
	if err := service.requireVisibleDefaultTarget(ctx, namespaceID, mutation.PrincipalID, input.DefaultTarget); err != nil {
		return ResourceMutationResult{}, err
	}
	return service.store.CreateProfile(ctx, namespaceID, uuid.NewString(), input, command)
}

func (service *Service) PatchProfile(
	ctx context.Context, namespaceID, id string, expected int64, patch ProfilePatch, mutation MutationContext,
) (Profile, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 || patchProfileEmpty(patch) {
		return Profile{}, ErrInvalid
	}
	current, err := service.store.GetProfile(ctx, namespaceID, id)
	if err != nil {
		return Profile{}, err
	}
	input, normalizedPatch, err := normalizeProfilePatch(current, patch)
	if err != nil {
		return Profile{}, err
	}
	registry, err := service.registries.Current(ctx, namespaceID)
	if err != nil {
		return Profile{}, err
	}
	if err := service.definitions.ValidateProfile(ctx, namespaceID, input, registry); err != nil {
		return Profile{}, err
	}
	if err := service.requireVisibleDefaultTarget(ctx, namespaceID, mutation.PrincipalID, input.DefaultTarget); err != nil {
		return Profile{}, err
	}
	return service.store.PatchProfile(ctx, namespaceID, id, expected, normalizedPatch, mutation)
}

func (service *Service) requireVisibleDefaultTarget(
	ctx context.Context, namespaceID, principalID string, target *Target,
) error {
	if target == nil {
		return nil
	}
	visible, err := service.targetVisibility.CanDiscover(ctx, namespaceID, principalID, *target)
	if err != nil {
		return err
	}
	if !visible {
		return ErrNotFound
	}
	return nil
}

func (service *Service) DeleteProfile(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	return service.store.DeleteProfile(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) ListSkills(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[Skill], error) {
	query, err := service.listQuery(namespaceID, "skills", request)
	if err != nil {
		return Page[Skill]{}, err
	}
	result, err := service.store.ListSkills(ctx, namespaceID, query)
	return makePage(service, namespaceID, "skills", query,
		request.PageSize, result.Items, result.HasMore, err)
}

func (service *Service) GetSkill(ctx context.Context, namespaceID, id string) (Skill, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil {
		return Skill{}, ErrInvalid
	}
	return service.store.GetSkill(ctx, namespaceID, id)
}

func (service *Service) CreateSkill(
	ctx context.Context, namespaceID, idempotencyKey string, input SkillInput, mutation MutationContext,
) (ResourceMutationResult, error) {
	input, err := NormalizeSkillInput(input)
	if err != nil || uuid.Validate(namespaceID) != nil || uuid.Validate(mutation.PrincipalID) != nil {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-skills", idempotencyKey, input, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentSkillResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	registry, err := service.registries.Current(ctx, namespaceID)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if err := service.definitions.ValidateSkill(ctx, namespaceID, input, registry); err != nil {
		return ResourceMutationResult{}, err
	}
	return service.store.CreateSkill(ctx, namespaceID, uuid.NewString(), input, command)
}

func (service *Service) PatchSkill(
	ctx context.Context, namespaceID, id string, expected int64, patch SkillPatch, mutation MutationContext,
) (Skill, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 || patchSkillEmpty(patch) {
		return Skill{}, ErrInvalid
	}
	current, err := service.store.GetSkill(ctx, namespaceID, id)
	if err != nil {
		return Skill{}, err
	}
	input, normalizedPatch, err := normalizeSkillPatch(current, patch)
	if err != nil {
		return Skill{}, err
	}
	registry, err := service.registries.Current(ctx, namespaceID)
	if err != nil {
		return Skill{}, err
	}
	if err := service.definitions.ValidateSkill(ctx, namespaceID, input, registry); err != nil {
		return Skill{}, err
	}
	return service.store.PatchSkill(ctx, namespaceID, id, expected, normalizedPatch, mutation)
}

func (service *Service) DeleteSkill(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	return service.store.DeleteSkill(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) ListTools(
	ctx context.Context, namespaceID string, request ToolPageRequest,
) (Page[ToolDefinition], string, error) {
	if uuid.Validate(namespaceID) != nil || request.PageSize < 1 || request.PageSize > 200 {
		return Page[ToolDefinition]{}, "", ErrInvalid
	}
	search, err := managementsearch.Normalize(request.Search)
	if err != nil {
		return Page[ToolDefinition]{}, "", ErrInvalid
	}
	registry, err := service.registries.Current(ctx, namespaceID)
	if err != nil {
		return Page[ToolDefinition]{}, "", err
	}
	afterName := ""
	if request.Cursor != "" {
		cursor, decodeErr := service.codec.decodeToolCursor(request.Cursor)
		if decodeErr != nil || cursor.NamespaceID != namespaceID || cursor.Search != search {
			return Page[ToolDefinition]{}, "", ErrInvalid
		}
		if cursor.RegistryRevision != registry.Revision() {
			return Page[ToolDefinition]{}, "", ErrConflict
		}
		afterName = cursor.AfterName
	}
	definitions := registry.AllDefinitions()
	items := make([]ToolDefinition, 0, min(request.PageSize+1, len(definitions)))
	for _, definition := range definitions {
		if definition.Name <= afterName || !toolMatchesSearch(definition, search) {
			continue
		}
		items = append(items, definition)
		if len(items) == request.PageSize+1 {
			break
		}
	}
	page := Page[ToolDefinition]{Items: items}
	if len(page.Items) > request.PageSize {
		page.Items = page.Items[:request.PageSize]
		page.HasMore = true
	}
	if page.HasMore {
		page.NextCursor, err = service.codec.encodeToolCursor(toolCursorPayload{
			NamespaceID: namespaceID, RegistryRevision: registry.Revision(),
			Search: search, AfterName: page.Items[len(page.Items)-1].Name,
		})
		if err != nil {
			return Page[ToolDefinition]{}, "", err
		}
	}
	return page, registry.Revision(), nil
}

func toolMatchesSearch(definition ToolDefinition, search string) bool {
	if search == "" {
		return true
	}
	return strings.HasPrefix(strings.ToLower(definition.Name), search) ||
		strings.HasPrefix(strings.ToLower(definition.Description), search)
}

func (service *Service) ListToolCredentials(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[ToolCredential], error) {
	query, err := service.listQuery(namespaceID, "tool-credentials", request)
	if err != nil {
		return Page[ToolCredential]{}, err
	}
	result, err := service.store.ListToolCredentials(ctx, namespaceID, query)
	return makePage(service, namespaceID, "tool-credentials", query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) GetToolCredential(
	ctx context.Context, namespaceID, id string,
) (ToolCredential, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil {
		return ToolCredential{}, ErrInvalid
	}
	return service.store.GetToolCredential(ctx, namespaceID, id)
}

func (service *Service) CreateToolCredential(
	ctx context.Context, namespaceID, idempotencyKey string, input ToolCredentialInput, mutation MutationContext,
) (ResourceMutationResult, error) {
	name := strings.TrimSpace(input.Name)
	if uuid.Validate(namespaceID) != nil || uuid.Validate(mutation.PrincipalID) != nil ||
		!validName(name) || len(input.Secret) < minimumToolCredentialBytes || len(input.Secret) > 64<<10 {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-tool-credentials", idempotencyKey,
		struct {
			Name   string `json:"name"`
			Secret []byte `json:"secret"`
		}{Name: name, Secret: input.Secret}, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentToolCredentialResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	plaintext := append([]byte(nil), input.Secret...)
	defer clear(plaintext)
	encrypted, err := service.secretCodec.Encrypt(ctx, plaintext)
	if err != nil {
		return ResourceMutationResult{}, ErrToolUnavailable
	}
	return service.store.CreateToolCredential(ctx, namespaceID, uuid.NewString(), name, encrypted, command)
}

func (service *Service) PatchToolCredential(
	ctx context.Context, namespaceID, id string, expected int64,
	patch ToolCredentialPatch, mutation MutationContext,
) (ToolCredential, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 ||
		(patch.Name == nil && patch.Status == nil) {
		return ToolCredential{}, ErrInvalid
	}
	return service.store.PatchToolCredential(ctx, namespaceID, id, expected, patch, mutation)
}

func (service *Service) RotateToolCredential(
	ctx context.Context, namespaceID, id, idempotencyKey string, expected int64, secret []byte,
	retirementGrace time.Duration, mutation MutationContext,
) (ResourceMutationResult, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 ||
		len(secret) < minimumToolCredentialBytes || len(secret) > 64<<10 ||
		retirementGrace < time.Minute || retirementGrace > 24*time.Hour {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-tool-credentials/"+id+":rotate", idempotencyKey,
		struct {
			ExpectedRevision int64  `json:"expectedRevision"`
			Secret           []byte `json:"secret"`
		}{ExpectedRevision: expected, Secret: secret}, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentToolCredentialResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	plaintext := append([]byte(nil), secret...)
	defer clear(plaintext)
	encrypted, err := service.secretCodec.Encrypt(ctx, plaintext)
	if err != nil {
		return ResourceMutationResult{}, ErrToolUnavailable
	}
	return service.store.RotateToolCredential(
		ctx, namespaceID, id, expected, encrypted, service.now().UTC().Add(retirementGrace), command,
	)
}

func (service *Service) DeleteToolCredential(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	return service.store.DeleteToolCredential(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) ListToolSources(
	ctx context.Context, namespaceID string, request PageRequest,
) (Page[ToolSource], error) {
	query, err := service.listQuery(namespaceID, "tool-sources", request)
	if err != nil {
		return Page[ToolSource]{}, err
	}
	result, err := service.store.ListToolSources(ctx, namespaceID, query)
	return makePage(service, namespaceID, "tool-sources", query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) GetToolSource(
	ctx context.Context, namespaceID, id string,
) (ToolSource, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil {
		return ToolSource{}, ErrInvalid
	}
	return service.store.GetToolSource(ctx, namespaceID, id)
}

func (service *Service) CreateToolSource(
	ctx context.Context, namespaceID, idempotencyKey string, input ToolSourceInput, mutation MutationContext,
) (ResourceMutationResult, error) {
	input, err := service.sourcePolicies.Normalize(input)
	if err != nil || uuid.Validate(namespaceID) != nil || uuid.Validate(mutation.PrincipalID) != nil {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-tool-sources", idempotencyKey, input, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentToolSourceResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	return service.store.CreateToolSource(ctx, namespaceID, uuid.NewString(), input, command)
}

func (service *Service) PatchToolSource(
	ctx context.Context, namespaceID, id string, expected int64,
	patch ToolSourcePatch, mutation MutationContext,
) (ToolSource, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 || toolSourcePatchEmpty(patch) {
		return ToolSource{}, ErrInvalid
	}
	current, err := service.store.GetToolSource(ctx, namespaceID, id)
	if err != nil {
		return ToolSource{}, err
	}
	input := ToolSourceInput{
		Name: current.Name, Description: current.Description, Kind: current.Kind,
		Transport: current.Transport, Endpoint: current.Endpoint, CredentialID: current.CredentialID,
		EgressPolicy: current.EgressPolicy,
	}
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.Transport != nil {
		input.Transport = *patch.Transport
	}
	if patch.Endpoint != nil {
		input.Endpoint = *patch.Endpoint
	}
	if patch.CredentialID.Present {
		input.CredentialID = ""
		if patch.CredentialID.Value != nil {
			input.CredentialID = *patch.CredentialID.Value
		}
	}
	if patch.EgressPolicy != nil {
		input.EgressPolicy = *patch.EgressPolicy
	}
	input, err = service.sourcePolicies.Normalize(input)
	if err != nil {
		return ToolSource{}, err
	}
	patch.Name, patch.Description = &input.Name, &input.Description
	patch.Transport, patch.Endpoint = &input.Transport, &input.Endpoint
	if patch.CredentialID.Present {
		patch.CredentialID.Value = &input.CredentialID
	}
	patch.EgressPolicy = &input.EgressPolicy
	return service.store.PatchToolSource(ctx, namespaceID, id, expected, patch, mutation)
}

func (service *Service) DeleteToolSource(
	ctx context.Context, namespaceID, id string, expected int64, mutation MutationContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	return service.store.DeleteToolSource(ctx, namespaceID, id, expected, mutation)
}

func (service *Service) TestToolSource(
	ctx context.Context, namespaceID, id, idempotencyKey string, mutation MutationContext,
) (ResourceMutationResult, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-tool-sources/"+id+":test", idempotencyKey,
		struct {
			SourceID string `json:"sourceId"`
		}{SourceID: id}, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentToolSourceResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	source, err := service.store.GetToolSource(ctx, namespaceID, id)
	if err != nil || source.Status != StatusActive {
		if err != nil {
			return ResourceMutationResult{}, err
		}
		return ResourceMutationResult{}, ErrConflict
	}
	definitions, _, err := service.toolSources.Discover(ctx, source)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	return service.store.UpdateToolSourceDiscovery(
		ctx, namespaceID, id, source.Revision, definitions, command,
	)
}

func (service *Service) ApproveToolSource(
	ctx context.Context, namespaceID, id, idempotencyKey string, expected int64,
	discoveryDigest string, mutation MutationContext,
) (ResourceMutationResult, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(id) != nil || expected < 1 ||
		!validSHA256Digest(discoveryDigest) {
		return ResourceMutationResult{}, ErrInvalid
	}
	command, err := service.bindResourceCommand(
		namespaceID, "/management/v1/agent-tool-sources/"+id+":approve", idempotencyKey,
		struct {
			ExpectedRevision int64  `json:"expectedRevision"`
			DiscoveryDigest  string `json:"discoveryDigest"`
		}{ExpectedRevision: expected, DiscoveryDigest: discoveryDigest}, mutation,
	)
	if err != nil {
		return ResourceMutationResult{}, err
	}
	if replay, found, replayErr := service.store.ReplayResourceCommand(
		ctx, command.Command, agentToolSourceResourceType,
	); replayErr != nil || found {
		return replay, replayErr
	}
	return service.store.ApproveToolSourceDiscovery(
		ctx, namespaceID, id, expected, discoveryDigest, command,
	)
}

func (service *Service) bindResourceCommand(
	namespaceID, endpoint, idempotencyKey string, canonical any, mutation MutationContext,
) (ResourceCommand, error) {
	encoded, err := json.Marshal(canonical)
	if err != nil {
		return ResourceCommand{}, ErrInvalid
	}
	defer clear(encoded)
	now := service.now().UTC()
	command, err := service.commandCodec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), mutation.PrincipalID,
		endpoint, idempotencyKey, encoded, now, now.Add(agentResourceCommandTTL),
	)
	if errors.Is(err, managementcommand.ErrConflict) {
		return ResourceCommand{}, ErrConflict
	}
	if err != nil {
		return ResourceCommand{}, ErrInvalid
	}
	return ResourceCommand{Mutation: mutation, Command: command}, nil
}

func (service *Service) PrepareSession(
	ctx context.Context, namespaceID, principalID string, input SessionInput,
) (SessionAuthorization, error) {
	profile, err := service.sessionProfile(ctx, namespaceID, principalID, input, AccessContext{
		PrincipalID: principalID,
		Scope:       accesscontrol.ResultScope{NamespaceID: accesscontrol.NamespaceID(namespaceID), All: true},
	})
	if err != nil {
		return SessionAuthorization{}, err
	}
	return service.sessionAuthority.Prepare(ctx, SessionAuthorizationRequest{
		NamespaceID: namespaceID, PrincipalID: principalID,
		EffectiveTeamID: input.EffectiveTeamID, Profile: profile, Target: input.Target,
	})
}

func (service *Service) ResolveSessionAccess(
	ctx context.Context, namespaceID, sessionID string,
) (SessionAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil {
		return SessionAccess{}, ErrInvalid
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return SessionAccess{}, err
	}
	return SessionAccess{
		ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
		EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
	}, nil
}

func (service *Service) CreateSession(
	ctx context.Context, namespaceID, principalID, idempotencyKey string, input SessionInput,
	mutation MutationContext, access AccessContext,
) (Session, bool, error) {
	profile, err := service.sessionProfile(ctx, namespaceID, principalID, input, access)
	if err != nil {
		return Session{}, false, err
	}
	encoded, err := json.Marshal(input)
	if err != nil {
		return Session{}, false, ErrInvalid
	}
	now := service.now().UTC()
	command, err := service.commandCodec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), principalID,
		"/management/v1/agent-sessions", idempotencyKey, encoded, now, now.Add(7*24*time.Hour),
	)
	if err != nil {
		return Session{}, false, ErrInvalid
	}
	return service.sessionAuthority.Bootstrap(ctx, SessionBootstrapRequest{
		SessionID: uuid.NewString(), NamespaceID: namespaceID, PrincipalID: principalID,
		EffectiveTeamID: input.EffectiveTeamID, Profile: profile, Target: input.Target,
		Mode: input.Mode, Title: strings.TrimSpace(input.Title), SessionTTL: service.sessionTTL,
		Mutation: mutation, Command: command,
	})
}

func (service *Service) sessionProfile(
	ctx context.Context, namespaceID, principalID string, input SessionInput, access AccessContext,
) (Profile, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(principalID) != nil ||
		(input.ProfileID != "" && uuid.Validate(input.ProfileID) != nil) ||
		(input.EffectiveTeamID != "" && uuid.Validate(input.EffectiveTeamID) != nil) ||
		validateTarget(input.Target) != nil ||
		(input.Mode != SessionChat && input.Mode != SessionBuilder) || len(input.Title) > 256 {
		return Profile{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil || access.PrincipalID != principalID {
		return Profile{}, ErrDenied
	}
	var (
		profile Profile
		err     error
	)
	if input.ProfileID == "" {
		profile, err = service.store.GetDefaultProfile(ctx, namespaceID, input.Mode)
	} else {
		profile, err = service.store.GetProfile(ctx, namespaceID, input.ProfileID)
	}
	if err != nil {
		return Profile{}, err
	}
	if profile.Status != StatusActive || !supportsMode(profile.SupportedModes, input.Mode) {
		return Profile{}, ErrNotFound
	}
	return profile, nil
}

func (service *Service) CreateTurn(
	ctx context.Context, namespaceID, sessionID, idempotencyKey string, input TurnInput, access AccessContext,
) (Turn, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		ValidateTurnInput(input) != nil {
		return Turn{}, false, ErrInvalid
	}
	original, createTurnErr := json.Marshal(input)
	if createTurnErr != nil {
		return Turn{}, false, ErrInvalid
	}
	input, createTurnErr = NormalizeTurnInput(input)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Turn{}, false, err
	}
	session, createTurnErr := service.store.GetSession(ctx, namespaceID, sessionID)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if session.Status != SessionActive || !accessCanReadSession(access, session) {
		return Turn{}, false, ErrDenied
	}
	currentProfile, createTurnErr := service.store.GetProfile(ctx, namespaceID, session.ProfileID)
	if createTurnErr != nil || currentProfile.Status != StatusActive {
		if createTurnErr != nil {
			return Turn{}, false, createTurnErr
		}
		return Turn{}, false, ErrDenied
	}
	profile, createTurnErr := service.store.GetProfileRevision(ctx, namespaceID, session.ProfileID, session.ProfileRevision)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if err := service.sessionAuthority.Reauthorize(ctx, session, profile.MinimumTargetCapabilities); err != nil {
		return Turn{}, false, err
	}
	registry, createTurnErr := service.registries.Current(ctx, namespaceID)
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	now := service.now().UTC()
	command, createTurnErr := service.commandCodec.Bind(
		managementcommand.NamespaceCommandScope(namespaceID), access.PrincipalID,
		"/management/v1/agent-sessions/"+sessionID+"/turns", idempotencyKey,
		original, now, now.Add(7*24*time.Hour),
	)
	if createTurnErr != nil {
		return Turn{}, false, ErrInvalid
	}
	turn := Turn{
		ID: uuid.NewString(), SessionID: sessionID, Status: TurnQueued,
		RegistryRevision: registry.Revision(), Input: input, Revision: 1, CreatedAt: now, UpdatedAt: now,
	}
	created, replayed, createTurnErr := service.store.CreateTurn(ctx, CreateTurnRequest{
		Turn: turn, NamespaceID: namespaceID, ActorPrincipalID: access.PrincipalID, Command: command,
	})
	if createTurnErr != nil {
		return Turn{}, false, createTurnErr
	}
	if !replayed && service.notifier != nil {
		if wakeErr := service.notifier.Wake(ctx, namespaceID, created.ID); wakeErr != nil {
			// The durable queued row remains discoverable by polling workers.
			_ = wakeErr
		}
	}
	return created, replayed, nil
}

func (service *Service) ListSessions(
	ctx context.Context, namespaceID string, request PageRequest, access AccessContext,
) (Page[Session], error) {
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Page[Session]{}, err
	}
	if !access.Scope.All && request.OwnerPrincipalID == "" {
		request.OwnerPrincipalID = access.PrincipalID
	}
	query, err := service.listQuery(namespaceID, "sessions", request)
	if err != nil {
		return Page[Session]{}, err
	}
	result, err := service.store.ListSessions(ctx, namespaceID, query)
	return makePage(service, namespaceID, "sessions", query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) GetSession(
	ctx context.Context, namespaceID, sessionID string, access AccessContext,
) (Session, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil {
		return Session{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Session{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Session{}, err
	}
	if !accessCanReadSession(access, session) {
		return Session{}, ErrNotFound
	}
	return session, nil
}

func (service *Service) PatchSession(
	ctx context.Context, namespaceID, sessionID string, expected int64,
	patch SessionPatch, mutation MutationContext, access AccessContext,
) (Session, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil || expected < 1 ||
		(patch.Title == nil && patch.Status == nil) {
		return Session{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Session{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Session{}, err
	}
	if !accessCanReadSession(access, session) {
		return Session{}, ErrNotFound
	}
	if patch.Status == nil {
		return service.store.PatchSession(ctx, namespaceID, sessionID, expected, patch, mutation)
	}
	if *patch.Status != SessionClosed || session.Status != SessionActive {
		return Session{}, ErrInvalid
	}
	return service.sessionAuthority.Close(ctx, session, expected, patch, mutation)
}

func (service *Service) DeleteSession(
	ctx context.Context, namespaceID, sessionID string, expected int64,
	mutation MutationContext, access AccessContext,
) (int64, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil || expected < 1 {
		return 0, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return 0, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return 0, err
	}
	if !accessCanReadSession(access, session) {
		return 0, ErrNotFound
	}
	if session.Status != SessionClosed {
		return 0, ErrConflict
	}
	return service.store.DeleteSession(ctx, namespaceID, sessionID, expected, mutation)
}

func (service *Service) ListTurns(
	ctx context.Context, namespaceID, sessionID string, request PageRequest, access AccessContext,
) (Page[Turn], error) {
	if _, err := service.GetSession(ctx, namespaceID, sessionID, access); err != nil {
		return Page[Turn]{}, err
	}
	query, err := service.listQuery(namespaceID, "turns:"+sessionID, request)
	if err != nil {
		return Page[Turn]{}, err
	}
	result, err := service.store.ListTurns(ctx, namespaceID, sessionID, query)
	return makePage(service, namespaceID, "turns:"+sessionID, query, request.PageSize,
		result.Items, result.HasMore, err)
}

func (service *Service) ResumeEvents(
	ctx context.Context, namespaceID, sessionID string, after int64, limit int, access AccessContext,
) ([]Event, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		after < 0 || limit < 1 || limit > 1000 {
		return nil, false, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return nil, false, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return nil, false, err
	}
	if !accessCanReadSession(access, session) {
		return nil, false, ErrDenied
	}
	oldest, err := service.store.OldestEventSequence(ctx, namespaceID, sessionID)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return nil, false, err
	}
	if oldest > 1 && after+1 < oldest {
		checkpoint, checkpointErr := service.store.LatestCheckpoint(ctx, namespaceID, sessionID)
		if checkpointErr != nil {
			return nil, false, HistoryExpiredError{Recovery: HistoryRecovery{ThroughSequence: oldest - 1}}
		}
		return nil, false, HistoryExpiredError{Recovery: HistoryRecovery{
			CheckpointID: checkpoint.ID, ThroughSequence: checkpoint.ThroughSequence,
		}}
	}
	return service.store.ListEventsAfter(ctx, namespaceID, sessionID, after, limit)
}

// ListEventHistory returns the newest retained page on the first request and
// older pages thereafter. Items are always ascending so a client can append
// them to a transcript without reordering. The opaque cursor is bound to the
// namespace and evaluated result scope.
func (service *Service) ListEventHistory(
	ctx context.Context,
	namespaceID string,
	sessionID string,
	request EventPageRequest,
	access AccessContext,
) (Page[Event], error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		request.PageSize < 1 || request.PageSize > 1000 {
		return Page[Event]{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Page[Event]{}, err
	}
	canonicalScope, err := request.Scope.Canonical()
	if err != nil || string(canonicalScope.NamespaceID) != namespaceID {
		return Page[Event]{}, ErrDenied
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil {
		return Page[Event]{}, err
	}
	if !accessCanReadSession(access, session) {
		return Page[Event]{}, ErrDenied
	}
	scopeDigest, err := canonicalScope.Digest()
	if err != nil {
		return Page[Event]{}, ErrDenied
	}
	before := int64(0)
	if request.Cursor != "" {
		cursor, decodeErr := service.codec.decodeCursor(request.Cursor)
		if decodeErr != nil || cursor.NamespaceID != namespaceID || cursor.Kind != "events" ||
			cursor.ScopeDigest != scopeDigest {
			return Page[Event]{}, ErrInvalid
		}
		before = cursor.Sequence
	}
	items, hasMore, err := service.store.ListEventHistory(ctx, namespaceID, sessionID,
		EventHistoryQuery{BeforeSequence: before, Limit: request.PageSize})
	if err != nil {
		return Page[Event]{}, err
	}
	page := Page[Event]{Items: items, HasMore: hasMore}
	if hasMore && len(items) > 0 {
		page.NextCursor, err = service.codec.encodeCursor(cursorPayload{
			NamespaceID: namespaceID,
			Kind:        "events",
			ScopeDigest: scopeDigest,
			Sequence:    items[0].Sequence,
		})
	}
	return page, err
}

func (service *Service) GetArtifactMetadata(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (Artifact, error) {
	artifact, err := service.getAuthorizedArtifact(ctx, namespaceID, artifactID, access)
	if err != nil {
		return Artifact{}, err
	}
	clear(artifact.Content)
	artifact.Content = nil
	return artifact, nil
}

func (service *Service) ResolveArtifactAccess(
	ctx context.Context, namespaceID, artifactID string,
) (ArtifactAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(artifactID) != nil {
		return ArtifactAccess{}, ErrInvalid
	}
	artifact, err := service.store.GetArtifact(ctx, namespaceID, artifactID)
	if err != nil || !service.now().UTC().Before(artifact.ExpiresAt) {
		if err != nil {
			return ArtifactAccess{}, err
		}
		return ArtifactAccess{}, ErrNotFound
	}
	session, err := service.store.GetSession(ctx, namespaceID, artifact.SessionID)
	if err != nil {
		return ArtifactAccess{}, err
	}
	return ArtifactAccess{ID: artifact.ID, Session: SessionAccess{
		ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
		EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
	}}, nil
}

func (service *Service) GetArtifactContent(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (ArtifactContent, error) {
	artifact, err := service.getAuthorizedArtifact(ctx, namespaceID, artifactID, access)
	if err != nil {
		return ArtifactContent{}, err
	}
	return ArtifactContent{
		ID: artifact.ID, MediaType: artifact.MediaType,
		Encoding: "base64", Content: append([]byte(nil), artifact.Content...), Digest: artifact.Digest,
	}, nil
}

func (service *Service) getAuthorizedArtifact(
	ctx context.Context, namespaceID, artifactID string, access AccessContext,
) (Artifact, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(artifactID) != nil {
		return Artifact{}, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Artifact{}, err
	}
	artifact, err := service.store.GetArtifact(ctx, namespaceID, artifactID)
	if err != nil {
		return Artifact{}, err
	}
	if !service.now().UTC().Before(artifact.ExpiresAt) {
		return Artifact{}, ErrNotFound
	}
	session, err := service.store.GetSession(ctx, namespaceID, artifact.SessionID)
	if err != nil {
		return Artifact{}, err
	}
	if !accessCanReadSession(access, session) {
		return Artifact{}, ErrDenied
	}
	return artifact, nil
}

func (service *Service) RequestCancellation(
	ctx context.Context, namespaceID, sessionID, turnID string, access AccessContext,
) (Turn, bool, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(sessionID) != nil ||
		uuid.Validate(turnID) != nil {
		return Turn{}, false, ErrInvalid
	}
	if err := validateAccessContext(namespaceID, access); err != nil {
		return Turn{}, false, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, sessionID)
	if err != nil || !accessCanReadSession(access, session) {
		if err != nil {
			return Turn{}, false, err
		}
		return Turn{}, false, ErrDenied
	}
	turn, replayed, err := service.store.RequestCancellation(
		ctx, namespaceID, sessionID, turnID, service.now().UTC(),
	)
	if err != nil || replayed {
		return turn, replayed, err
	}
	if service.notifier != nil {
		// Notification is acceleration only. The durable cancellation flag was
		// committed above and a worker observes it even when fan-out fails.
		_ = service.notifier.NotifyCancellation(ctx, namespaceID, turnID)
	}
	return turn, false, nil
}

func (service *Service) ResolvePublicationAccess(
	ctx context.Context, namespaceID, planID string,
) (PublicationAccess, error) {
	if uuid.Validate(namespaceID) != nil || uuid.Validate(planID) != nil {
		return PublicationAccess{}, ErrInvalid
	}
	plan, err := service.store.GetPublicationPlan(ctx, namespaceID, planID)
	if err != nil {
		return PublicationAccess{}, err
	}
	session, err := service.store.GetSession(ctx, namespaceID, plan.SessionID)
	if err != nil {
		return PublicationAccess{}, err
	}
	models, err := service.store.GetPublicationModelIDs(ctx, namespaceID, planID)
	if err != nil {
		return PublicationAccess{}, err
	}
	return PublicationAccess{
		PlanID: plan.ID, SessionID: plan.SessionID, RecipeID: plan.RecipeID,
		EntrypointID: plan.EntrypointID, ModelIDs: models, Revision: plan.Revision,
		Digest: plan.Digest, ExpiresAt: plan.ExpiresAt,
		Session: SessionAccess{
			ID: session.ID, OwnerPrincipalID: session.OwnerPrincipalID,
			EffectiveUserID: session.EffectiveUserID, EffectiveTeamID: session.EffectiveTeamID,
		},
	}, nil
}

func (service *Service) listQuery(namespaceID, kind string, request PageRequest) (ListQuery, error) {
	if uuid.Validate(namespaceID) != nil || request.PageSize < 1 || request.PageSize > 200 {
		return ListQuery{}, ErrInvalid
	}
	search, err := managementsearch.Normalize(request.Search)
	if err != nil {
		return ListQuery{}, ErrInvalid
	}
	canonicalScope, err := request.Scope.Canonical()
	if err != nil || string(canonicalScope.NamespaceID) != namespaceID {
		return ListQuery{}, ErrDenied
	}
	scopeDigest, err := canonicalScope.Digest()
	if err != nil {
		return ListQuery{}, ErrDenied
	}
	query := ListQuery{
		Limit: request.PageSize + 1, Search: search,
		OwnerPrincipalID: request.OwnerPrincipalID, Scope: canonicalScope,
	}
	if request.Cursor == "" {
		return query, nil
	}
	cursor, err := service.codec.decodeCursor(request.Cursor)
	if err != nil || cursor.NamespaceID != namespaceID || cursor.Kind != kind ||
		cursor.ScopeDigest != scopeDigest || cursor.Search != search ||
		cursor.OwnerPrincipalID != request.OwnerPrincipalID {
		return ListQuery{}, ErrInvalid
	}
	query.After = &Seek{Timestamp: cursor.Timestamp, ID: cursor.ID}
	return query, nil
}

func makePage[T any](
	service *Service, namespaceID, kind string, query ListQuery,
	pageSize int, items []T, hasMore bool, listErr error,
) (Page[T], error) {
	if listErr != nil {
		return Page[T]{}, listErr
	}
	if len(items) > pageSize {
		items = items[:pageSize]
		hasMore = true
	}
	page := Page[T]{Items: items, HasMore: hasMore}
	if hasMore && len(items) > 0 {
		timestamp, id, ok := pageIdentity(any(items[len(items)-1]))
		if !ok {
			return Page[T]{}, fmt.Errorf("%w: unsupported Agent page type", ErrInvalid)
		}
		scopeDigest, err := query.Scope.Digest()
		if err != nil {
			return Page[T]{}, ErrDenied
		}
		cursor, err := service.codec.encodeCursor(cursorPayload{
			NamespaceID: namespaceID, Kind: kind, ScopeDigest: scopeDigest,
			Search: query.Search, OwnerPrincipalID: query.OwnerPrincipalID,
			Timestamp: timestamp, ID: id,
		})
		if err != nil {
			return Page[T]{}, err
		}
		page.NextCursor = cursor
	}
	return page, nil
}

func pageIdentity(value interface{}) (time.Time, string, bool) {
	switch typed := value.(type) {
	case Profile:
		return typed.CreatedAt, typed.ID, true
	case Skill:
		return typed.CreatedAt, typed.ID, true
	case ToolSource:
		return typed.CreatedAt, typed.ID, true
	case ToolCredential:
		return typed.CreatedAt, typed.ID, true
	case Session:
		return typed.UpdatedAt, typed.ID, true
	case Turn:
		return typed.CreatedAt, typed.ID, true
	default:
		return time.Time{}, "", false
	}
}

func patchProfileEmpty(patch ProfilePatch) bool {
	return patch.Name == nil && patch.Description == nil && !patch.DefaultTarget.Present &&
		patch.MinimumTargetCapabilities == nil && patch.Skills == nil &&
		patch.SupportedModes == nil && patch.DefaultForModes == nil &&
		patch.ToolPolicy == nil && patch.ApprovalPolicy == nil &&
		patch.MaximumTurnSeconds == nil && patch.MaximumToolSteps == nil &&
		patch.ContextTokenBudget == nil
}

func patchSkillEmpty(patch SkillPatch) bool {
	return patch.Name == nil && patch.Description == nil && patch.Instructions == nil &&
		patch.RequiredTools == nil && patch.MinimumCapabilities == nil
}

func toolSourcePatchEmpty(patch ToolSourcePatch) bool {
	return patch.Name == nil && patch.Description == nil && patch.Transport == nil &&
		patch.Endpoint == nil && !patch.CredentialID.Present && patch.EgressPolicy == nil && patch.Status == nil
}

func supportsMode(modes []SessionMode, requested SessionMode) bool {
	for _, mode := range modes {
		if mode == requested {
			return true
		}
	}
	return false
}

func normalizeProfilePatch(current Profile, patch ProfilePatch) (ProfileInput, ProfilePatch, error) {
	input := ProfileInput{
		Name: current.Name, Description: current.Description, DefaultTarget: current.DefaultTarget,
		MinimumTargetCapabilities: current.MinimumTargetCapabilities,
		SupportedModes:            current.SupportedModes, DefaultForModes: current.DefaultForModes,
		Skills: current.Skills, ToolPolicy: current.ToolPolicy, ApprovalPolicy: current.ApprovalPolicy,
		MaximumTurnSeconds: current.MaximumTurnSeconds, MaximumToolSteps: current.MaximumToolSteps,
		ContextTokenBudget: current.ContextTokenBudget,
	}
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.DefaultTarget.Present {
		input.DefaultTarget = patch.DefaultTarget.Value
	}
	if patch.MinimumTargetCapabilities != nil {
		input.MinimumTargetCapabilities = *patch.MinimumTargetCapabilities
	}
	if patch.SupportedModes != nil {
		input.SupportedModes = *patch.SupportedModes
	}
	if patch.DefaultForModes != nil {
		input.DefaultForModes = *patch.DefaultForModes
	}
	if patch.Skills != nil {
		input.Skills = *patch.Skills
	}
	if patch.ToolPolicy != nil {
		input.ToolPolicy = *patch.ToolPolicy
	}
	if patch.ApprovalPolicy != nil {
		input.ApprovalPolicy = *patch.ApprovalPolicy
	}
	if patch.MaximumTurnSeconds != nil {
		input.MaximumTurnSeconds = *patch.MaximumTurnSeconds
	}
	if patch.MaximumToolSteps != nil {
		input.MaximumToolSteps = *patch.MaximumToolSteps
	}
	if patch.ContextTokenBudget != nil {
		input.ContextTokenBudget = *patch.ContextTokenBudget
	}
	input, err := NormalizeProfileInput(input)
	if err != nil {
		return ProfileInput{}, ProfilePatch{}, err
	}
	if patch.Name != nil {
		patch.Name = &input.Name
	}
	if patch.Description != nil {
		patch.Description = &input.Description
	}
	if patch.DefaultTarget.Present {
		patch.DefaultTarget.Value = input.DefaultTarget
	}
	if patch.MinimumTargetCapabilities != nil {
		patch.MinimumTargetCapabilities = &input.MinimumTargetCapabilities
	}
	if patch.SupportedModes != nil {
		patch.SupportedModes = &input.SupportedModes
	}
	if patch.DefaultForModes != nil {
		patch.DefaultForModes = &input.DefaultForModes
	}
	if patch.Skills != nil {
		patch.Skills = &input.Skills
	}
	if patch.ToolPolicy != nil {
		patch.ToolPolicy = &input.ToolPolicy
	}
	if patch.ApprovalPolicy != nil {
		patch.ApprovalPolicy = &input.ApprovalPolicy
	}
	return input, patch, nil
}

func normalizeSkillPatch(current Skill, patch SkillPatch) (SkillInput, SkillPatch, error) {
	input := SkillInput{
		Name: current.Name, Description: current.Description, Instructions: current.Instructions,
		RequiredTools: current.RequiredTools, MinimumCapabilities: current.MinimumCapabilities,
	}
	if patch.Name != nil {
		input.Name = *patch.Name
	}
	if patch.Description != nil {
		input.Description = *patch.Description
	}
	if patch.Instructions != nil {
		input.Instructions = *patch.Instructions
	}
	if patch.RequiredTools != nil {
		input.RequiredTools = *patch.RequiredTools
	}
	if patch.MinimumCapabilities != nil {
		input.MinimumCapabilities = *patch.MinimumCapabilities
	}
	input, err := NormalizeSkillInput(input)
	if err != nil {
		return SkillInput{}, SkillPatch{}, err
	}
	if patch.Name != nil {
		patch.Name = &input.Name
	}
	if patch.Description != nil {
		patch.Description = &input.Description
	}
	if patch.Instructions != nil {
		patch.Instructions = &input.Instructions
	}
	if patch.RequiredTools != nil {
		patch.RequiredTools = &input.RequiredTools
	}
	if patch.MinimumCapabilities != nil {
		patch.MinimumCapabilities = &input.MinimumCapabilities
	}
	return input, patch, nil
}

func validateAccessContext(namespaceID string, access AccessContext) error {
	if uuid.Validate(access.PrincipalID) != nil {
		return ErrDenied
	}
	canonical, err := access.Scope.Canonical()
	if err != nil || string(canonical.NamespaceID) != namespaceID {
		return ErrDenied
	}
	return nil
}

func accessCanReadSession(access AccessContext, session Session) bool {
	if access.PrincipalID == session.OwnerPrincipalID || access.Scope.All {
		return true
	}
	for _, id := range access.Scope.IDs(accesscontrol.ScopeResourceAgentSession) {
		if string(id) == session.ID {
			return true
		}
	}
	for _, id := range access.Scope.UserIDs {
		if string(id) == session.EffectiveUserID {
			return true
		}
	}
	for _, id := range access.Scope.TeamIDs {
		if string(id) == session.EffectiveTeamID {
			return true
		}
	}
	return false
}
