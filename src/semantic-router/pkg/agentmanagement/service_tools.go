package agentmanagement

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementcommand"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

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
