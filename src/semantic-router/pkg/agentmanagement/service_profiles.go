package agentmanagement

import (
	"context"

	"github.com/google/uuid"
)

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
