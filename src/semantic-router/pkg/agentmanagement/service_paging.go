package agentmanagement

import (
	"fmt"
	"time"

	"github.com/google/uuid"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementsearch"
)

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
