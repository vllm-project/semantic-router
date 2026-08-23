package agentmanagement

import (
	"context"
	"errors"
	"fmt"
)

// SkillRevisionReader is the narrow persistence seam required while
// validating immutable Profile pins. Validation always reads the exact
// revision named by the Profile; it never substitutes the mutable Skill head.
type SkillRevisionReader interface {
	GetSkillRevision(context.Context, string, string, int64) (Skill, error)
}

// RegistryDefinitionValidator keeps Profile, Skill, and Tool Registry
// references coherent before a new immutable resource revision is persisted.
// It deliberately validates definitions only; runtime authorization remains a
// separate, live decision at session creation and every invocation.
type RegistryDefinitionValidator struct {
	skills SkillRevisionReader
}

func NewRegistryDefinitionValidator(skills SkillRevisionReader) (*RegistryDefinitionValidator, error) {
	if skills == nil {
		return nil, errors.New("agent definition validator requires a Skill revision reader")
	}
	return &RegistryDefinitionValidator{skills: skills}, nil
}

func (validator *RegistryDefinitionValidator) ValidateProfile(
	ctx context.Context,
	namespaceID string,
	input ProfileInput,
	registry *ToolRegistry,
) error {
	if validator == nil || validator.skills == nil || registry == nil {
		return fmt.Errorf("%w: Agent definition validation is unavailable", ErrToolUnavailable)
	}
	visible := registry.Definitions(input.ToolPolicy)
	if len(visible) == 0 {
		return fmt.Errorf("%w: Agent Profile grants no available tools", ErrInvalid)
	}
	available := make(map[string]struct{}, len(visible))
	for _, definition := range visible {
		available[definition.Name] = struct{}{}
	}
	for _, reference := range input.Skills {
		skill, err := validator.skills.GetSkillRevision(
			ctx, namespaceID, reference.ID, reference.Revision,
		)
		if err != nil {
			return err
		}
		for _, required := range skill.RequiredTools {
			if _, found := available[required]; !found {
				return fmt.Errorf("%w: Profile policy does not grant required Skill tool %q", ErrInvalid, required)
			}
		}
		if !containsEveryString(input.MinimumTargetCapabilities, skill.MinimumCapabilities) {
			return fmt.Errorf("%w: Profile capabilities do not satisfy pinned Skill %q", ErrInvalid, skill.Name)
		}
	}
	return nil
}

func (validator *RegistryDefinitionValidator) ValidateSkill(
	_ context.Context,
	_ string,
	input SkillInput,
	registry *ToolRegistry,
) error {
	if validator == nil || validator.skills == nil || registry == nil {
		return fmt.Errorf("%w: Agent definition validation is unavailable", ErrToolUnavailable)
	}
	available := make(map[string]struct{})
	for _, definition := range registry.AllDefinitions() {
		available[definition.Name] = struct{}{}
	}
	for _, required := range input.RequiredTools {
		if _, found := available[required]; !found {
			return fmt.Errorf("%w: Skill requires unavailable tool %q", ErrInvalid, required)
		}
	}
	return nil
}

func containsEveryString(have, required []string) bool {
	values := make(map[string]struct{}, len(have))
	for _, value := range have {
		values[value] = struct{}{}
	}
	for _, value := range required {
		if _, found := values[value]; !found {
			return false
		}
	}
	return true
}

var _ DefinitionValidator = (*RegistryDefinitionValidator)(nil)
