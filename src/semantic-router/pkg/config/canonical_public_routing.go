package config

import "strings"

// RoutingModel is the connection-free semantic half of one public v0.3
// Model. Its name joins exactly one providers.models member.
type RoutingModel struct {
	Name              string         `yaml:"name"`
	ParamSize         string         `yaml:"param_size,omitempty"`
	ContextWindowSize int            `yaml:"context_window_size,omitempty"`
	Description       string         `yaml:"description,omitempty"`
	Capabilities      []string       `yaml:"capabilities,omitempty"`
	Reasoning         ModelReasoning `yaml:"reasoning,omitempty"`
	LoRAs             []LoRAAdapter  `yaml:"loras,omitempty"`
	QualityScore      float64        `yaml:"quality_score,omitempty"`
	Modality          string         `yaml:"modality,omitempty"`
	Tags              []string       `yaml:"tags,omitempty"`
}

// ModelReasoning is the semantic reasoning surface exposed by a Model card.
// Type must agree with the physical Model's selected reasoning family, while
// Efforts declares the values that Entrypoint assignments may request.
type ModelReasoning struct {
	Type    string   `yaml:"type,omitempty"`
	Efforts []string `yaml:"efforts,omitempty"`
}

// CanonicalRecipe is one reusable public v0.3 routing profile. Connections,
// credentials, immutable IDs, and Entrypoint assignments do not belong here.
type CanonicalRecipe struct {
	Name        string           `yaml:"name"`
	Description string           `yaml:"description,omitempty"`
	Routing     CanonicalRouting `yaml:"routing"`
}

// CanonicalEntrypoint is the request-facing Mixture-of-Models name set. Its
// optional assignments are one complete Decision-to-Model mapping.
type CanonicalEntrypoint struct {
	ModelNames  []string                           `yaml:"model_names"`
	Recipe      string                             `yaml:"recipe"`
	Assignments map[string]EntrypointAssignmentSet `yaml:"assignments,omitempty"`
}

type EntrypointAssignmentSet struct {
	Models   []EntrypointModelAssignment `yaml:"models"`
	Fallback *EntrypointFallbackPolicy   `yaml:"fallback,omitempty"`
}

type EntrypointModelAssignment struct {
	Model     string                         `yaml:"model"`
	Priority  int                            `yaml:"priority,omitempty"`
	Weight    string                         `yaml:"weight,omitempty"`
	LoRA      string                         `yaml:"lora,omitempty"`
	Reasoning *EntrypointAssignmentReasoning `yaml:"reasoning,omitempty"`
}

type EntrypointFallbackPolicy struct {
	Strategy string   `yaml:"strategy"`
	On       []string `yaml:"on"`
}

type EntrypointAssignmentReasoning struct {
	Enabled     bool   `yaml:"enabled"`
	Effort      string `yaml:"effort,omitempty"`
	Description string `yaml:"description,omitempty"`
}

func canonicalRoutingModels(routing CanonicalRouting) []RoutingModel {
	return routing.ModelCards
}

func publicAssignmentSetToAuthoring(input EntrypointAssignmentSet) AuthoringAssignmentSet {
	models := make([]AuthoringModelAssignment, 0, len(input.Models))
	for _, model := range input.Models {
		var reasoning *AuthoringAssignmentReasoning
		if model.Reasoning != nil {
			reasoning = &AuthoringAssignmentReasoning{
				Enabled: model.Reasoning.Enabled, Effort: model.Reasoning.Effort,
				Description: model.Reasoning.Description,
			}
		}
		models = append(models, AuthoringModelAssignment{
			Model: strings.TrimSpace(model.Model), Priority: model.Priority,
			Weight: model.Weight, LoRAName: model.LoRA, Reasoning: reasoning,
		})
	}
	result := AuthoringAssignmentSet{Models: models}
	if input.Fallback != nil {
		result.Fallback = &AuthoringFallbackPolicy{
			Strategy: input.Fallback.Strategy, On: append([]string(nil), input.Fallback.On...),
		}
	}
	return result
}

func publicAssignmentsToAuthoring(
	input map[string]EntrypointAssignmentSet,
) map[string]AuthoringAssignmentSet {
	if len(input) == 0 {
		return nil
	}
	result := make(map[string]AuthoringAssignmentSet, len(input))
	for decisionName, set := range input {
		result[decisionName] = publicAssignmentSetToAuthoring(set)
	}
	return result
}

func authoringAssignmentSetToPublic(input AuthoringAssignmentSet) EntrypointAssignmentSet {
	models := make([]EntrypointModelAssignment, 0, len(input.Models))
	for _, model := range input.Models {
		var reasoning *EntrypointAssignmentReasoning
		if model.Reasoning != nil {
			reasoning = &EntrypointAssignmentReasoning{
				Enabled: model.Reasoning.Enabled, Effort: model.Reasoning.Effort,
				Description: model.Reasoning.Description,
			}
		}
		models = append(models, EntrypointModelAssignment{
			Model: model.Model, Priority: model.Priority, Weight: model.Weight,
			LoRA: model.LoRAName, Reasoning: reasoning,
		})
	}
	result := EntrypointAssignmentSet{Models: models}
	if input.Fallback != nil {
		result.Fallback = &EntrypointFallbackPolicy{
			Strategy: input.Fallback.Strategy, On: append([]string(nil), input.Fallback.On...),
		}
	}
	return result
}
