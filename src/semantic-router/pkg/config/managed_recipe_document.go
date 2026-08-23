package config

import (
	"bytes"
	"encoding/json"
	"fmt"

	"gopkg.in/yaml.v2"
	jsonyaml "sigs.k8s.io/yaml"
)

func validateRecipeDocumentModelFree(routing CanonicalRouting) error {
	original, marshalErr := json.Marshal(routing.Decisions)
	if marshalErr != nil {
		return fmt.Errorf("encode decisions: %w", marshalErr)
	}
	var cloned []Decision
	if err := json.Unmarshal(original, &cloned); err != nil {
		return fmt.Errorf("clone decisions: %w", err)
	}
	for index := range cloned {
		stripManagedRecipeModelSelection(&cloned[index])
	}
	stripped, marshalErr := json.Marshal(cloned)
	if marshalErr != nil {
		return fmt.Errorf("encode model-free decisions: %w", marshalErr)
	}
	if !bytes.Equal(original, stripped) {
		return fmt.Errorf("physical Model selection belongs exclusively to Entrypoint assignments")
	}
	return nil
}

// MarshalManagedRecipeDocument projects a canonical routing profile into the
// model-free Recipe document accepted by the Management API. Model selection
// belongs exclusively to Entrypoint decision assignments.
func MarshalManagedRecipeDocument(routing CanonicalRouting) (json.RawMessage, error) {
	for index := range routing.Decisions {
		// Decision identity is publication state. Human Recipe documents use the
		// readable Decision name; snapshot compilation owns stable IDs.
		routing.Decisions[index].ID = ""
		stripManagedRecipeModelSelection(&routing.Decisions[index])
	}
	payload, err := yaml.Marshal(ManagedRecipeDocument(routing))
	if err != nil {
		return nil, err
	}
	document, err := jsonyaml.YAMLToJSON(payload)
	if err != nil {
		return nil, err
	}
	_, canonical, err := ParseManagedRecipeDocument(document)
	if err != nil {
		return nil, err
	}
	return canonical, nil
}

func stripManagedRecipeModelSelection(decision *Decision) {
	decision.ModelRefs = nil
	for index := range decision.CandidateIterations {
		decision.CandidateIterations[index].Models = nil
	}
	if decision.Algorithm == nil {
		return
	}
	if value := decision.Algorithm.Fusion; value != nil {
		value.Model = ""
		value.AnalysisModels = nil
		value.AnalysisOverrides = nil
	}
	if value := decision.Algorithm.Workflows; value != nil {
		value.Planner.Model = ""
		value.Final.Model = ""
		for index := range value.Roles {
			value.Roles[index].Models = nil
		}
	}
	if value := decision.Algorithm.ReMoM; value != nil {
		value.SynthesisModel = ""
	}
	if value := decision.Algorithm.Prompt; value != nil {
		value.Model = ""
	}
}
