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
		stripRoutingRecipeModelSelection(&cloned[index])
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

// MarshalRoutingRecipeDocument projects a canonical routing profile into the
// model-free Recipe document accepted by the Management API. Model selection
// belongs exclusively to Entrypoint decision assignments.
func MarshalRoutingRecipeDocument(routing CanonicalRouting) (json.RawMessage, error) {
	document := routingRecipeDocumentFromCanonical(routing)
	for index := range document.Decisions {
		// Decision identity is publication state. Human Recipe documents use the
		// readable Decision name; snapshot compilation owns stable IDs.
		document.Decisions[index].ID = ""
		stripRoutingRecipeModelSelection(&document.Decisions[index])
	}
	payload, err := yaml.Marshal(document)
	if err != nil {
		return nil, err
	}
	encoded, err := jsonyaml.YAMLToJSON(payload)
	if err != nil {
		return nil, err
	}
	_, canonical, err := ParseRoutingRecipeDocument(encoded)
	if err != nil {
		return nil, err
	}
	return canonical, nil
}

func routingRecipeDocumentFromCanonical(routing CanonicalRouting) RoutingRecipeDocument {
	return RoutingRecipeDocument{
		Signals:     routing.Signals,
		Projections: routing.Projections,
		Decisions:   cloneEntrypointDecisions(routing.Decisions),
		Strategy:    routing.Strategy,
	}
}

// CanonicalRoutingFromRecipeDocument projects the strict, model-free Recipe
// document into the routing view shared by DSL emitters and publication
// exporters. Keeping this conversion here prevents sibling packages from
// relying on struct conversions that break as either contract evolves.
func CanonicalRoutingFromRecipeDocument(document RoutingRecipeDocument) CanonicalRouting {
	return CanonicalRouting{
		Signals:     document.Signals,
		Projections: document.Projections,
		Decisions:   cloneEntrypointDecisions(document.Decisions),
		Strategy:    document.Strategy,
	}
}

func stripRoutingRecipeModelSelection(decision *Decision) {
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
