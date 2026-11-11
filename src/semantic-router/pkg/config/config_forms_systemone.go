package config

import (
	"slices"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
)

// systemOneForms are the names the systemone signal, its conditions and the
// systemone selector had before: decision. A configuration that still uses
// them loads as systemone, with a warning.
var systemOneForms = []ConfigForm{
	{
		Code:  CodeRenamedSignalType,
		Paths: RoutingScopes("signals.decision"),
		Describe: func(at FormMatch) (string, string) {
			return "the decision signal is named systemone", strings.TrimSuffix(at.Path, "decision") + SignalTypeSystemOne
		},
		Renamed: true,
		Migrate: func(m *Migration, at FormMatch) error {
			return m.Rename(at.Parent, "decision", SignalTypeSystemOne)
		},
	},
	{
		Code: CodeRenamedSignalType,
		Paths: slices.Concat(
			RoutingScopes("decisions[].rules.**"),
			RoutingScopes("signals.complexity[].composer.**"),
			RoutingScopes("projections.scores[].inputs[]"),
		),
		Match:   hasType("decision"),
		What:    "condition type decision is named systemone",
		Becomes: "type: systemone",
		Renamed: true,
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Node, "type", Scalar(SignalTypeSystemOne))
			return nil
		},
	},
	{
		Code:  CodeRenamedAlgorithmType,
		Paths: RoutingScopes("decisions[].algorithm"),
		Match: func(node *yaml.Node, path string) bool {
			return hasType("decision")(node, path) || mappingValue(node, "decision") != nil
		},
		What:    "the decision selector is named systemone",
		Becomes: "algorithm.type: systemone and an algorithm.systemone block",
		Renamed: true,
		Migrate: func(m *Migration, at FormMatch) error {
			if scalarAt(at.Node, "type") == "decision" {
				m.Set(at.Node, "type", Scalar(DecisionAlgorithmSystemOne))
			}
			return m.Rename(at.Node, "decision", DecisionAlgorithmSystemOne)
		},
	},
}

// CodeSystemOneSchema is the migrate note of a systemone deployment that gets
// the schema of vllm-sr serve.
const CodeSystemOneSchema = "systemone_schema"

// systemOneSchemaForms run in `vllm-sr config migrate` only, after every
// other form. A systemone deployment that names no schema speaks the
// published standard, systemone/0.2; before it could name one, every such
// deployment spoke the model runtime's own contract. One whose model is a
// vllm-sr model (vllm-sr/...) asks vllm-sr serve, so migrate gives it that
// API's schema, vllm-sr/5.0, which keeps its batches, presets and card.
var systemOneSchemaForms = []ConfigForm{
	{
		Code:  CodeSystemOneSchema,
		Paths: []string{"global.model_catalog.deployments.*"},
		Match: func(node *yaml.Node, _ string) bool {
			return node.Kind == yaml.MappingNode && scalarAt(node, "provider") == SystemOneProvider &&
				mappingValue(node, "schema") == nil && strings.HasPrefix(scalarAt(node, "model"), "vllm-sr/")
		},
		What:    "a systemone deployment of a vllm-sr model names no schema, so it would speak the published standard, systemone/0.2",
		Becomes: "schema: vllm-sr/5.0, the API vllm-sr serve speaks",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Node, "schema", Scalar(string(systemonecodec.SchemaVLLMSR50)))
			return nil
		},
	},
	{
		Code:  CodeSystemOneSchema,
		Paths: []string{"providers.models[]"},
		Match: func(node *yaml.Node, _ string) bool {
			return node.Kind == yaml.MappingNode && scalarAt(node, "api_format") == APIFormatSystemOne &&
				mappingValue(node, "schema") == nil && mappingValue(node, "deployment") == nil &&
				(strings.HasPrefix(scalarAt(node, "provider_model_id"), "vllm-sr/") || strings.HasPrefix(scalarAt(node, "name"), "vllm-sr/"))
		},
		What:    "a systemone provider model of a vllm-sr model names no schema, so its backends would be spoken to in the published standard, systemone/0.2",
		Becomes: "schema: vllm-sr/5.0, the API vllm-sr serve speaks",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Set(at.Node, "schema", Scalar(string(systemonecodec.SchemaVLLMSR50)))
			return nil
		},
	},
}

// hasType matches a mapping whose type is value.
func hasType(value string) func(*yaml.Node, string) bool {
	return func(node *yaml.Node, _ string) bool {
		return node.Kind == yaml.MappingNode && scalarAt(node, "type") == value
	}
}
