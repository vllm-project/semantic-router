package config

import (
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
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

// hasType matches a mapping whose type is value.
func hasType(value string) func(*yaml.Node, string) bool {
	return func(node *yaml.Node, _ string) bool {
		return node.Kind == yaml.MappingNode && scalarAt(node, "type") == value
	}
}
