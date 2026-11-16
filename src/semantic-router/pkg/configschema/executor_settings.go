package configschema

import (
	"github.com/invopop/jsonschema"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Executors share the algorithm's budget and on_error; a selector calls one
// candidate and takes neither. Escalate's System One stages run within a
// budget and name their own models, so they need a budget and take no
// minimum_candidates. Publish the same rules as the runtime validator.
func addExecutorSettingConditions(root *jsonschema.Schema) {
	owner := root.Definitions["AlgorithmConfig"]
	if owner == nil {
		return
	}
	var selectors []any
	for _, entry := range routerconfig.DecisionAlgorithmCatalog() {
		if entry.Kind == routerconfig.AlgorithmKindSelector {
			selectors = append(selectors, entry.Type)
		}
	}
	selector := jsonschema.NewProperties()
	selector.Set("type", &jsonschema.Schema{Enum: selectors})
	stages := jsonschema.NewProperties()
	stages.Set("type", &jsonschema.Schema{Const: routerconfig.DecisionAlgorithmEscalate})
	stages.Set(routerconfig.DecisionAlgorithmEscalate, &jsonschema.Schema{Required: []string{"stages"}})
	owner.AllOf = append(owner.AllOf,
		&jsonschema.Schema{
			If: &jsonschema.Schema{Properties: selector, Required: []string{"type"}},
			Then: &jsonschema.Schema{Not: &jsonschema.Schema{AnyOf: []*jsonschema.Schema{
				{Required: []string{"budget"}},
				{Required: []string{"on_error"}},
			}}},
		},
		&jsonschema.Schema{
			If:   &jsonschema.Schema{Properties: stages, Required: []string{"type", routerconfig.DecisionAlgorithmEscalate}},
			Then: &jsonschema.Schema{Required: []string{"budget"}, Not: &jsonschema.Schema{Required: []string{"minimum_candidates"}}},
		},
	)
}
