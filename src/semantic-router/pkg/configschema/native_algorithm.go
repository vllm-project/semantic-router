package configschema

import "github.com/invopop/jsonschema"

// Native execution has one budget per selected algorithm, never per recipe.
// Publish the same conditional ownership rule as the runtime validator.
func addNativeAlgorithmConditions(root *jsonschema.Schema) {
	owner := root.Definitions["AlgorithmConfig"]
	if owner == nil {
		return
	}
	properties := jsonschema.NewProperties()
	properties.Set("type", &jsonschema.Schema{Const: "cascade"})
	required := []string{"budget", "quality", "stages"}
	var foreignFields []*jsonschema.Schema
	for _, name := range required {
		foreignFields = append(foreignFields, &jsonschema.Schema{Required: []string{name}})
	}
	owner.AllOf = append(owner.AllOf, &jsonschema.Schema{
		If:   &jsonschema.Schema{Properties: properties, Required: []string{"type"}},
		Then: &jsonschema.Schema{Required: required, Not: &jsonschema.Schema{Required: []string{"minimum_candidates"}}},
		Else: &jsonschema.Schema{Not: &jsonschema.Schema{AnyOf: foreignFields}},
	})
}
