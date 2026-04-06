package dsl

// YAMLValue/YAMLObject/YAMLList model the recursive YAML tree used by the DSL
// emit helpers without reopening raw map[string]interface{} field bags.
type (
	YAMLValue  = any
	YAMLObject = map[string]YAMLValue
	YAMLList   = []YAMLValue
)
