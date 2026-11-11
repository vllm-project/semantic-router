package config

import (
	"fmt"
	"strings"
)

// removedBackendRefFields are backend ref settings the Router no longer has.
// Both opted a backend in to tolerating response fields its wire format does
// not define, which every backend now does.
var removedBackendRefFields = []string{"compat_gateway", "openai_compat_gateway"}

// rejectRemovedBackendRefFields refuses a backend ref that still sets a removed
// field with a one-line hint instead of the generic unknown-field error.
func rejectRemovedBackendRefFields(raw map[string]interface{}) error {
	models, _ := nestedStringMap(raw["providers"])["models"].([]interface{})
	for modelIndex, rawModel := range models {
		model := nestedStringMap(rawModel)
		refs, _ := model["backend_refs"].([]interface{})
		for refIndex, rawRef := range refs {
			ref := nestedStringMap(rawRef)
			for _, field := range removedBackendRefFields {
				if _, present := ref[field]; present {
					return fmt.Errorf(
						"providers.models[%s].backend_refs[%d]: %s was removed; gateway response fields are accepted by default, so delete it",
						removedFieldModelLabel(model, modelIndex), refIndex, field,
					)
				}
			}
		}
	}
	return nil
}

func removedFieldModelLabel(model map[string]interface{}, index int) string {
	if name, ok := model["name"].(string); ok && strings.TrimSpace(name) != "" {
		return name
	}
	return fmt.Sprint(index)
}
