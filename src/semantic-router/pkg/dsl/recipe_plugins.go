package dsl

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// parseRecipePlugins reads ROUTING plugins: an array of {type, configuration}
// objects, the plugins of the scope's recipe.
func parseRecipePlugins(value Value) ([]config.DecisionPlugin, error) {
	array, ok := value.(ArrayValue)
	if !ok {
		return nil, fmt.Errorf("must be an array of {type, configuration} objects")
	}
	payload, err := json.Marshal(valueToInterface(array))
	if err != nil {
		return nil, err
	}
	var plugins []config.DecisionPlugin
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&plugins); err != nil {
		return nil, err
	}
	for index, plugin := range plugins {
		if strings.TrimSpace(plugin.Type) == "" {
			return nil, fmt.Errorf("plugins[%d].type cannot be empty", index)
		}
	}
	return plugins, nil
}

// formatRecipePlugins writes recipe plugins in the DSL value syntax.
func formatRecipePlugins(plugins []config.DecisionPlugin) string {
	payload, _ := json.Marshal(plugins)
	var items []interface{}
	_ = json.Unmarshal(payload, &items)
	return formatPluginConfigValue(items)
}
