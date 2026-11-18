package config

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Plugins layer by type: a decision's plugin overrides its recipe's plugin of
// the same type, which overrides the plugin of global.plugins. Within a type
// the layers merge field by field: mappings key by key, and any other value
// (a list, a scalar) replaces the one below it.

// validatePluginLayer checks one layer of default plugins: each type once.
func validatePluginLayer(path string, plugins []DecisionPlugin) error {
	seen := make(map[string]int, len(plugins))
	for index, plugin := range plugins {
		pluginType := NormalizeDecisionPluginType(plugin.Type)
		if pluginType == "" {
			return fmt.Errorf("%s[%d].type cannot be empty", path, index)
		}
		if previous, exists := seen[pluginType]; exists {
			return fmt.Errorf("%s[%d]: plugin %q is already %s[%d]; declare each type once", path, index, pluginType, path, previous)
		}
		seen[pluginType] = index
	}
	return nil
}

// layerRecipePlugins sets a recipe's default plugins (global.plugins layered
// with the recipe's) and the effective plugins of each of its decisions. The
// defaults are Chat plugins: a native decision keeps only its own.
func layerRecipePlugins(global []DecisionPlugin, recipe *RoutingRecipe) error {
	path := recipePath(string(recipe.Name))
	defaults, err := layerPlugins(global, recipe.Profile.Plugins)
	if err != nil {
		return fmt.Errorf("%s.plugins: %w", path, err)
	}
	recipe.Profile.DefaultPlugins = defaults
	if len(defaults) == 0 {
		for i := range recipe.Profile.Decisions {
			if decision := &recipe.Profile.Decisions[i]; decision.pluginsLayered {
				decision.Plugins = decision.declaredPlugins
				decision.declaredPlugins, decision.pluginsLayered = nil, false
			}
		}
		return nil
	}
	for i := range recipe.Profile.Decisions {
		decision := &recipe.Profile.Decisions[i]
		if decision.Algorithm.IsNative() {
			continue
		}
		declared := decision.DeclaredPlugins()
		effective, err := layerPlugins(defaults, declared)
		if err != nil {
			return fmt.Errorf("%s.decisions[%s].plugins: %w", path, decision.Name, err)
		}
		decision.declaredPlugins = declared
		decision.pluginsLayered = true
		decision.Plugins = effective
	}
	return nil
}

// LayerPlugins sets the effective plugins of a programmatically assembled
// config as the loader does: each recipe's default plugins, and each of its
// decisions' plugins, layered over GlobalPlugins. It may run again after the
// plugins of a layer change.
func (c *RouterConfig) LayerPlugins() error {
	if len(c.Recipes) == 0 {
		recipe := c.DefaultRecipe()
		if err := layerRecipePlugins(c.GlobalPlugins, recipe); err != nil {
			return err
		}
		c.DefaultPlugins, c.Decisions = recipe.Profile.DefaultPlugins, recipe.Profile.Decisions
		return nil
	}
	for i := range c.Recipes {
		if err := layerRecipePlugins(c.GlobalPlugins, &c.Recipes[i]); err != nil {
			return err
		}
	}
	c.mirrorDefaultRecipe()
	return nil
}

// layerPlugins returns base with each plugin of over merged into the base
// plugin of the same type, or appended after them.
func layerPlugins(base, over []DecisionPlugin) ([]DecisionPlugin, error) {
	if len(over) == 0 {
		return copyDecisionPlugins(base), nil
	}
	layered := copyDecisionPlugins(base)
	for _, plugin := range over {
		index := pluginIndex(layered, plugin.Type)
		if index < 0 {
			layered = append(layered, plugin)
			continue
		}
		configuration, err := mergePluginConfiguration(layered[index].Configuration, plugin.Configuration)
		if err != nil {
			return nil, fmt.Errorf("plugin %q: %w", plugin.Type, err)
		}
		layered[index] = DecisionPlugin{Type: plugin.Type, Configuration: configuration}
	}
	return layered, nil
}

func pluginIndex(plugins []DecisionPlugin, pluginType string) int {
	normalized := NormalizeDecisionPluginType(pluginType)
	for index, plugin := range plugins {
		if NormalizeDecisionPluginType(plugin.Type) == normalized {
			return index
		}
	}
	return -1
}

func mergePluginConfiguration(base, over *StructuredPayload) (*StructuredPayload, error) {
	if over.IsEmpty() {
		return base, nil
	}
	if base.IsEmpty() {
		return over, nil
	}
	var below, above interface{}
	if err := json.Unmarshal(base.Raw, &below); err != nil {
		return nil, err
	}
	if err := json.Unmarshal(over.Raw, &above); err != nil {
		return nil, err
	}
	return NewStructuredPayload(mergeFields(below, above))
}

// mergeFields merges above into below: a mapping key by key, anything else
// replaced.
func mergeFields(below, above interface{}) interface{} {
	belowMap, belowIsMap := below.(map[string]interface{})
	aboveMap, aboveIsMap := above.(map[string]interface{})
	if !belowIsMap || !aboveIsMap {
		return above
	}
	merged := make(map[string]interface{}, len(belowMap)+len(aboveMap))
	for key, value := range belowMap {
		merged[key] = value
	}
	for key, value := range aboveMap {
		merged[key] = mergeFields(merged[key], value)
	}
	return merged
}

// RequestPlugins are the plugins a request runs: its decision's effective
// plugins, or, for a request that matches no decision, its recipe's default
// plugins.
func (c *RouterConfig) RequestPlugins(decision *Decision) []DecisionPlugin {
	if decision != nil {
		return decision.Plugins
	}
	if c == nil {
		return nil
	}
	return c.DefaultPlugins
}

// RequestPlugin is the request's plugin of a type, nil without one.
func (c *RouterConfig) RequestPlugin(decision *Decision, pluginType string) *DecisionPlugin {
	plugins := c.RequestPlugins(decision)
	if index := pluginIndex(plugins, pluginType); index >= 0 {
		return &plugins[index]
	}
	return nil
}

// decodePlugin decodes the plugin of a type in plugins, nil without one.
func decodePlugin[T any](plugins []DecisionPlugin, pluginType string, result *T) *T {
	index := pluginIndex(plugins, pluginType)
	if index < 0 || plugins[index].Configuration == nil {
		return nil
	}
	if err := UnmarshalPluginConfig(plugins[index].Configuration, result); err != nil {
		logging.Errorf("Failed to unmarshal %s config: %v", pluginType, err)
		return nil
	}
	return result
}

// visitReachablePlugins calls visit with the plugins of every request: a
// passthrough request's global.plugins, and for each reachable recipe its
// default plugins, for a request that matches no decision, and each
// decision's.
func (c *RouterConfig) visitReachablePlugins(visit func([]DecisionPlugin)) {
	c.visitPlugins(c.ReachableRoutingRecipes(), visit)
}

// VisitEveryPlugins is visitReachablePlugins over every recipe, reachable or
// not.
func (c *RouterConfig) VisitEveryPlugins(visit func([]DecisionPlugin)) {
	recipes := make([]*RoutingRecipe, 0, len(c.Recipes))
	for i := range c.Recipes {
		recipes = append(recipes, &c.Recipes[i])
	}
	if len(recipes) == 0 {
		recipes = append(recipes, c.DefaultRecipe())
	}
	c.visitPlugins(recipes, visit)
}

func (c *RouterConfig) visitPlugins(recipes []*RoutingRecipe, visit func([]DecisionPlugin)) {
	visit(c.GlobalPlugins)
	for _, recipe := range recipes {
		visit(recipe.Profile.DefaultPlugins)
		for i := range recipe.Profile.Decisions {
			visit(recipe.Profile.Decisions[i].Plugins)
		}
	}
}

// DeclaredPlugins are the plugins the decision itself declares, without the
// recipe and global layers its Plugins include.
func (d Decision) DeclaredPlugins() []DecisionPlugin {
	if d.pluginsLayered {
		return copyDecisionPlugins(d.declaredPlugins)
	}
	return copyDecisionPlugins(d.Plugins)
}

func copyDecisionPlugins(plugins []DecisionPlugin) []DecisionPlugin {
	if len(plugins) == 0 {
		return nil
	}
	return append([]DecisionPlugin(nil), plugins...)
}
