package config

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func normalizeResponseCacheAliases(raw map[string]interface{}) error {
	if err := normalizeResponseCacheStoreAlias(raw); err != nil {
		return err
	}
	return normalizeResponseCachePluginAliases(raw)
}

func normalizeResponseCacheStoreAlias(raw map[string]interface{}) error {
	global := nestedMutableMap(raw["global"])
	stores := nestedMutableMap(global.get("stores"))
	canonical, hasCanonical := stores.lookup("response_cache")
	legacy, hasLegacy := stores.lookup("semantic_cache")
	if hasCanonical && hasLegacy {
		return fmt.Errorf("global.stores.response_cache conflicts with deprecated global.stores.semantic_cache")
	}
	if hasLegacy {
		stores.set("response_cache", legacy)
		stores.delete("semantic_cache")
		logging.ComponentWarnEvent("config", "deprecated_response_cache_store_alias", map[string]interface{}{
			"deprecated": "global.stores.semantic_cache",
			"canonical":  "global.stores.response_cache",
		})
	}
	_ = canonical
	return nil
}

func normalizeResponseCachePluginAliases(value interface{}) error {
	switch typed := value.(type) {
	case map[string]interface{}:
		return normalizeResponseCacheStringMap(typed)
	case map[interface{}]interface{}:
		return normalizeResponseCacheInterfaceMap(typed)
	case []interface{}:
		return normalizeResponseCacheSlice(typed)
	}
	return nil
}

func normalizeResponseCacheStringMap(value map[string]interface{}) error {
	if plugins, ok := value["plugins"]; ok {
		if err := normalizePluginList(plugins); err != nil {
			return err
		}
	}
	for _, child := range value {
		if err := normalizeResponseCachePluginAliases(child); err != nil {
			return err
		}
	}
	return nil
}

func normalizeResponseCacheInterfaceMap(value map[interface{}]interface{}) error {
	if plugins, ok := value["plugins"]; ok {
		if err := normalizePluginList(plugins); err != nil {
			return err
		}
	}
	for _, child := range value {
		if err := normalizeResponseCachePluginAliases(child); err != nil {
			return err
		}
	}
	return nil
}

func normalizeResponseCacheSlice(value []interface{}) error {
	for _, child := range value {
		if err := normalizeResponseCachePluginAliases(child); err != nil {
			return err
		}
	}
	return nil
}

func normalizePluginList(raw interface{}) error {
	plugins, ok := raw.([]interface{})
	if !ok {
		return nil
	}
	seen := make(map[string]string, len(plugins))
	for _, rawPlugin := range plugins {
		plugin := nestedMutableMap(rawPlugin)
		rawType, _ := plugin.get("type").(string)
		normalized := NormalizeDecisionPluginType(rawType)
		if normalized == "" {
			continue
		}
		if previous, exists := seen[normalized]; exists {
			return fmt.Errorf("duplicate plugin %q via %q and %q", normalized, previous, rawType)
		}
		seen[normalized] = rawType
		if normalized != rawType {
			plugin.set("type", normalized)
			logging.ComponentWarnEvent("config", "deprecated_response_cache_plugin_alias", map[string]interface{}{
				"deprecated": rawType,
				"canonical":  normalized,
			})
		}
	}
	return nil
}

type mutableMap struct {
	stringMap map[string]interface{}
	anyMap    map[interface{}]interface{}
}

func nestedMutableMap(value interface{}) mutableMap {
	switch typed := value.(type) {
	case map[string]interface{}:
		return mutableMap{stringMap: typed}
	case map[interface{}]interface{}:
		return mutableMap{anyMap: typed}
	default:
		return mutableMap{}
	}
}

func (m mutableMap) lookup(key string) (interface{}, bool) {
	if m.stringMap != nil {
		value, ok := m.stringMap[key]
		return value, ok
	}
	if m.anyMap != nil {
		value, ok := m.anyMap[key]
		return value, ok
	}
	return nil, false
}

func (m mutableMap) get(key string) interface{} {
	value, _ := m.lookup(key)
	return value
}

func (m mutableMap) set(key string, value interface{}) {
	if m.stringMap != nil {
		m.stringMap[key] = value
	}
	if m.anyMap != nil {
		m.anyMap[key] = value
	}
}

func (m mutableMap) delete(key string) {
	if m.stringMap != nil {
		delete(m.stringMap, key)
	}
	if m.anyMap != nil {
		delete(m.anyMap, key)
	}
}
