package config

import "testing"

func TestNormalizeResponseCachePluginAliases(t *testing.T) {
	for _, alias := range []string{
		DecisionPluginResponseCache,
		"semantic-cache",
		"semantic_cache",
		"response-cache",
	} {
		t.Run(alias, func(t *testing.T) {
			plugin := map[interface{}]interface{}{"type": alias}
			raw := map[string]interface{}{
				"routing": map[interface{}]interface{}{
					"decisions": []interface{}{
						map[interface{}]interface{}{"plugins": []interface{}{plugin}},
					},
				},
			}
			if err := normalizeResponseCacheAliases(raw); err != nil {
				t.Fatalf("normalizeResponseCacheAliases() error = %v", err)
			}
			if got := plugin["type"]; got != DecisionPluginResponseCache {
				t.Fatalf("plugin type = %v, want %s", got, DecisionPluginResponseCache)
			}
		})
	}
}

func TestNormalizeResponseCacheRejectsDuplicateAliases(t *testing.T) {
	raw := map[string]interface{}{
		"routing": map[interface{}]interface{}{
			"decisions": []interface{}{
				map[interface{}]interface{}{
					"plugins": []interface{}{
						map[interface{}]interface{}{"type": "semantic-cache"},
						map[interface{}]interface{}{"type": "response_cache"},
					},
				},
			},
		},
	}
	if err := normalizeResponseCacheAliases(raw); err == nil {
		t.Fatal("normalizeResponseCacheAliases() accepted duplicate aliases")
	}
}

func TestNormalizeResponseCacheStoreAliasAndConflict(t *testing.T) {
	stores := map[interface{}]interface{}{"semantic_cache": map[interface{}]interface{}{"enabled": true}}
	raw := map[string]interface{}{
		"global": map[interface{}]interface{}{"stores": stores},
	}
	if err := normalizeResponseCacheAliases(raw); err != nil {
		t.Fatalf("normalizeResponseCacheAliases() error = %v", err)
	}
	if _, ok := stores["response_cache"]; !ok {
		t.Fatal("canonical response_cache store was not created")
	}
	if _, ok := stores["semantic_cache"]; ok {
		t.Fatal("deprecated semantic_cache store was retained")
	}

	stores["semantic_cache"] = map[interface{}]interface{}{"enabled": false}
	if err := normalizeResponseCacheAliases(raw); err == nil {
		t.Fatal("normalizeResponseCacheAliases() accepted both store names")
	}
}
