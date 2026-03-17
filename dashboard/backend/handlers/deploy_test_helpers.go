package handlers

import "fmt"

func deepMerge(dst, src map[string]interface{}) map[string]interface{} {
	for key, srcVal := range src {
		if dstVal, exists := dst[key]; exists {
			if dstMap, ok := dstVal.(map[string]interface{}); ok {
				if srcMap, ok := srcVal.(map[string]interface{}); ok {
					dst[key] = deepMerge(dstMap, srcMap)
					continue
				}
			}
			if dstMap, ok := toStringKeyMap(dstVal); ok {
				if srcMap, ok := toStringKeyMap(srcVal); ok {
					dst[key] = deepMerge(dstMap, srcMap)
					continue
				}
			}
		}
		dst[key] = srcVal
	}
	return dst
}

func toStringKeyMap(v interface{}) (map[string]interface{}, bool) {
	switch m := v.(type) {
	case map[string]interface{}:
		return m, true
	case map[interface{}]interface{}:
		result := make(map[string]interface{}, len(m))
		for k, val := range m {
			result[fmt.Sprintf("%v", k)] = val
		}
		return result, true
	default:
		return nil, false
	}
}
