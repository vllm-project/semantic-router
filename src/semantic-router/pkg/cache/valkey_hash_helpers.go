package cache

import "fmt"

func parseValkeyHashFields(raw any) map[string]string {
	fields := make(map[string]string)
	switch value := raw.(type) {
	case map[string]string:
		return value
	case map[string]interface{}:
		for key, nested := range value {
			fields[key] = fmt.Sprint(nested)
		}
	case map[interface{}]interface{}:
		for key, nested := range value {
			fields[fmt.Sprint(key)] = fmt.Sprint(nested)
		}
	case []interface{}:
		for index := 0; index+1 < len(value); index += 2 {
			fields[fmt.Sprint(value[index])] = fmt.Sprint(value[index+1])
		}
	case []string:
		for index := 0; index+1 < len(value); index += 2 {
			fields[value[index]] = value[index+1]
		}
	}
	return fields
}

func valkeyFallbackBytes(raw any) []byte {
	switch value := raw.(type) {
	case string:
		return []byte(value)
	case []byte:
		return append([]byte(nil), value...)
	default:
		return []byte(fmt.Sprint(value))
	}
}
