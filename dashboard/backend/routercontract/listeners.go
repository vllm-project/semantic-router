package routercontract

import (
	"os"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// ListenerEndpoint is the dashboard-owned view of a router listener.
type ListenerEndpoint struct {
	Address string
	Port    int
}

// ReadFirstListenerEndpoint reads the first listener endpoint from the router
// config shapes that the dashboard needs to discover a local model gateway.
func ReadFirstListenerEndpoint(configPath string) (ListenerEndpoint, bool, error) {
	data, err := os.ReadFile(strings.TrimSpace(configPath))
	if err != nil {
		return ListenerEndpoint{}, false, err
	}

	var config map[string]any
	if err := yaml.Unmarshal(data, &config); err != nil {
		return ListenerEndpoint{}, false, err
	}

	for _, listener := range extractListenerMaps(config) {
		port, ok := endpointPort(listener["port"])
		if !ok {
			continue
		}
		return ListenerEndpoint{
			Address: endpointString(listener["address"]),
			Port:    port,
		}, true, nil
	}

	return ListenerEndpoint{}, false, nil
}

func extractListenerMaps(config map[string]any) []map[string]any {
	listeners := make([]map[string]any, 0)
	appendListeners := func(value any) {
		entries, ok := value.([]any)
		if !ok {
			return
		}
		for _, entry := range entries {
			if listener, ok := endpointStringMap(entry); ok {
				listeners = append(listeners, listener)
			}
		}
	}

	appendListeners(config["listeners"])
	if apiServer, ok := endpointStringMap(config["api_server"]); ok {
		appendListeners(apiServer["listeners"])
	}
	return listeners
}

func endpointStringMap(value any) (map[string]any, bool) {
	switch typed := value.(type) {
	case map[string]any:
		return typed, true
	case map[any]any:
		normalized := make(map[string]any, len(typed))
		for key, nested := range typed {
			textKey, ok := key.(string)
			if !ok {
				continue
			}
			normalized[textKey] = nested
		}
		return normalized, true
	default:
		return nil, false
	}
}

func endpointString(value any) string {
	text, ok := value.(string)
	if !ok {
		return ""
	}
	return strings.TrimSpace(text)
}

func endpointPort(value any) (int, bool) {
	switch typed := value.(type) {
	case int:
		return validateEndpointPort(typed)
	case int64:
		return validateEndpointPort(int(typed))
	case float64:
		port := int(typed)
		if float64(port) == typed {
			return validateEndpointPort(port)
		}
	case string:
		port, err := strconv.Atoi(strings.TrimSpace(typed))
		if err == nil {
			return validateEndpointPort(port)
		}
	}
	return 0, false
}

func validateEndpointPort(port int) (int, bool) {
	if port >= 1 && port <= 65535 {
		return port, true
	}
	return 0, false
}
