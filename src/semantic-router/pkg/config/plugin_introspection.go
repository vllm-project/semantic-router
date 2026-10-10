package config

import "reflect"

// DecodeDecisionPlugin validates and decodes a plugin through the canonical
// registry. Management consumers must not maintain another plugin inventory.
func DecodeDecisionPlugin(plugin DecisionPlugin) (interface{}, error) {
	return DecodeDecisionPluginAt(PluginAt{Decision: "plugin", Type: plugin.Type}, plugin)
}

// DecisionPluginEnabled reports the declared policy activation. It does not
// imply runtime readiness or that a request will meet the plugin's conditions.
// Registered typed payloads with no enabled field are active when declared;
// payloads with conditional defaults own them through IsEnabled().
func DecisionPluginEnabled(payload interface{}) bool {
	if policy, ok := payload.(interface{ IsEnabled() bool }); ok {
		return policy.IsEnabled()
	}
	value := reflect.ValueOf(payload)
	if value.Kind() != reflect.Pointer || value.IsNil() {
		return false
	}
	field := value.Elem().FieldByName("Enabled")
	if !field.IsValid() {
		return true
	}
	if field.Kind() == reflect.Pointer {
		return !field.IsNil() && field.Elem().Bool()
	}
	return field.Kind() == reflect.Bool && field.Bool()
}
