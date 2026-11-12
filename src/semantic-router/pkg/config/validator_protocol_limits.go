package config

import "fmt"

// validateProtocolLimitContracts rejects overrides outside the supported range
// instead of clamping them.
func validateProtocolLimitContracts(cfg *RouterConfig) error {
	limits := cfg.ProtocolLimits
	if err := validateProtocolLimit("tool_name_bytes", limits.ToolNameBytes, MaxProtocolToolNameBytes); err != nil {
		return err
	}
	return validateProtocolLimit("tool_description_bytes", limits.ToolDescriptionBytes, MaxProtocolToolDescriptionBytes)
}

func validateProtocolLimit(field string, value, maximum int) error {
	if value < 0 || value > maximum {
		return fmt.Errorf("global.router.protocol_limits.%s: %d is out of range, use 1 to %d bytes or 0 for the default",
			field, value, maximum)
	}
	return nil
}
