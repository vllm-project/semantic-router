package mcp

import (
	"errors"
	"fmt"
	"strings"
)

// ErrUnsupportedSecurity marks security settings the Dashboard MCP client cannot enforce.
var ErrUnsupportedSecurity = errors.New("MCP server security settings are not supported")

// ValidateSecurity fails closed; its error names fields only, so API clients may see it.
func ValidateSecurity(security *SecurityConfig) error {
	if security == nil {
		return nil
	}
	var fields []string
	if security.OAuth != nil {
		fields = append(fields, "security.oauth")
	}
	if security.LocalOnly {
		fields = append(fields, "security.local_only")
	}
	if len(security.AllowedOrigins) > 0 {
		fields = append(fields, "security.allowed_origins")
	}
	if len(fields) == 0 {
		return nil
	}
	return fmt.Errorf(
		`%w: %s. Remove them, or send "security": {} to clear stored settings`,
		ErrUnsupportedSecurity,
		strings.Join(fields, ", "),
	)
}
