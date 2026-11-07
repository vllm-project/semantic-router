package mcp

import "errors"

// ErrInvalidServerConfig marks a configuration the Dashboard cannot store or
// dial. The message stays caller-facing; callers match the fault with
// errors.Is and answer 400, the status create has always answered.
var ErrInvalidServerConfig = errors.New("invalid MCP server configuration")

// invalidServerConfig carries the caller-facing message while matching the
// sentinel, so the text an entry surface answers with does not change when
// the check moves into this package.
type invalidServerConfig struct{ message string }

func (e invalidServerConfig) Error() string { return e.message }

func (e invalidServerConfig) Unwrap() error { return ErrInvalidServerConfig }

// ValidateServerConfig checks the invariants every entry surface enforces: a
// name, a known transport, security settings the client can enforce, and the
// connection field that transport needs. The collection test path runs it as
// well, so a configuration that cannot be created cannot be tested (#4338).
func ValidateServerConfig(config *ServerConfig) error {
	if err := validateServerStructure(config); err != nil {
		return err
	}
	return ValidateSecurity(config.Security)
}

// validateServerStructure checks the fields a stored configuration cannot be
// without. The update merge runs it on the merged result: a partial body
// starts from the request, so without it the merge strips the transport and
// the connection field from what gets persisted (#4338). The security policy
// stays at the API door, where the request is vetted, so persisting a shape
// that was stored before the policy does not have to clear it first.
func validateServerStructure(config *ServerConfig) error {
	if config == nil {
		return invalidServerConfig{message: "MCP server configuration is required"}
	}
	if config.Name == "" {
		return invalidServerConfig{message: "Name is required"}
	}
	if config.Transport == "" {
		return invalidServerConfig{message: "Transport is required"}
	}
	if config.Transport != TransportStdio && config.Transport != TransportStreamableHTTP {
		return invalidServerConfig{
			message: "Invalid transport type. Must be 'stdio' or 'streamable-http'",
		}
	}
	if config.Transport == TransportStdio && config.Connection.Command == "" {
		return invalidServerConfig{message: "Command is required for stdio transport"}
	}
	if config.Transport == TransportStreamableHTTP && config.Connection.URL == "" {
		return invalidServerConfig{message: "URL is required for streamable-http transport"}
	}
	return nil
}
