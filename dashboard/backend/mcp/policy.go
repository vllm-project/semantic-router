package mcp

import (
	"errors"
	"net"
	"net/url"
	"os"
	"strings"
)

var (
	ErrServerExists          = errors.New("MCP server already exists")
	ErrServerNotFound        = errors.New("MCP server not found")
	ErrStdioTransportBlocked = errors.New("stdio MCP transport is disabled by runtime policy")
	ErrManagerClosed         = errors.New("MCP manager is closed")
	ErrLocalOnlyAddress      = errors.New("local-only MCP URL must use a literal loopback address or localhost")
)

// ManagerOptions defines process-level MCP capabilities. The zero value is
// intentionally production-safe: local subprocess execution is disabled.
type ManagerOptions struct {
	AllowStdio bool
}

func (m *Manager) validateRuntimePolicy(config *ServerConfig) error {
	if config == nil {
		return errors.New("MCP server config is required")
	}
	if config.Transport == TransportStdio && !m.allowStdio {
		return ErrStdioTransportBlocked
	}
	return ValidateConnectionSecurity(config)
}

// ValidateConnectionSecurity enforces transport security flags independently
// of the HTTP handler so persisted and programmatic configs cannot bypass them.
func ValidateConnectionSecurity(config *ServerConfig) error {
	if config == nil || config.Transport != TransportStreamableHTTP ||
		config.Security == nil || !config.Security.LocalOnly {
		return nil
	}
	parsed, err := url.Parse(strings.TrimSpace(config.Connection.URL))
	if err != nil || parsed.Hostname() == "" ||
		(parsed.Scheme != "http" && parsed.Scheme != "https") {
		return ErrLocalOnlyAddress
	}
	host := strings.TrimSuffix(strings.ToLower(parsed.Hostname()), ".")
	if host == "localhost" {
		return nil
	}
	ip := net.ParseIP(host)
	if ip == nil || !ip.IsLoopback() {
		return ErrLocalOnlyAddress
	}
	return nil
}

// AllowsStdio reports the immutable process-level subprocess policy.
func (m *Manager) AllowsStdio() bool {
	return m != nil && m.allowStdio
}

// stdioBaseEnvironment deliberately excludes Dashboard credentials and other
// ambient process state. An administrator may explicitly configure additional
// variables on the MCP server, but a child never inherits secrets merely
// because they were present in the Dashboard environment.
func stdioBaseEnvironment() []string {
	allowed := [...]string{
		"HOME",
		"LANG",
		"LC_ALL",
		"LOGNAME",
		"PATH",
		"SHELL",
		"TEMP",
		"TMP",
		"TMPDIR",
		"USER",
		"XDG_CACHE_HOME",
		"XDG_CONFIG_HOME",
	}

	environment := make([]string, 0, len(allowed))
	for _, name := range allowed {
		if value, ok := os.LookupEnv(name); ok {
			environment = append(environment, name+"="+value)
		}
	}
	return environment
}

func cloneServerConfig(config *ServerConfig) *ServerConfig {
	if config == nil {
		return nil
	}
	clone := *config
	clone.Connection.Args = append([]string(nil), config.Connection.Args...)
	clone.Connection.Env = cloneStringMap(config.Connection.Env)
	clone.Connection.Headers = cloneStringMap(config.Connection.Headers)
	if config.Security != nil {
		securityClone := *config.Security
		securityClone.AllowedOrigins = append([]string(nil), config.Security.AllowedOrigins...)
		if config.Security.OAuth != nil {
			oauthClone := *config.Security.OAuth
			oauthClone.Scopes = append([]string(nil), config.Security.OAuth.Scopes...)
			securityClone.OAuth = &oauthClone
		}
		clone.Security = &securityClone
	}
	if config.Options != nil {
		optionsClone := *config.Options
		clone.Options = &optionsClone
	}
	return &clone
}

func cloneStringMap(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	clone := make(map[string]string, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func cloneToolDefinitions(source []ToolDefinition) []ToolDefinition {
	if source == nil {
		return nil
	}
	clone := make([]ToolDefinition, len(source))
	for index, tool := range source {
		clone[index] = tool
		clone[index].InputSchema = append([]byte(nil), tool.InputSchema...)
	}
	return clone
}
