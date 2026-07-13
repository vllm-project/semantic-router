package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"mime"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

const (
	mcpConfigBodyLimit = 64 << 10
	mcpToolBodyLimit   = 256 << 10
)

type mcpServerConfigView struct {
	ID          string             `json:"id"`
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	Transport   mcp.TransportType  `json:"transport"`
	Connection  mcpConnectionView  `json:"connection"`
	Enabled     bool               `json:"enabled"`
	Security    *mcpSecurityView   `json:"security,omitempty"`
	Options     *mcp.ServerOptions `json:"options,omitempty"`
}

// mcpConnectionView intentionally exposes no argument, environment, header,
// or host working-directory values. Safe display values are paired with
// presence flags so clients can distinguish an omitted secret from no config.
type mcpConnectionView struct {
	Command                    string `json:"command,omitempty"`
	URL                        string `json:"url,omitempty"`
	CommandConfigured          bool   `json:"command_configured,omitempty"`
	CommandRedacted            bool   `json:"command_redacted,omitempty"`
	ArgumentsConfigured        bool   `json:"arguments_configured,omitempty"`
	EnvironmentConfigured      bool   `json:"environment_configured,omitempty"`
	WorkingDirectoryConfigured bool   `json:"working_directory_configured,omitempty"`
	URLConfigured              bool   `json:"url_configured,omitempty"`
	URLRedacted                bool   `json:"url_redacted,omitempty"`
	HeadersConfigured          bool   `json:"headers_configured,omitempty"`
}

type mcpSecurityView struct {
	OAuth          *mcpOAuthView `json:"oauth,omitempty"`
	AllowedOrigins []string      `json:"allowed_origins,omitempty"`
	LocalOnly      bool          `json:"local_only,omitempty"`
}

type mcpOAuthView struct {
	ClientID                 string   `json:"client_id"`
	ClientSecretConfigured   bool     `json:"client_secret_configured,omitempty"`
	AuthorizationURL         string   `json:"authorization_url"`
	AuthorizationURLRedacted bool     `json:"authorization_url_redacted,omitempty"`
	TokenURL                 string   `json:"token_url"`
	TokenURLRedacted         bool     `json:"token_url_redacted,omitempty"`
	Scopes                   []string `json:"scopes,omitempty"`
	UsePKCE                  bool     `json:"use_pkce,omitempty"`
}

type mcpServerStateView struct {
	Config      mcpServerConfigView  `json:"config"`
	Status      mcp.ServerStatus     `json:"status"`
	Error       string               `json:"error,omitempty"`
	Tools       []mcp.ToolDefinition `json:"tools,omitempty"`
	ConnectedAt *time.Time           `json:"connected_at,omitempty"`
}

func newMCPServerConfigView(config *mcp.ServerConfig) mcpServerConfigView {
	if config == nil {
		return mcpServerConfigView{}
	}
	view := mcpServerConfigView{
		ID:          config.ID,
		Name:        config.Name,
		Description: config.Description,
		Transport:   config.Transport,
		Enabled:     config.Enabled,
		Options:     cloneMCPServerOptions(config.Options),
		Connection: mcpConnectionView{
			CommandConfigured:          config.Connection.Command != "",
			ArgumentsConfigured:        len(config.Connection.Args) != 0,
			EnvironmentConfigured:      len(config.Connection.Env) != 0,
			WorkingDirectoryConfigured: config.Connection.Cwd != "",
			URLConfigured:              config.Connection.URL != "",
			HeadersConfigured:          len(config.Connection.Headers) != 0,
		},
	}
	view.Connection.Command, view.Connection.CommandRedacted = safeMCPCommand(config.Connection.Command)
	view.Connection.URL, view.Connection.URLRedacted = safeMCPURL(config.Connection.URL)
	if config.Security != nil {
		view.Security = &mcpSecurityView{
			AllowedOrigins: append([]string(nil), config.Security.AllowedOrigins...),
			LocalOnly:      config.Security.LocalOnly,
		}
		if config.Security.OAuth != nil {
			view.Security.OAuth = &mcpOAuthView{
				ClientID:               config.Security.OAuth.ClientID,
				ClientSecretConfigured: config.Security.OAuth.ClientSecret != "",
				Scopes:                 append([]string(nil), config.Security.OAuth.Scopes...),
				UsePKCE:                config.Security.OAuth.UsePKCE,
			}
			view.Security.OAuth.AuthorizationURL, view.Security.OAuth.AuthorizationURLRedacted = safeMCPURL(config.Security.OAuth.AuthorizationURL)
			view.Security.OAuth.TokenURL, view.Security.OAuth.TokenURLRedacted = safeMCPURL(config.Security.OAuth.TokenURL)
		}
	}
	return view
}

func newMCPServerStateView(state *mcp.ServerState, exposeTools bool) mcpServerStateView {
	view := mcpServerStateView{
		Config:      newMCPServerConfigView(state.Config),
		Status:      state.Status,
		ConnectedAt: state.ConnectedAt,
	}
	if state.Error != "" {
		view.Error = "connection failed"
	}
	if exposeTools {
		view.Tools = append([]mcp.ToolDefinition(nil), state.Tools...)
	}
	return view
}

func safeMCPCommand(command string) (string, bool) {
	command = strings.TrimSpace(command)
	if command == "" {
		return "", false
	}
	base := filepath.Base(strings.ReplaceAll(command, "\\", "/"))
	return base, base != command
}

func safeMCPURL(raw string) (string, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", false
	}
	parsed, err := url.Parse(raw)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", true
	}
	redacted := parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != ""
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.ForceQuery = false
	parsed.Fragment = ""
	return parsed.String(), redacted
}

func cloneMCPServerOptions(options *mcp.ServerOptions) *mcp.ServerOptions {
	if options == nil {
		return nil
	}
	clone := *options
	return &clone
}

type mcpServerUpdateRequest struct {
	ID          *string                `json:"id,omitempty"`
	Name        *string                `json:"name,omitempty"`
	Description *string                `json:"description,omitempty"`
	Transport   *mcp.TransportType     `json:"transport,omitempty"`
	Connection  *mcpConnectionPatch    `json:"connection,omitempty"`
	Enabled     *bool                  `json:"enabled,omitempty"`
	Security    *mcpSecurityPatch      `json:"security,omitempty"`
	Options     *mcpServerOptionsPatch `json:"options,omitempty"`
}

type mcpConnectionPatch struct {
	Command        *string            `json:"command,omitempty"`
	ReplaceCommand bool               `json:"replace_command,omitempty"`
	Args           *[]string          `json:"args,omitempty"`
	Env            *map[string]string `json:"env,omitempty"`
	Cwd            *string            `json:"cwd,omitempty"`
	URL            *string            `json:"url,omitempty"`
	ReplaceURL     bool               `json:"replace_url,omitempty"`
	Headers        *map[string]string `json:"headers,omitempty"`
}

type mcpSecurityPatch struct {
	OAuth          *mcpOAuthPatch `json:"oauth,omitempty"`
	AllowedOrigins *[]string      `json:"allowed_origins,omitempty"`
	LocalOnly      *bool          `json:"local_only,omitempty"`
}

type mcpOAuthPatch struct {
	ClientID         *string   `json:"client_id,omitempty"`
	ClientSecret     *string   `json:"client_secret,omitempty"`
	AuthorizationURL *string   `json:"authorization_url,omitempty"`
	TokenURL         *string   `json:"token_url,omitempty"`
	Scopes           *[]string `json:"scopes,omitempty"`
	UsePKCE          *bool     `json:"use_pkce,omitempty"`
}

type mcpServerOptionsPatch struct {
	AutoReconnect     *bool `json:"auto_reconnect,omitempty"`
	ReconnectInterval *int  `json:"reconnect_interval,omitempty"`
	Timeout           *int  `json:"timeout,omitempty"`
	MaxRetries        *int  `json:"max_retries,omitempty"`
}

func applyMCPServerUpdate(existing *mcp.ServerConfig, patch *mcpServerUpdateRequest) (*mcp.ServerConfig, error) {
	if existing == nil || patch == nil {
		return nil, errors.New("server update is required")
	}
	updated := *existing
	updated.Connection.Args = append([]string(nil), existing.Connection.Args...)
	updated.Connection.Env = cloneMCPStringMap(existing.Connection.Env)
	updated.Connection.Headers = cloneMCPStringMap(existing.Connection.Headers)
	updated.Security = cloneMCPSecurity(existing.Security)
	updated.Options = cloneMCPServerOptions(existing.Options)

	if patch.ID != nil && *patch.ID != "" && *patch.ID != existing.ID {
		return nil, errors.New("server id does not match request path")
	}
	if patch.Name != nil {
		updated.Name = *patch.Name
	}
	if patch.Description != nil {
		updated.Description = *patch.Description
	}
	if patch.Transport != nil && *patch.Transport != updated.Transport {
		updated.Transport = *patch.Transport
		updated.Connection = mcp.ConnectionConfig{}
	}
	if patch.Enabled != nil {
		updated.Enabled = *patch.Enabled
	}
	if patch.Connection != nil {
		applyMCPConnectionPatch(&updated.Connection, &existing.Connection, patch.Connection)
	}
	if patch.Security != nil {
		if updated.Security == nil {
			updated.Security = &mcp.SecurityConfig{}
		}
		applyMCPSecurityPatch(updated.Security, patch.Security)
	}
	if patch.Options != nil {
		if updated.Options == nil {
			updated.Options = &mcp.ServerOptions{}
		}
		applyMCPOptionsPatch(updated.Options, patch.Options)
	}
	return &updated, nil
}

func resolveMCPServerTestConfig(manager *mcp.Manager, patch *mcpServerUpdateRequest) (*mcp.ServerConfig, error) {
	if patch == nil {
		return nil, errors.New("server test config is required")
	}
	var base *mcp.ServerConfig
	if patch.ID != nil && manager != nil {
		if existing, ok := manager.GetServer(*patch.ID); ok {
			base = existing
		}
	}
	if base == nil {
		base = &mcp.ServerConfig{}
		if patch.ID != nil {
			base.ID = *patch.ID
		}
	}
	return applyMCPServerUpdate(base, patch)
}

func applyMCPConnectionPatch(target, previous *mcp.ConnectionConfig, patch *mcpConnectionPatch) {
	if patch.Command != nil {
		command := *patch.Command
		if previous != nil {
			safe, redacted := safeMCPCommand(previous.Command)
			if redacted && command == safe && !patch.ReplaceCommand {
				command = previous.Command
			}
		}
		target.Command = command
	}
	if patch.Args != nil {
		target.Args = append([]string(nil), (*patch.Args)...)
	}
	if patch.Env != nil {
		target.Env = cloneMCPStringMap(*patch.Env)
	}
	if patch.Cwd != nil {
		target.Cwd = *patch.Cwd
	}
	if patch.URL != nil {
		rawURL := *patch.URL
		if previous != nil {
			safe, redacted := safeMCPURL(previous.URL)
			if redacted && rawURL == safe && !patch.ReplaceURL {
				rawURL = previous.URL
			}
		}
		target.URL = rawURL
	}
	if patch.Headers != nil {
		target.Headers = cloneMCPStringMap(*patch.Headers)
	}
}

func applyMCPSecurityPatch(target *mcp.SecurityConfig, patch *mcpSecurityPatch) {
	if patch.AllowedOrigins != nil {
		target.AllowedOrigins = append([]string(nil), (*patch.AllowedOrigins)...)
	}
	if patch.LocalOnly != nil {
		target.LocalOnly = *patch.LocalOnly
	}
	if patch.OAuth != nil {
		if target.OAuth == nil {
			target.OAuth = &mcp.OAuthConfig{}
		}
		if patch.OAuth.ClientID != nil {
			target.OAuth.ClientID = *patch.OAuth.ClientID
		}
		if patch.OAuth.ClientSecret != nil {
			target.OAuth.ClientSecret = *patch.OAuth.ClientSecret
		}
		if patch.OAuth.AuthorizationURL != nil {
			target.OAuth.AuthorizationURL = *patch.OAuth.AuthorizationURL
		}
		if patch.OAuth.TokenURL != nil {
			target.OAuth.TokenURL = *patch.OAuth.TokenURL
		}
		if patch.OAuth.Scopes != nil {
			target.OAuth.Scopes = append([]string(nil), (*patch.OAuth.Scopes)...)
		}
		if patch.OAuth.UsePKCE != nil {
			target.OAuth.UsePKCE = *patch.OAuth.UsePKCE
		}
	}
}

func applyMCPOptionsPatch(target *mcp.ServerOptions, patch *mcpServerOptionsPatch) {
	if patch.AutoReconnect != nil {
		target.AutoReconnect = *patch.AutoReconnect
	}
	if patch.ReconnectInterval != nil {
		target.ReconnectInterval = *patch.ReconnectInterval
	}
	if patch.Timeout != nil {
		target.Timeout = *patch.Timeout
	}
	if patch.MaxRetries != nil {
		target.MaxRetries = *patch.MaxRetries
	}
}

func cloneMCPStringMap(source map[string]string) map[string]string {
	if source == nil {
		return nil
	}
	clone := make(map[string]string, len(source))
	for key, value := range source {
		clone[key] = value
	}
	return clone
}

func cloneMCPSecurity(security *mcp.SecurityConfig) *mcp.SecurityConfig {
	if security == nil {
		return nil
	}
	clone := *security
	clone.AllowedOrigins = append([]string(nil), security.AllowedOrigins...)
	if security.OAuth != nil {
		oauthClone := *security.OAuth
		oauthClone.Scopes = append([]string(nil), security.OAuth.Scopes...)
		clone.OAuth = &oauthClone
	}
	return &clone
}

func decodeMCPJSON(w http.ResponseWriter, r *http.Request, limit int64, dst any) (int, error) {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || !strings.EqualFold(mediaType, "application/json") {
		return http.StatusUnsupportedMediaType, errors.New("Content-Type must be application/json")
	}
	return decodeBoundedJSON(w, r, limit, dst)
}

// boundMCPProtocolJSON applies the same hard body and Unicode/single-value
// guarantees to the SDK-owned JSON-RPC endpoint before handing the request to
// mcp-go, whose Streamable HTTP handler otherwise performs an unbounded ReadAll.
func boundMCPProtocolJSON(w http.ResponseWriter, r *http.Request) (int, error) {
	mediaType, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if err != nil || !strings.EqualFold(mediaType, "application/json") {
		return http.StatusUnsupportedMediaType, errors.New("Content-Type must be application/json")
	}
	if r.Body == nil {
		return http.StatusBadRequest, errors.New("request body is required")
	}
	body := http.MaxBytesReader(w, r.Body, mcpToolBodyLimit)
	raw, err := io.ReadAll(body)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return http.StatusRequestEntityTooLarge, errors.New("request body is too large")
		}
		return http.StatusBadRequest, errors.New("request body could not be read")
	}
	var message json.RawMessage
	if status, err := decodeStrictJSONBytes(raw, &message); err != nil {
		return status, err
	}
	if err := body.Close(); err != nil {
		return http.StatusBadRequest, errors.New("request body could not be closed")
	}
	r.Body = io.NopCloser(bytes.NewReader(raw))
	return 0, nil
}

func authorizeMCPBuiltin(r *http.Request) (int, error) {
	_, err := auth.RevalidateAuthorization(r.Context(), auth.PermOpenClaw)
	if err == nil {
		return 0, nil
	}
	if errors.Is(err, auth.ErrLivePermissionDenied) {
		return http.StatusForbidden, errors.New("Forbidden")
	}
	return http.StatusUnauthorized, errors.New("Unauthorized")
}

func mayUseMCPBuiltin(r *http.Request) bool {
	_, err := auth.RevalidateAuthorization(r.Context(), auth.PermOpenClaw)
	return err == nil
}

func authorizeStdioMCP(r *http.Request, manager *mcp.Manager, config *mcp.ServerConfig) (int, error) {
	if config == nil || config.Transport != mcp.TransportStdio {
		return 0, nil
	}
	ac, err := auth.RevalidateAuthorization(r.Context(), auth.PermMcpManage)
	if err != nil {
		if errors.Is(err, auth.ErrLivePermissionDenied) {
			return http.StatusForbidden, errors.New("Forbidden")
		}
		return http.StatusUnauthorized, errors.New("Unauthorized")
	}
	if ac.Role != auth.RoleAdmin || manager == nil || !manager.AllowsStdio() {
		return http.StatusForbidden, errors.New("stdio transport is restricted to administrators in development")
	}
	return 0, nil
}

func validateMCPServerConfig(config *mcp.ServerConfig) error {
	if config == nil {
		return errors.New("server config is required")
	}
	config.Name = strings.TrimSpace(config.Name)
	if config.Name == "" {
		return errors.New("name is required")
	}
	switch config.Transport {
	case mcp.TransportStdio:
		if strings.TrimSpace(config.Connection.Command) == "" {
			return errors.New("command is required for stdio transport")
		}
	case mcp.TransportStreamableHTTP:
		parsed, err := url.Parse(strings.TrimSpace(config.Connection.URL))
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
			return errors.New("a valid HTTP(S) URL is required for streamable-http transport")
		}
		if parsed.User != nil {
			return errors.New("URL userinfo is not allowed; use a protected header instead")
		}
	default:
		return errors.New("transport must be stdio or streamable-http")
	}
	if err := mcp.ValidateConnectionSecurity(config); err != nil {
		return errors.New("local-only URL must use loopback or localhost")
	}
	return nil
}

func validateMCPServerID(id string) error {
	id = strings.TrimSpace(id)
	if id == "" || len(id) > 128 {
		return errors.New("server id must contain 1 to 128 characters")
	}
	for _, character := range id {
		if (character >= 'a' && character <= 'z') ||
			(character >= 'A' && character <= 'Z') ||
			(character >= '0' && character <= '9') ||
			character == '-' || character == '_' || character == '.' {
			continue
		}
		return errors.New("server id contains an unsupported character")
	}
	return nil
}

func managerErrorStatus(err error) (int, string) {
	switch {
	case errors.Is(err, mcp.ErrServerExists):
		return http.StatusConflict, "MCP server already exists"
	case errors.Is(err, mcp.ErrServerNotFound):
		return http.StatusNotFound, "MCP server not found"
	case errors.Is(err, mcp.ErrStdioTransportBlocked):
		return http.StatusForbidden, "stdio transport is disabled"
	case errors.Is(err, mcp.ErrManagerClosed):
		return http.StatusServiceUnavailable, "MCP manager is unavailable"
	case errors.Is(err, mcp.ErrLocalOnlyAddress):
		return http.StatusBadRequest, "local-only URL must use loopback or localhost"
	default:
		return http.StatusInternalServerError, "MCP operation failed"
	}
}
