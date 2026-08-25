package routingsnapshot

import (
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

const (
	maximumBackendHeaders = 64
)

var headerNamePattern = regexp.MustCompile(`^[!#$%&'*+.^_` + "`" + `|~0-9A-Za-z-]+$`)

var forbiddenBackendHeaders = map[string]struct{}{
	"api-key": {}, "authorization": {}, "connection": {}, "content-length": {},
	"cookie": {}, "host": {}, "keep-alive": {}, "proxy-authenticate": {},
	"proxy-authorization": {}, "set-cookie": {}, "te": {}, "trailer": {},
	"transfer-encoding": {}, "upgrade": {}, "x-api-key": {},
	"x-amz-security-token": {}, "x-auth-token": {}, "x-azure-api-key": {},
	"x-goog-api-key": {}, "x-openai-api-key": {}, "x-subscription-key": {},
	"x-user-anthropic-key": {}, "x-user-azure-openai-key": {},
	"x-user-bedrock-key": {}, "x-user-gemini-key": {},
	"x-user-minimax-key": {}, "x-user-openai-key": {},
	"x-user-vertex-ai-key": {},
}

func normalizeBackendConnection(connection *BackendConnection) error {
	if connection == nil {
		return fmt.Errorf("connection is required")
	}
	path, err := url.Parse(connection.Path)
	if err != nil || connection.Path == "" || !strings.HasPrefix(path.Path, "/") ||
		path.IsAbs() || path.Host != "" || path.User != nil || path.RawQuery != "" || path.Fragment != "" ||
		path.Path != connection.Path {
		return fmt.Errorf("connection.path must be an absolute-path reference without query or fragment")
	}
	canonicalHeaders, err := canonicalizeBackendHeaders(connection.Headers)
	if err != nil {
		return err
	}
	connection.Headers = canonicalHeaders
	return nil
}

// ValidateBackendHeaders applies the same non-secret, runtime-safe header
// contract used by immutable published snapshots. File-authored configurations
// compile connections through this boundary before headers enter dispatch state.
func ValidateBackendHeaders(headers map[string]string) error {
	_, err := canonicalizeBackendHeaders(headers)
	return err
}

func canonicalizeBackendHeaders(headers map[string]string) (map[string]string, error) {
	if len(headers) > maximumBackendHeaders {
		return nil, fmt.Errorf("connection.headers must contain at most %d entries", maximumBackendHeaders)
	}
	canonicalHeaders := make(map[string]string, len(headers))
	for name, value := range headers {
		canonicalName := http.CanonicalHeaderKey(name)
		if !headerNamePattern.MatchString(name) || canonicalName == "" {
			return nil, fmt.Errorf("connection header %q has an invalid name", name)
		}
		if _, forbidden := forbiddenBackendHeaders[strings.ToLower(canonicalName)]; forbidden {
			return nil, fmt.Errorf("connection header %q is runtime- or credential-owned", name)
		}
		if _, duplicate := canonicalHeaders[canonicalName]; duplicate {
			return nil, fmt.Errorf("connection header %q is duplicated after canonicalization", name)
		}
		if !canonicalOptionalWireText(value, 4096) {
			return nil, fmt.Errorf("connection header %q has an invalid value", name)
		}
		canonicalHeaders[canonicalName] = value
	}
	return canonicalHeaders, nil
}

// CanonicalizeBackendConnection validates and defensively copies a connection
// emitted by a control-plane Provider compiler before it can be persisted in a
// routing revision. Product-specific form values must already have been
// consumed; only the stable data-plane wire contract remains here.
func CanonicalizeBackendConnection(input BackendConnection) (BackendConnection, error) {
	connection := BackendConnection{Path: input.Path}
	if input.Headers != nil {
		connection.Headers = make(map[string]string, len(input.Headers))
		for name, value := range input.Headers {
			connection.Headers[name] = value
		}
	}
	if err := normalizeBackendConnection(&connection); err != nil {
		return BackendConnection{}, err
	}
	return connection, nil
}

func canonicalOptionalWireText(value string, maximum int) bool {
	if len(value) > maximum || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}
