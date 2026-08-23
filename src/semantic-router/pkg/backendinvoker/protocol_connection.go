package backendinvoker

import (
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"
)

const (
	maximumRuntimeBackendHeaders = 64
)

var runtimeHeaderNamePattern = regexp.MustCompile(`^[!#$%&'*+.^_` + "`" + `|~0-9A-Za-z-]+$`)

func validateRuntimeConnection(connection Connection) error {
	path, err := url.Parse(connection.Path)
	if err != nil || connection.Path == "" || !strings.HasPrefix(path.Path, "/") || path.Path != connection.Path ||
		path.IsAbs() || path.Host != "" || path.User != nil || path.RawQuery != "" || path.Fragment != "" {
		return fmt.Errorf("path must be an absolute-path reference without query or fragment")
	}
	if len(connection.Headers) > maximumRuntimeBackendHeaders {
		return fmt.Errorf("headers exceed %d entries", maximumRuntimeBackendHeaders)
	}
	for name, values := range connection.Headers {
		if !runtimeHeaderNamePattern.MatchString(name) || http.CanonicalHeaderKey(name) != name || len(values) != 1 {
			return fmt.Errorf("header %q must have exactly one value", name)
		}
		if _, forbidden := strippedHeaders[strings.ToLower(name)]; forbidden {
			return fmt.Errorf("header %q is runtime- or credential-owned", name)
		}
		if !boundedOptionalWireValue(values[0]) {
			return fmt.Errorf("header %q has an invalid value", name)
		}
	}
	return nil
}

func boundedOptionalWireValue(value string) bool {
	if len(value) > 4096 || strings.TrimSpace(value) != value {
		return false
	}
	for _, character := range value {
		if character < 0x20 || character == 0x7f {
			return false
		}
	}
	return true
}
