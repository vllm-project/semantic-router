package apicontract

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Route is one registered method on one ServeMux pattern, flattened from the
// auth route contract so this package does not depend on auth.
type Route struct {
	Pattern               string
	Method                string
	Permission            string
	AdditionalPermissions []string
	Sensitivity           string
	ResourceOwner         string
	AuditMode             string
	AuditAction           string
	Public                bool
	CSRF                  bool
	MaxBodyBytes          int64
	ProxyUpstream         bool
	Operation             *Operation
}

const (
	bearerScheme = "bearerAuth"
	cookieScheme = "sessionCookie"
	// subtreeParameter names the remainder matched by a ServeMux subtree
	// pattern such as /api/traces/.
	subtreeParameter = "subpath"
	auditNone        = "none"
)

var (
	operationIDPattern = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)
	pathWildcard       = regexp.MustCompile(`\{([^}]*)\}`)
	nonIdentifier      = regexp.MustCompile(`[^A-Za-z0-9]+`)
)

// Build renders routes as an OpenAPI document. sessionCookie names the cookie
// the authentication middleware reads. It fails on any ambiguity a generated
// client could not resolve: a duplicate operation id, two patterns that render
// to the same templated path, or a malformed explicit id.
func Build(info Info, sessionCookie string, routes []Route) (Spec, error) {
	spec := Spec{
		OpenAPI: "3.0.3",
		Info:    info,
		Paths:   map[string]PathItem{},
		Components: Components{SecuritySchemes: map[string]SecurityScheme{
			bearerScheme: {
				Type: "http", Scheme: "bearer", BearerFormat: "JWT",
				Description: "Dashboard access token. Bearer requests are exempt from CSRF checks.",
			},
			cookieScheme: {
				Type: "apiKey", In: "cookie", Name: sessionCookie,
				Description: "Browser session. Operations marked x-vllm-sr-csrf also require the X-CSRF-Token header.",
			},
		}},
	}
	operationIDs := map[string]string{}
	templates := map[string]string{}
	for _, route := range routes {
		path, err := openAPIPath(route.Pattern)
		if err != nil {
			return Spec{}, err
		}
		template := pathWildcard.ReplaceAllString(path, "{}")
		if previous, exists := templates[template]; exists && previous != route.Pattern {
			return Spec{}, fmt.Errorf("routes %q and %q render to the same OpenAPI path", previous, route.Pattern)
		}
		templates[template] = route.Pattern

		operation := renderOperation(route, path)
		if !operationIDPattern.MatchString(operation.OperationID) {
			return Spec{}, fmt.Errorf("%s %s has invalid operation id %q", route.Method, route.Pattern, operation.OperationID)
		}
		key := route.Method + " " + route.Pattern
		if previous, exists := operationIDs[operation.OperationID]; exists {
			return Spec{}, fmt.Errorf("operation id %q is used by %s and %s", operation.OperationID, previous, key)
		}
		operationIDs[operation.OperationID] = key

		item := spec.Paths[path]
		if item == nil {
			item = PathItem{}
			spec.Paths[path] = item
		}
		method := strings.ToLower(route.Method)
		if _, exists := item[method]; exists {
			return Spec{}, fmt.Errorf("%s %s is registered twice", route.Method, path)
		}
		item[method] = operation
	}
	return spec, nil
}

// openAPIPath converts a ServeMux pattern to an OpenAPI path. A trailing slash
// is a subtree match, so it gains a parameter for the remaining path.
func openAPIPath(pattern string) (string, error) {
	if pattern == "" || pattern[0] != '/' {
		return "", fmt.Errorf("route pattern %q is not an absolute path", pattern)
	}
	path := strings.ReplaceAll(pattern, "...}", "}")
	if path != "/" && strings.HasSuffix(path, "/") {
		if strings.Contains(path, "{"+subtreeParameter+"}") {
			return "", fmt.Errorf("route pattern %q already names the %s parameter", pattern, subtreeParameter)
		}
		return path + "{" + subtreeParameter + "}", nil
	}
	return path, nil
}

func renderOperation(route Route, path string) *OperationObject {
	operation := &OperationObject{
		OperationID:           defaultOperationID(route.Method, path),
		Summary:               route.Method + " " + route.Pattern,
		Tags:                  []string{tagFor(path)},
		Parameters:            pathParameters(path),
		Security:              security(route.Public),
		Permission:            route.Permission,
		AdditionalPermissions: route.AdditionalPermissions,
		Sensitivity:           route.Sensitivity,
		ResourceOwner:         route.ResourceOwner,
		AuditMode:             route.AuditMode,
		CSRF:                  route.CSRF,
		MaxBodyBytes:          route.MaxBodyBytes,
		ProxyUpstream:         route.ProxyUpstream,
		SchemaStatus:          SchemaUndocumented,
		Responses: map[string]Response{
			"default": {Description: "Response body is not yet described by the Dashboard contract."},
		},
	}
	// Read policies carry a default action even when nothing is audited;
	// publishing it would suggest the read is recorded.
	if route.AuditMode != auditNone {
		operation.AuditAction = route.AuditAction
	}
	// Record the registered pattern whenever the OpenAPI path differs, so the
	// operation maps back to exactly one route.
	if path != route.Pattern {
		operation.RoutePattern = route.Pattern
	}
	if route.Operation == nil {
		return operation
	}
	declared := route.Operation
	if declared.ID != "" {
		operation.OperationID = declared.ID
	}
	if declared.Summary != "" {
		operation.Summary = declared.Summary
	}
	operation.Description = declared.Description
	operation.RequestBody = declared.Request
	if len(declared.Responses) > 0 {
		operation.SchemaStatus = SchemaTyped
		operation.Responses = make(map[string]Response, len(declared.Responses))
		if !route.Public {
			// The policy middleware answers these before the handler runs.
			operation.Responses["401"] = TextResponse("Authentication is missing or invalid")
			operation.Responses["403"] = TextResponse("The caller lacks the permission, or the origin or CSRF check failed")
		}
		for status, response := range declared.Responses {
			operation.Responses[strconv.Itoa(status)] = response
		}
	}
	return operation
}

// defaultOperationID derives a camelCase id, matching declared ids such as
// getSettings: DELETE /api/mcp/servers/{id} becomes deleteApiMcpServersId.
func defaultOperationID(method, path string) string {
	var id strings.Builder
	id.WriteString(strings.ToLower(method))
	for _, word := range nonIdentifier.Split(path, -1) {
		if word == "" {
			continue
		}
		id.WriteString(strings.ToUpper(word[:1]))
		id.WriteString(word[1:])
	}
	return id.String()
}

// tagFor groups an operation by its first segment below /api/, or by its
// first path segment elsewhere: /api/mcp/tools is mcp, /healthz is healthz.
func tagFor(path string) string {
	trimmed := strings.TrimPrefix(strings.TrimPrefix(path, "/api"), "/")
	tag := strings.TrimSuffix(strings.Split(trimmed, "/")[0], ".json")
	if tag == "" {
		return "root"
	}
	return tag
}

func pathParameters(path string) []Parameter {
	matches := pathWildcard.FindAllStringSubmatch(path, -1)
	if len(matches) == 0 {
		return nil
	}
	parameters := make([]Parameter, 0, len(matches))
	for _, match := range matches {
		description := ""
		if match[1] == subtreeParameter {
			description = "Remaining path below the registered subtree pattern."
		}
		parameters = append(parameters, Parameter{
			Name: match[1], In: "path", Required: true, Description: description,
			Schema: Schema{Type: "string"},
		})
	}
	return parameters
}

func security(public bool) []map[string][]string {
	if public {
		return []map[string][]string{}
	}
	return []map[string][]string{{bearerScheme: {}}, {cookieScheme: {}}}
}
