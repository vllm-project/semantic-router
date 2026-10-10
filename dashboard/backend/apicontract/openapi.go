// Package apicontract renders the Dashboard route registration as an OpenAPI
// document. It does not own routes or authorization: the auth package records
// each route's policy, and this package only describes what was registered.
package apicontract

// Spec is the subset of OpenAPI 3.0 the Dashboard publishes.
type Spec struct {
	OpenAPI    string              `json:"openapi"`
	Info       Info                `json:"info"`
	Paths      map[string]PathItem `json:"paths"`
	Components Components          `json:"components"`
}

type Info struct {
	Title       string `json:"title"`
	Description string `json:"description"`
	Version     string `json:"version"`
}

// PathItem maps a lowercase HTTP method to its operation.
type PathItem map[string]*OperationObject

// OperationObject is one rendered operation. The x-vllm-sr-* extensions carry
// the route policy so clients and reviewers see authorization, audit, and CSRF
// requirements beside the schema.
type OperationObject struct {
	OperationID           string                `json:"operationId"`
	Summary               string                `json:"summary"`
	Description           string                `json:"description,omitempty"`
	Tags                  []string              `json:"tags,omitempty"`
	Parameters            []Parameter           `json:"parameters,omitempty"`
	Security              []map[string][]string `json:"security"`
	RequestBody           *RequestBody          `json:"requestBody,omitempty"`
	Responses             map[string]Response   `json:"responses"`
	Permission            string                `json:"x-vllm-sr-permission,omitempty"`
	AdditionalPermissions []string              `json:"x-vllm-sr-additional-permissions,omitempty"`
	Sensitivity           string                `json:"x-vllm-sr-sensitivity"`
	ResourceOwner         string                `json:"x-vllm-sr-resource-owner"`
	AuditMode             string                `json:"x-vllm-sr-audit-mode"`
	AuditAction           string                `json:"x-vllm-sr-audit-action,omitempty"`
	CSRF                  bool                  `json:"x-vllm-sr-csrf"`
	MaxBodyBytes          int64                 `json:"x-vllm-sr-max-body-bytes,omitempty"`
	ProxyUpstream         bool                  `json:"x-vllm-sr-proxy-upstream,omitempty"`
	RoutePattern          string                `json:"x-vllm-sr-route-pattern,omitempty"`
	SchemaStatus          SchemaStatus          `json:"x-vllm-sr-schema-status"`
}

// SchemaStatus says whether an operation's request and response bodies are
// described. Undocumented operations still carry their complete route policy.
type SchemaStatus string

const (
	SchemaTyped        SchemaStatus = "typed"
	SchemaUndocumented SchemaStatus = "undocumented"
)

type Parameter struct {
	Name        string `json:"name"`
	In          string `json:"in"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required"`
	Schema      Schema `json:"schema"`
}

type RequestBody struct {
	Description string           `json:"description,omitempty"`
	Required    bool             `json:"required"`
	Content     map[string]Media `json:"content"`
}

type Response struct {
	Description string           `json:"description"`
	Content     map[string]Media `json:"content,omitempty"`
}

type Media struct {
	Schema *Schema `json:"schema,omitempty"`
}

type Schema struct {
	Type                 string            `json:"type,omitempty"`
	Format               string            `json:"format,omitempty"`
	Description          string            `json:"description,omitempty"`
	Nullable             bool              `json:"nullable,omitempty"`
	Enum                 []string          `json:"enum,omitempty"`
	OneOf                []Schema          `json:"oneOf,omitempty"`
	Properties           map[string]Schema `json:"properties,omitempty"`
	Required             []string          `json:"required,omitempty"`
	Items                *Schema           `json:"items,omitempty"`
	AdditionalProperties any               `json:"additionalProperties,omitempty"`
}

// Accepts reports whether an object with exactly these property names would
// validate against the schema: every required property present, and, when
// the schema closes additional properties, no name outside those declared.
// It exists so a oneOf's branches can be tested for exclusivity against the
// real field sets a handler encodes.
func (s Schema) Accepts(keys []string) bool {
	present := make(map[string]bool, len(keys))
	for _, key := range keys {
		present[key] = true
	}
	for _, required := range s.Required {
		if !present[required] {
			return false
		}
	}
	if closed, ok := s.AdditionalProperties.(bool); ok && !closed {
		for _, key := range keys {
			if _, declared := s.Properties[key]; !declared {
				return false
			}
		}
	}
	return true
}

type Components struct {
	SecuritySchemes map[string]SecurityScheme `json:"securitySchemes"`
}

type SecurityScheme struct {
	Type         string `json:"type"`
	Scheme       string `json:"scheme,omitempty"`
	BearerFormat string `json:"bearerFormat,omitempty"`
	In           string `json:"in,omitempty"`
	Name         string `json:"name,omitempty"`
	Description  string `json:"description,omitempty"`
}
