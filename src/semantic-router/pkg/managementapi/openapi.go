package managementapi

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
)

const openAPIJSONSchemaDialect = "https://json-schema.org/draft/2020-12/schema"

type OpenAPIDocument struct {
	OpenAPI           string                 `json:"openapi"`
	JSONSchemaDialect string                 `json:"jsonSchemaDialect"`
	Info              OpenAPIInfo            `json:"info"`
	Servers           []OpenAPIServer        `json:"servers"`
	Paths             map[string]OpenAPIPath `json:"paths"`
	Components        OpenAPIComponents      `json:"components"`
}

type OpenAPIInfo struct {
	Title   string `json:"title"`
	Version string `json:"version"`
}

type OpenAPIServer struct {
	URL         string                           `json:"url"`
	Description string                           `json:"description,omitempty"`
	Variables   map[string]OpenAPIServerVariable `json:"variables,omitempty"`
}

type OpenAPIServerVariable struct {
	Default     string `json:"default"`
	Description string `json:"description,omitempty"`
}

type OpenAPIPath map[string]OpenAPIOperation

type OpenAPIOperation struct {
	OperationID                string                     `json:"operationId"`
	Tags                       []string                   `json:"tags"`
	Parameters                 []OpenAPIParameter         `json:"parameters,omitempty"`
	RequestBody                *OpenAPIRequestBody        `json:"requestBody,omitempty"`
	Responses                  map[string]OpenAPIResponse `json:"responses"`
	Security                   []map[string][]string      `json:"security,omitempty"`
	RouterPermissionExpression PermissionExpression       `json:"x-router-permission-expression"`
	RouterPermissionCanonical  string                     `json:"x-router-permission-canonical"`
	RouterScope                OperationScope             `json:"x-router-scope"`
	RouterAsync                AsyncMode                  `json:"x-router-async"`
	RouterSecret               SecretMetadata             `json:"x-router-secret"`
	RouterPagination           PaginationMode             `json:"x-router-pagination"`
	RouterIdempotency          IdempotencyMode            `json:"x-router-idempotency"`
	RouterRevision             RevisionMode               `json:"x-router-revision"`
}

type OpenAPIRequestBody struct {
	Required bool                    `json:"required"`
	Content  map[string]OpenAPIMedia `json:"content"`
}

type OpenAPIParameter struct {
	Name        string     `json:"name"`
	In          string     `json:"in"`
	Required    bool       `json:"required"`
	Description string     `json:"description,omitempty"`
	Schema      JSONSchema `json:"schema"`
}

type OpenAPIResponse struct {
	Description string                   `json:"description"`
	Headers     map[string]OpenAPIHeader `json:"headers,omitempty"`
	Content     map[string]OpenAPIMedia  `json:"content,omitempty"`
}

type OpenAPIHeader struct {
	Description string     `json:"description,omitempty"`
	Schema      JSONSchema `json:"schema"`
}

type OpenAPIMedia struct {
	Schema JSONSchema `json:"schema"`
}

type OpenAPIComponents struct {
	Schemas         map[string]JSONSchema            `json:"schemas"`
	SecuritySchemes map[string]OpenAPISecurityScheme `json:"securitySchemes"`
}

type OpenAPISecurityScheme struct {
	Type         string `json:"type"`
	Scheme       string `json:"scheme,omitempty"`
	BearerFormat string `json:"bearerFormat,omitempty"`
	Name         string `json:"name,omitempty"`
	In           string `json:"in,omitempty"`
	Description  string `json:"description,omitempty"`
}

type JSONSchema struct {
	Ref                  string                `json:"$ref,omitempty"`
	Type                 string                `json:"type,omitempty"`
	Format               string                `json:"format,omitempty"`
	Description          string                `json:"description,omitempty"`
	Pattern              string                `json:"pattern,omitempty"`
	Minimum              *int64                `json:"minimum,omitempty"`
	Maximum              *int64                `json:"maximum,omitempty"`
	MinLength            *int64                `json:"minLength,omitempty"`
	MaxLength            *int64                `json:"maxLength,omitempty"`
	Enum                 []string              `json:"enum,omitempty"`
	Required             []string              `json:"required,omitempty"`
	Properties           map[string]JSONSchema `json:"properties,omitempty"`
	PatternProperties    map[string]JSONSchema `json:"patternProperties,omitempty"`
	Items                *JSONSchema           `json:"items,omitempty"`
	MinItems             *int64                `json:"minItems,omitempty"`
	MaxItems             *int64                `json:"maxItems,omitempty"`
	UniqueItems          bool                  `json:"uniqueItems,omitempty"`
	OneOf                []JSONSchema          `json:"oneOf,omitempty"`
	AdditionalProperties *bool                 `json:"additionalProperties,omitempty"`
}

func GenerateOpenAPI() OpenAPIDocument {
	document := OpenAPIDocument{
		OpenAPI:           "3.1.0",
		JSONSchemaDialect: openAPIJSONSchemaDialect,
		Info: OpenAPIInfo{
			Title:   "vLLM Semantic Router Management API",
			Version: ContractVersion,
		},
		Servers: []OpenAPIServer{{
			URL:         "https://{managementAddress}",
			Description: "Router-terminated TLS Management listener.",
			Variables: map[string]OpenAPIServerVariable{
				"managementAddress": {
					Default:     "localhost:8080",
					Description: "Configured Management listener host and port.",
				},
			},
		}},
		Paths: make(map[string]OpenAPIPath),
		Components: OpenAPIComponents{
			Schemas:         canonicalSchemas(),
			SecuritySchemes: securitySchemes(),
		},
	}
	for _, contract := range Operations() {
		path := document.Paths[contract.Path]
		if path == nil {
			path = make(OpenAPIPath)
			document.Paths[contract.Path] = path
		}
		path[strings.ToLower(string(contract.Method))] = openAPIOperation(contract)
	}
	return document
}

func GenerateOpenAPIJSON() ([]byte, error) {
	document := GenerateOpenAPI()
	encoded, err := json.MarshalIndent(document, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal Management OpenAPI: %w", err)
	}
	return append(encoded, '\n'), nil
}

func openAPIOperation(contract OperationContract) OpenAPIOperation {
	return OpenAPIOperation{
		OperationID:                contract.OperationID,
		Tags:                       []string{contract.Tag},
		Parameters:                 operationParameters(contract),
		RequestBody:                operationRequestBody(contract),
		Responses:                  operationResponses(contract),
		Security:                   operationSecurity(contract),
		RouterPermissionExpression: contract.Permission,
		RouterPermissionCanonical:  contract.Permission.Canonical(),
		RouterScope:                contract.Scope,
		RouterAsync:                contract.Async,
		RouterSecret:               contract.Secret,
		RouterPagination:           contract.Pagination,
		RouterIdempotency:          contract.Idempotency,
		RouterRevision:             contract.Revision,
	}
}

var openAPIPathParameterPattern = regexp.MustCompile(`\{([A-Za-z][A-Za-z0-9]*)\}`)

func operationParameters(contract OperationContract) []OpenAPIParameter {
	parameters := []OpenAPIParameter{managementAcceptParameter(contract)}
	for _, match := range openAPIPathParameterPattern.FindAllStringSubmatch(contract.Path, -1) {
		parameters = append(parameters, OpenAPIParameter{
			Name:     match[1],
			In:       "path",
			Required: true,
			Schema:   JSONSchema{Type: "string", Pattern: `^[^/?#]+$`},
		})
	}
	if contract.Pagination == PaginationKeyset {
		parameters = append(parameters,
			OpenAPIParameter{Name: "cursor", In: "query", Required: false, Schema: JSONSchema{Type: "string"}},
			OpenAPIParameter{Name: "pageSize", In: "query", Required: false, Schema: boundedIntegerSchema(1, 200)},
		)
	}
	if requiresExplicitNamespace(contract.Path) {
		parameters = append(parameters, OpenAPIParameter{
			Name:        HeaderNamespaceID,
			In:          "header",
			Required:    true,
			Description: "Canonical namespace UUID for namespace-owned Management resources.",
			Schema: JSONSchema{
				Type:    "string",
				Pattern: `^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`,
			},
		})
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/providers" {
		parameters = append(parameters,
			OpenAPIParameter{Name: "search", In: "query", Schema: JSONSchema{Type: "string"}},
			OpenAPIParameter{Name: "category", In: "query", Schema: JSONSchema{Type: "string"}},
			OpenAPIParameter{Name: "capability", In: "query", Schema: JSONSchema{Type: "string"}},
		)
	}
	parameters = append(parameters, collectionSearchParameters(contract)...)
	parameters = append(parameters, extensionParameters(contract)...)
	if contract.Idempotency == IdempotencyRequired {
		parameters = append(parameters, OpenAPIParameter{
			Name: HeaderIdempotencyKey, In: "header", Required: true,
			Description: "Opaque key scoped to actor and normalized request digest.",
			Schema:      JSONSchema{Type: "string", Pattern: `^[!-~]{16,200}$`},
		})
	}
	if contract.Revision == RevisionCAS {
		parameters = append(parameters, OpenAPIParameter{
			Name: HeaderIfMatch, In: "header", Required: true,
			Schema: JSONSchema{Type: "string", Pattern: `^\"[^\"]+\"$`},
		})
	}
	sort.SliceStable(parameters, func(i, j int) bool {
		if parameters[i].In == parameters[j].In {
			return parameters[i].Name < parameters[j].Name
		}
		return parameters[i].In < parameters[j].In
	})
	return parameters
}

func managementAcceptParameter(contract OperationContract) OpenAPIParameter {
	mediaTypes := []string{JSONMediaType}
	description := "Required response media type. Generic JSON and wildcard negotiation are not supported."
	if contract.Method == MethodGET && contract.Path == BasePath+"/agent-sessions/{session}/events" {
		mediaTypes = append(mediaTypes, EventStreamMediaType)
		description = "Required response media type. Use text/event-stream for live delivery or the Management API media type for durable history."
	}
	return OpenAPIParameter{
		Name:        "Accept",
		In:          "header",
		Required:    true,
		Description: description,
		Schema:      JSONSchema{Type: "string", Enum: mediaTypes},
	}
}

var explicitlyNamespacedPathPrefixes = []string{
	BasePath + "/users",
	BasePath + "/teams",
	BasePath + "/api-keys",
	BasePath + "/access-policies",
	BasePath + "/access-policy-bindings",
	BasePath + "/rate-limit-policies",
	BasePath + "/rate-limit-bindings",
	BasePath + "/access:check",
	BasePath + "/unknown-usage-fences",
	BasePath + "/routing",
	BasePath + "/providers",
	BasePath + "/provider-credentials",
	BasePath + "/invitations",
	BasePath + "/onboarding",
	BasePath + "/operations",
	BasePath + "/usage",
	BasePath + "/request-logs",
	BasePath + "/audit-events",
	BasePath + "/agent-profiles",
	BasePath + "/agent-skills",
	BasePath + "/agent-tools",
	BasePath + "/agent-tool-credentials",
	BasePath + "/agent-tool-sources",
	BasePath + "/agent-sessions",
	BasePath + "/agent-artifacts",
	BasePath + "/publication-plans",
}

func requiresExplicitNamespace(path string) bool {
	if strings.Contains(path, "{namespaceId}") {
		return false
	}
	if path == BasePath+"/self/inference-keys" ||
		path == BasePath+"/self/inference-sessions" ||
		strings.HasPrefix(path, BasePath+"/self/inference-sessions/") {
		return true
	}
	for _, prefix := range explicitlyNamespacedPathPrefixes {
		if path == prefix || strings.HasPrefix(path, prefix+"/") || strings.HasPrefix(path, prefix+":") {
			return true
		}
	}
	return false
}

func operationRequestBody(contract OperationContract) *OpenAPIRequestBody {
	var schema string
	if extensionSchema, found := extensionRequestSchema(contract); found {
		schema = extensionSchema
	} else {
		switch {
		case contract.Method == MethodPOST && contract.Path == BasePath+"/users":
			schema = "UserCreateRequest"
		case contract.Method == MethodPATCH && contract.Path == BasePath+"/users/{userId}":
			schema = "UserPatchRequest"
		case contract.Method == MethodPOST && contract.Path == BasePath+"/teams":
			schema = "TeamCreateRequest"
		case contract.Method == MethodPATCH && contract.Path == BasePath+"/teams/{teamId}":
			schema = "TeamPatchRequest"
		case contract.Method == MethodPUT && contract.Path == BasePath+"/teams/{teamId}/members/{userId}":
			schema = "MembershipPutRequest"
		case contract.Method == MethodPATCH && contract.Path == BasePath+"/teams/{teamId}/members/{userId}":
			schema = "MembershipPatchRequest"
		case contract.Method == MethodPOST && contract.Path == BasePath+"/providers/{providerId}:discover-models":
			schema = "DiscoverModelsRequest"
		case contract.Method == MethodPOST && contract.Path == BasePath+"/provider-credentials":
			schema = "ProviderCredentialCreateRequest"
		case contract.Method == MethodPATCH && contract.Path == BasePath+"/provider-credentials/{credentialId}":
			schema = "ProviderCredentialPatchRequest"
		case contract.Method == MethodPOST && contract.Path == BasePath+"/provider-credentials/{credentialId}:rotate":
			schema = "ProviderCredentialRotateRequest"
		default:
			return nil
		}
	}
	return &OpenAPIRequestBody{
		Required: true,
		Content: map[string]OpenAPIMedia{
			JSONMediaType: {Schema: refSchema(schema)},
		},
	}
}

func operationResponses(contract OperationContract) map[string]OpenAPIResponse {
	status := "200"
	if contract.Async == AsyncOperation {
		status = "202"
	} else if contract.Method == MethodPOST && !strings.Contains(contract.Path, ":") &&
		contract.Path != BasePath+"/auth/token-exchange" &&
		contract.Path != BasePath+"/auth/backchannel-logout" {
		status = "201"
	} else if contract.Method == MethodDELETE {
		status = "204"
	}

	successSchema := responseSchema(contract)
	response := OpenAPIResponse{Description: "Success"}
	if status != "204" {
		response.Content = map[string]OpenAPIMedia{JSONMediaType: {Schema: successSchema}}
	}
	if contract.Revision != RevisionNone {
		response.Headers = map[string]OpenAPIHeader{
			HeaderETag: {Description: "Strong resource revision validator.", Schema: JSONSchema{Type: "string"}},
		}
	}
	if contract.Idempotency == IdempotencyRequired {
		if response.Headers == nil {
			response.Headers = make(map[string]OpenAPIHeader)
		}
		response.Headers[HeaderIdempotencyReplayed] = OpenAPIHeader{
			Description: "Present with value true only when a stored result was replayed.",
			Schema:      JSONSchema{Type: "string", Enum: []string{"true"}},
		}
	}
	if contract.Secret.NoStore {
		if response.Headers == nil {
			response.Headers = make(map[string]OpenAPIHeader)
		}
		response.Headers["Cache-Control"] = OpenAPIHeader{Schema: JSONSchema{Type: "string", Enum: []string{"no-store"}}}
		if contract.Secret.Authenticated {
			response.Headers["Vary"] = OpenAPIHeader{Schema: JSONSchema{Type: "string", Enum: []string{"Authorization"}}}
		}
	}

	responses := map[string]OpenAPIResponse{status: response}
	for _, errorStatus := range []string{"400", "401", "403", "404", "409", "410", "412", "413", "415", "428", "429", "500", "502", "503"} {
		headers := map[string]OpenAPIHeader(nil)
		if contract.Secret.NoStore {
			headers = map[string]OpenAPIHeader{
				"Cache-Control": {Schema: JSONSchema{Type: "string", Enum: []string{"no-store"}}},
			}
			if contract.Secret.Authenticated {
				headers["Vary"] = OpenAPIHeader{Schema: JSONSchema{Type: "string", Enum: []string{"Authorization"}}}
			}
		}
		responses[errorStatus] = OpenAPIResponse{
			Description: "Canonical Management API error.",
			Headers:     headers,
			Content: map[string]OpenAPIMedia{
				JSONMediaType: {Schema: refSchema("ErrorResponse")},
			},
		}
	}
	extensionAmendResponses(contract, responses)
	return responses
}

func responseSchema(contract OperationContract) JSONSchema {
	if schema, found := extensionResponseSchema(contract); found {
		return schema
	}
	switch contract.Secret.Output {
	case SecretOutputOneTime:
		return refSchema("SecretEnvelope")
	case SecretOutputAccessToken:
		return refSchema("ManagementTokenEnvelope")
	}
	if contract.Async == AsyncOperation {
		return refSchema("Operation")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/users" {
		return refSchema("UserPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/users/{userId}" {
		return refSchema("UserDetail")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/users/{userId}/memberships" {
		return refSchema("UserMembershipPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/teams" {
		return refSchema("TeamPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/teams/{teamId}" {
		return refSchema("TeamDetail")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/teams/{teamId}/members" {
		return refSchema("TeamMemberPage")
	}
	if (strings.HasPrefix(contract.Path, BasePath+"/users") || strings.HasPrefix(contract.Path, BasePath+"/teams")) &&
		(contract.Method == MethodPOST || contract.Method == MethodPUT || contract.Method == MethodPATCH) {
		return refSchema("MutationReceipt")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/providers" {
		return refSchema("ProviderCatalogPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/providers/{providerId}" {
		return refSchema("ProviderCatalogDetail")
	}
	if contract.Method == MethodPOST && contract.Path == BasePath+"/providers/{providerId}:discover-models" {
		return refSchema("DiscoverModelsPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/provider-credentials" {
		return refSchema("ProviderCredentialPage")
	}
	if contract.Method == MethodGET && contract.Path == BasePath+"/provider-credentials/{credentialId}" {
		return refSchema("ProviderCredentialDetail")
	}
	if strings.HasPrefix(contract.Path, BasePath+"/provider-credentials") &&
		(contract.Method == MethodPOST || contract.Method == MethodPATCH) {
		return refSchema("MutationReceipt")
	}
	if strings.HasSuffix(contract.Path, "/effective-policy") {
		return refSchema("EffectivePolicy")
	}
	if strings.HasSuffix(contract.Path, "/quota") {
		return refSchema("EffectiveQuota")
	}
	if strings.Contains(contract.Path, "/operations") && contract.Method == MethodGET {
		return refSchema("Operation")
	}
	if contract.Pagination == PaginationKeyset {
		return refSchema("Page")
	}
	return JSONSchema{Type: "object", AdditionalProperties: boolPointer(true)}
}

func operationSecurity(contract OperationContract) []map[string][]string {
	if contract.Permission.Operator != PermissionSpecial {
		return []map[string][]string{{"managementBearer": {}}}
	}
	if contract.Permission.Mechanism == "service_credential_or_mtls" {
		return []map[string][]string{{"serviceCredential": {}}, {"mutualTLS": {}}}
	}
	scheme := map[string]string{
		"bootstrap_credential":               "bootstrapCredential",
		"recovery_credential":                "recoveryCredential",
		"trusted_issuer_logout_token":        "issuerLogoutToken",
		"exchange_challenge":                 "publicExchange",
		"subject_token_exchange":             "publicExchange",
		"onboarding_secret_claim_capability": "managementBearer",
	}[contract.Permission.Mechanism]
	if scheme == "publicExchange" {
		return nil
	}
	return []map[string][]string{{scheme: {}}}
}
