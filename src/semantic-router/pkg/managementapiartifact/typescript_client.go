package managementapiartifact

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func renderManagementClient(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	clientOperations := sortedManagementClientOperations(operations)
	renderManagementOperationTypes(output, document, clientOperations)
	renderManagementTypeSupport(output)
	renderManagementResponseValidators(output, document, clientOperations)
	renderManagementClientInterface(output, document, clientOperations)
	renderManagementClientFactory(output, document, clientOperations)
}

func sortedManagementClientOperations(operations []managementapi.OperationContract) []managementapi.OperationContract {
	clientOperations := append([]managementapi.OperationContract(nil), operations...)
	sort.Slice(clientOperations, func(i, j int) bool {
		return clientOperations[i].OperationID < clientOperations[j].OperationID
	})
	return clientOperations
}

func renderManagementOperationTypes(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("\nexport interface ManagementApiOperationTypes {\n")
	for _, operation := range operations {
		openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
		fmt.Fprintf(output, "  %s: {\n", operation.OperationID)
		fmt.Fprintf(output, "    path: %s\n", operationPathType(openAPIOperation))
		fmt.Fprintf(output, "    body: %s\n", operationRequestType(openAPIOperation))
		fmt.Fprintf(output, "    query: %s\n", operationQueryType(openAPIOperation))
		fmt.Fprintf(output, "    headers: %s\n", operationHeaderType(openAPIOperation))
		fmt.Fprintf(output, "    response: %s\n", operationResponseType(openAPIOperation))
		output.WriteString("  }\n")
	}
	output.WriteString("}\n\n")
}

func renderManagementTypeSupport(output *strings.Builder) {
	output.WriteString("export type ManagementApiClientOperationId = keyof ManagementApiOperationTypes\n")
	output.WriteString("export type ManagementApiRequestBody<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  ManagementApiOperationTypes[OperationId]['body']\n")
	output.WriteString("export type ManagementApiQuery<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  ManagementApiOperationTypes[OperationId]['query']\n")
	output.WriteString("export type ManagementApiRequestHeaders<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  ManagementApiOperationTypes[OperationId]['headers']\n")
	output.WriteString("export type ManagementApiResponse<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  ManagementApiOperationTypes[OperationId]['response']\n\n")
	output.WriteString("export type ManagementApiSuccessStatus<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  (typeof MANAGEMENT_API_OPERATIONS)[OperationId]['successStatuses'][number]\n")
	output.WriteString("export type ManagementApiSuccessMediaType<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  (typeof MANAGEMENT_API_OPERATIONS)[OperationId]['successMediaTypes'][number]\n\n")

	output.WriteString("type ManagementApiPathOption<OperationId extends ManagementApiClientOperationId> = [\n")
	output.WriteString("  ManagementApiPathParameters<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { pathParameters?: never }\n")
	output.WriteString("  : { pathParameters: ManagementApiPathParameters<OperationId> }\n\n")
	output.WriteString("type ManagementApiBodyOption<OperationId extends ManagementApiClientOperationId> = [\n")
	output.WriteString("  ManagementApiRequestBody<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { body?: never }\n")
	output.WriteString("  : (typeof MANAGEMENT_API_OPERATIONS)[OperationId]['requestBodyRequired'] extends true\n")
	output.WriteString("    ? { body: ManagementApiRequestBody<OperationId> }\n")
	output.WriteString("    : { body?: ManagementApiRequestBody<OperationId> }\n\n")
	output.WriteString("type ManagementApiQueryOption<OperationId extends ManagementApiClientOperationId> = [\n")
	output.WriteString("  ManagementApiQuery<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { query?: never }\n")
	output.WriteString("  : [ManagementApiRequiredKeys<ManagementApiQuery<OperationId>>] extends [never]\n")
	output.WriteString("    ? { query?: ManagementApiQuery<OperationId> }\n")
	output.WriteString("    : { query: ManagementApiQuery<OperationId> }\n\n")
	output.WriteString("type ManagementApiRequiredKeys<Value> = keyof {\n")
	output.WriteString("  [Key in keyof Value as Value extends Required<Pick<Value, Key>> ? Key : never]: true\n")
	output.WriteString("}\n\n")
	output.WriteString("type ManagementApiHeaderOption<OperationId extends ManagementApiClientOperationId> = [\n")
	output.WriteString("  ManagementApiRequestHeaders<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { headers?: never }\n")
	output.WriteString("  : [ManagementApiRequiredKeys<ManagementApiRequestHeaders<OperationId>>] extends [never]\n")
	output.WriteString("    ? { headers?: ManagementApiRequestHeaders<OperationId> }\n")
	output.WriteString("    : { headers: ManagementApiRequestHeaders<OperationId> }\n\n")

	output.WriteString("export type ManagementApiRequestOptions<OperationId extends ManagementApiClientOperationId> = {\n")
	output.WriteString("  signal?: AbortSignal\n")
	output.WriteString("  namespace?: string | null\n")
	output.WriteString("} & ManagementApiPathOption<OperationId> &\n")
	output.WriteString("  ManagementApiBodyOption<OperationId> &\n")
	output.WriteString("  ManagementApiQueryOption<OperationId> &\n")
	output.WriteString("  ManagementApiHeaderOption<OperationId>\n\n")

	output.WriteString("export interface ManagementApiClientResponse<ResponseBody> {\n")
	output.WriteString("  data: ResponseBody\n")
	output.WriteString("  status: number\n")
	output.WriteString("  mediaType?: string\n")
	output.WriteString("  etag?: string\n")
	output.WriteString("  requestId?: string\n")
	output.WriteString("  idempotencyReplayed?: boolean\n")
	output.WriteString("  secretResultClaim?: string\n")
	output.WriteString("}\n\n")
	output.WriteString("export type ManagementApiOperationClientResponse<OperationId extends ManagementApiClientOperationId> =\n")
	output.WriteString("  Omit<ManagementApiClientResponse<ManagementApiResponse<OperationId>>, 'status' | 'mediaType'> & {\n")
	output.WriteString("    status: ManagementApiSuccessStatus<OperationId>\n")
	output.WriteString("    mediaType?: ManagementApiSuccessMediaType<OperationId>\n")
	output.WriteString("  }\n\n")
	output.WriteString("export interface ManagementApiTransport {\n")
	output.WriteString("  request<OperationId extends ManagementApiClientOperationId>(\n")
	output.WriteString("    operationId: OperationId,\n")
	output.WriteString("    options: ManagementApiRequestOptions<OperationId>,\n")
	output.WriteString("  ): Promise<ManagementApiClientResponse<unknown>>\n")
	output.WriteString("}\n\n")
}

func renderManagementResponseValidators(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	responseSchemas := make(map[string]map[string]map[string]managementapi.JSONSchema)
	for _, operation := range operations {
		openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
		byStatus := make(map[string]map[string]managementapi.JSONSchema)
		for _, status := range successfulStatuses(openAPIOperation) {
			byMediaType := make(map[string]managementapi.JSONSchema)
			for mediaType, media := range openAPIOperation.Responses[status].Content {
				if mediaType == managementapi.EventStreamMediaType {
					continue
				}
				byMediaType[mediaType] = media.Schema
			}
			byStatus[status] = byMediaType
		}
		responseSchemas[operation.OperationID] = byStatus
	}
	encoded, err := json.Marshal(responseSchemas)
	if err != nil {
		panic(fmt.Sprintf("marshal generated Management response schemas: %v", err))
	}
	fmt.Fprintf(
		output,
		"const MANAGEMENT_API_RESPONSE_SCHEMAS = JSON.parse(\n  %s,\n) as Record<ManagementApiClientOperationId, Record<string, Record<string, ManagementApiRuntimeSchema>>>\n\n",
		tsString(string(encoded)),
	)
	output.WriteString(`export function assertManagementApiOperationResponse<OperationId extends ManagementApiClientOperationId>(
  operationId: OperationId,
  value: unknown,
  status: number,
  mediaType?: string,
): ManagementApiResponse<OperationId> {
  const operation = MANAGEMENT_API_OPERATIONS[operationId]
  const successStatuses = operation.successStatuses as readonly number[]
  if (!successStatuses.includes(status)) {
    throw new Error(
      'Router returned HTTP ' + status + ' for Management operation ' + operationId +
        '; expected ' + successStatuses.join(' or ') + '.',
    )
  }
  const responseSchemas = MANAGEMENT_API_RESPONSE_SCHEMAS[operationId][String(status)]
  const successMediaTypes = Object.keys(responseSchemas)
  if (successMediaTypes.length === 0) {
    if (mediaType !== undefined) {
      throw new Error(
        'Router returned media type ' + mediaType + ' for empty Management operation ' + operationId + '.',
      )
    }
    if (value !== undefined) {
      throw new Error('Router returned a body for empty Management operation ' + operationId + '.')
    }
    return value as ManagementApiResponse<OperationId>
  }
  if (mediaType === undefined) {
    throw new Error('Router omitted the response media type for Management operation ' + operationId + '.')
  }
  const schema = responseSchemas[mediaType]
  if (!schema) {
	throw new Error(
	  'Router returned media type ' + mediaType + ' for Management operation ' + operationId +
		'; expected ' + successMediaTypes.join(' or ') + '.',
	)
  }
  if (!managementApiSchemaMatches(schema, value)) {
    const schemaName = schema.$ref?.replace('#/components/schemas/', '') ?? operationId
    throw new Error('Router returned a response that does not match ' + schemaName + '.')
  }
  return value as ManagementApiResponse<OperationId>
}

`)
}

func renderManagementClientInterface(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("export interface ManagementApiClient {\n")
	for _, operation := range operations {
		fmt.Fprintf(output, "  %s(\n", operation.OperationID)
		if operationRequiresOptions(operation, document) {
			fmt.Fprintf(output, "    options: ManagementApiRequestOptions<%s>,\n", tsString(operation.OperationID))
		} else {
			fmt.Fprintf(output, "    options?: ManagementApiRequestOptions<%s>,\n", tsString(operation.OperationID))
		}
		fmt.Fprintf(
			output,
			"  ): Promise<ManagementApiOperationClientResponse<%s>>\n",
			tsString(operation.OperationID),
		)
	}
	output.WriteString("}\n\n")
}

func renderManagementClientFactory(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("export function createManagementApiClient(\n")
	output.WriteString("  transport: ManagementApiTransport,\n")
	output.WriteString("): ManagementApiClient {\n")
	output.WriteString("  return {\n")
	for _, operation := range operations {
		fmt.Fprintf(output, "    %s: async (options%s) => {\n", operation.OperationID, optionalDefault(operation, document))
		fmt.Fprintf(output, "      const response = await transport.request(%s, options)\n", tsString(operation.OperationID))
		fmt.Fprintf(output, "      const data = assertManagementApiOperationResponse(%s, response.data, response.status, response.mediaType)\n", tsString(operation.OperationID))
		fmt.Fprintf(output, "      return { ...response, data } as ManagementApiOperationClientResponse<%s>\n", tsString(operation.OperationID))
		output.WriteString("    },\n")
	}
	output.WriteString("  }\n")
	output.WriteString("}\n")
}

func successfulStatuses(operation managementapi.OpenAPIOperation) []string {
	statuses := make([]string, 0, len(operation.Responses))
	for status := range operation.Responses {
		code, err := strconv.Atoi(status)
		if err == nil && code >= 200 && code < 300 {
			statuses = append(statuses, status)
		}
	}
	sort.Slice(statuses, func(i, j int) bool {
		left, _ := strconv.Atoi(statuses[i])
		right, _ := strconv.Atoi(statuses[j])
		return left < right
	})
	return statuses
}

func renderSuccessStatuses(operation managementapi.OpenAPIOperation) string {
	return "[" + strings.Join(successfulStatuses(operation), ", ") + "] as const"
}

func successfulMediaTypes(operation managementapi.OpenAPIOperation) []string {
	mediaTypes := make(map[string]struct{})
	for _, status := range successfulStatuses(operation) {
		for mediaType := range operation.Responses[status].Content {
			mediaTypes[mediaType] = struct{}{}
		}
	}
	result := make([]string, 0, len(mediaTypes))
	for mediaType := range mediaTypes {
		result = append(result, mediaType)
	}
	sort.Strings(result)
	return result
}

func renderSuccessMediaTypes(operation managementapi.OpenAPIOperation) string {
	mediaTypes := successfulMediaTypes(operation)
	rendered := make([]string, 0, len(mediaTypes))
	for _, mediaType := range mediaTypes {
		rendered = append(rendered, tsString(mediaType))
	}
	return "[" + strings.Join(rendered, ", ") + "] as const"
}

func renderSuccessResponses(operation managementapi.OpenAPIOperation) string {
	responses := make([]string, 0)
	for _, status := range successfulStatuses(operation) {
		mediaTypes := make([]string, 0, len(operation.Responses[status].Content))
		for mediaType := range operation.Responses[status].Content {
			mediaTypes = append(mediaTypes, mediaType)
		}
		sort.Strings(mediaTypes)
		rendered := make([]string, 0, len(mediaTypes))
		for _, mediaType := range mediaTypes {
			rendered = append(rendered, tsString(mediaType))
		}
		responses = append(responses, status+": ["+strings.Join(rendered, ", ")+"]")
	}
	return "{ " + strings.Join(responses, ", ") + " } as const"
}

func operationResponseMode(operation managementapi.OpenAPIOperation) string {
	hasJSON := false
	hasEventStream := false
	hasYAML := false
	hasEmpty := false
	for _, status := range successfulStatuses(operation) {
		response := operation.Responses[status]
		if len(response.Content) == 0 {
			hasEmpty = true
		}
		if _, found := response.Content[managementapi.JSONMediaType]; found {
			hasJSON = true
		}
		if _, found := response.Content[managementapi.EventStreamMediaType]; found {
			hasEventStream = true
		}
		if _, found := response.Content[managementapi.YAMLMediaType]; found {
			hasYAML = true
		}
	}
	switch {
	case hasJSON && hasEventStream:
		return "json_or_event_stream"
	case hasJSON:
		return "json"
	case hasYAML:
		return "yaml"
	case hasEmpty:
		return "empty"
	default:
		panic("Management operation has no supported success response")
	}
}

func operationRequestType(operation managementapi.OpenAPIOperation) string {
	if operation.RequestBody == nil {
		return "never"
	}
	media, found := operation.RequestBody.Content[managementapi.JSONMediaType]
	if !found {
		panic("generated Management request omits the Management JSON media type")
	}
	return renderTypeScriptSchema(media.Schema, "    ")
}

func operationQueryType(operation managementapi.OpenAPIOperation) string {
	return operationParameterObjectType(operation, "query", nil)
}

func operationPathType(operation managementapi.OpenAPIOperation) string {
	return operationParameterObjectType(operation, "path", nil)
}

func operationHeaderType(operation managementapi.OpenAPIOperation) string {
	return operationParameterObjectType(operation, "header", map[string]struct{}{
		"Accept":                        {},
		managementapi.HeaderNamespaceID: {},
	})
}

func operationParameterObjectType(
	operation managementapi.OpenAPIOperation,
	location string,
	skipped map[string]struct{},
) string {
	properties := make(map[string]managementapi.JSONSchema)
	required := make([]string, 0)
	for _, parameter := range operation.Parameters {
		if parameter.In != location {
			continue
		}
		if _, found := skipped[parameter.Name]; found {
			continue
		}
		properties[parameter.Name] = parameter.Schema
		if parameter.Required {
			required = append(required, parameter.Name)
		}
	}
	if len(properties) == 0 {
		return "never"
	}
	return renderTypeScriptSchema(managementapi.JSONSchema{
		Type:                 "object",
		Required:             required,
		Properties:           properties,
		AdditionalProperties: boolPointer(false),
	}, "    ")
}

func operationResponseType(operation managementapi.OpenAPIOperation) string {
	types := make(map[string]struct{})
	for _, status := range successfulStatuses(operation) {
		response := operation.Responses[status]
		if len(response.Content) == 0 {
			types["void"] = struct{}{}
			continue
		}
		for mediaType, media := range response.Content {
			if mediaType == managementapi.EventStreamMediaType {
				continue
			}
			types[renderTypeScriptSchema(media.Schema, "    ")] = struct{}{}
		}
	}
	result := make([]string, 0, len(types))
	for responseType := range types {
		result = append(result, responseType)
	}
	sort.Strings(result)
	if len(result) == 0 {
		panic("Management operation has no supported success response schema")
	}
	return strings.Join(result, " | ")
}

func operationRequiresOptions(
	operation managementapi.OperationContract,
	document managementapi.OpenAPIDocument,
) bool {
	if strings.Contains(operation.Path, "{") {
		return true
	}
	openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
	if openAPIOperation.RequestBody != nil && openAPIOperation.RequestBody.Required {
		return true
	}
	for _, parameter := range openAPIOperation.Parameters {
		if !parameter.Required {
			continue
		}
		if parameter.In == "query" {
			return true
		}
		if parameter.In == "header" &&
			parameter.Name != "Accept" && parameter.Name != managementapi.HeaderNamespaceID {
			return true
		}
	}
	return false
}

func optionalDefault(operation managementapi.OperationContract, document managementapi.OpenAPIDocument) string {
	if operationRequiresOptions(operation, document) {
		return ""
	}
	return " = {}"
}

func boolPointer(value bool) *bool {
	return &value
}
