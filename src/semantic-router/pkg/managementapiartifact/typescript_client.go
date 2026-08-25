package managementapiartifact

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

func renderAgentClient(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	agentOperations := sortedAgentOperations(operations)
	renderAgentOperationTypes(output, document, agentOperations)
	renderAgentTypeSupport(output)
	renderAgentClientInterface(output, document, agentOperations)
	renderAgentClientFactory(output, document, agentOperations)
}

func sortedAgentOperations(operations []managementapi.OperationContract) []managementapi.OperationContract {
	agentOperations := make([]managementapi.OperationContract, 0)
	for _, operation := range operations {
		if isAgentClientOperation(operation) {
			agentOperations = append(agentOperations, operation)
		}
	}
	sort.Slice(agentOperations, func(i, j int) bool {
		return agentOperations[i].OperationID < agentOperations[j].OperationID
	})
	return agentOperations
}

func renderAgentOperationTypes(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("\nexport interface ManagementApiAgentOperationTypes {\n")
	for _, operation := range operations {
		openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
		fmt.Fprintf(output, "  %s: {\n", operation.OperationID)
		fmt.Fprintf(output, "    body: %s\n", operationRequestType(openAPIOperation))
		fmt.Fprintf(output, "    query: %s\n", operationQueryType(openAPIOperation))
		fmt.Fprintf(output, "    response: %s\n", operationResponseType(openAPIOperation))
		output.WriteString("  }\n")
	}
	output.WriteString("}\n\n")
}

func renderAgentTypeSupport(output *strings.Builder) {
	output.WriteString("export type ManagementApiAgentOperationId = keyof ManagementApiAgentOperationTypes\n")
	output.WriteString("export type ManagementApiAgentRequestBody<OperationId extends ManagementApiAgentOperationId> =\n")
	output.WriteString("  ManagementApiAgentOperationTypes[OperationId]['body']\n")
	output.WriteString("export type ManagementApiAgentQuery<OperationId extends ManagementApiAgentOperationId> =\n")
	output.WriteString("  ManagementApiAgentOperationTypes[OperationId]['query']\n")
	output.WriteString("export type ManagementApiAgentResponse<OperationId extends ManagementApiAgentOperationId> =\n")
	output.WriteString("  ManagementApiAgentOperationTypes[OperationId]['response']\n\n")

	output.WriteString("type ManagementApiAgentPathOption<OperationId extends ManagementApiAgentOperationId> = [\n")
	output.WriteString("  ManagementApiPathParameters<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { pathParameters?: never }\n")
	output.WriteString("  : { pathParameters: ManagementApiPathParameters<OperationId> }\n\n")
	output.WriteString("type ManagementApiAgentBodyOption<OperationId extends ManagementApiAgentOperationId> = [\n")
	output.WriteString("  ManagementApiAgentRequestBody<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { body?: never }\n")
	output.WriteString("  : { body: ManagementApiAgentRequestBody<OperationId> }\n\n")
	output.WriteString("type ManagementApiAgentQueryOption<OperationId extends ManagementApiAgentOperationId> = [\n")
	output.WriteString("  ManagementApiAgentQuery<OperationId>,\n")
	output.WriteString("] extends [never]\n")
	output.WriteString("  ? { query?: never }\n")
	output.WriteString("  : { query?: ManagementApiAgentQuery<OperationId> }\n\n")

	output.WriteString("export type ManagementApiAgentRequestOptions<OperationId extends ManagementApiAgentOperationId> = {\n")
	output.WriteString("  headers?: Record<string, string>\n")
	output.WriteString("  signal?: AbortSignal\n")
	output.WriteString("  namespace?: string | null\n")
	output.WriteString("} & ManagementApiAgentPathOption<OperationId> &\n")
	output.WriteString("  ManagementApiAgentBodyOption<OperationId> &\n")
	output.WriteString("  ManagementApiAgentQueryOption<OperationId>\n\n")

	output.WriteString("export interface ManagementApiClientResponse<ResponseBody> {\n")
	output.WriteString("  data: ResponseBody\n")
	output.WriteString("  etag?: string\n")
	output.WriteString("}\n\n")
	output.WriteString("export interface ManagementApiAgentTransport {\n")
	output.WriteString("  request<OperationId extends ManagementApiAgentOperationId>(\n")
	output.WriteString("    operationId: OperationId,\n")
	output.WriteString("    options: ManagementApiAgentRequestOptions<OperationId>,\n")
	output.WriteString("  ): Promise<ManagementApiClientResponse<unknown>>\n")
	output.WriteString("}\n\n")
}

func renderAgentClientInterface(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("export interface ManagementApiAgentClient {\n")
	for _, operation := range operations {
		fmt.Fprintf(output, "  %s(\n", operation.OperationID)
		if operationRequiresOptions(operation, document) {
			fmt.Fprintf(output, "    options: ManagementApiAgentRequestOptions<%s>,\n", tsString(operation.OperationID))
		} else {
			fmt.Fprintf(output, "    options?: ManagementApiAgentRequestOptions<%s>,\n", tsString(operation.OperationID))
		}
		fmt.Fprintf(
			output,
			"  ): Promise<ManagementApiClientResponse<ManagementApiAgentResponse<%s>>>\n",
			tsString(operation.OperationID),
		)
	}
	output.WriteString("}\n\n")
}

func renderAgentClientFactory(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
	operations []managementapi.OperationContract,
) {
	output.WriteString("export function createManagementApiAgentClient(\n")
	output.WriteString("  transport: ManagementApiAgentTransport,\n")
	output.WriteString("): ManagementApiAgentClient {\n")
	output.WriteString("  return {\n")
	for _, operation := range operations {
		openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
		fmt.Fprintf(output, "    %s: async (options%s) => {\n", operation.OperationID, optionalDefault(operation, document))
		fmt.Fprintf(output, "      const response = await transport.request(%s, options)\n", tsString(operation.OperationID))
		if schema, found := successfulJSONResponseSchema(openAPIOperation); found {
			name, isReference := referencedSchemaName(schema)
			if !isReference {
				panic(fmt.Sprintf("generated Agent response %q is not a component schema", operation.OperationID))
			}
			fmt.Fprintf(output, "      return { ...response, data: assertManagementApiAgentSchema(%s, response.data) }\n", tsString(name))
		} else {
			output.WriteString("      return { ...response, data: undefined }\n")
		}
		output.WriteString("    },\n")
	}
	output.WriteString("  }\n")
	output.WriteString("}\n")
}

func isAgentClientOperation(operation managementapi.OperationContract) bool {
	return strings.HasPrefix(operation.Path, managementapi.BasePath+"/agent-") ||
		operation.Path == managementapi.BasePath+"/publication-plans/{plan}:commit" ||
		(operation.Method == managementapi.MethodGET && operation.Path == managementapi.BasePath+"/operations/{operationId}")
}

func successfulJSONResponseSchema(operation managementapi.OpenAPIOperation) (managementapi.JSONSchema, bool) {
	statuses := make([]string, 0, len(operation.Responses))
	for status := range operation.Responses {
		code, err := strconv.Atoi(status)
		if err == nil && code >= 200 && code < 300 {
			statuses = append(statuses, status)
		}
	}
	sort.Strings(statuses)
	for _, status := range statuses {
		media, found := operation.Responses[status].Content[managementapi.JSONMediaType]
		if found {
			return media.Schema, true
		}
	}
	return managementapi.JSONSchema{}, false
}

func referencedSchemaName(schema managementapi.JSONSchema) (string, bool) {
	if !strings.HasPrefix(schema.Ref, componentSchemaPrefix) {
		return "", false
	}
	return strings.TrimPrefix(schema.Ref, componentSchemaPrefix), true
}

func operationRequestType(operation managementapi.OpenAPIOperation) string {
	if operation.RequestBody == nil {
		return "never"
	}
	media, found := operation.RequestBody.Content[managementapi.JSONMediaType]
	if !found {
		panic("generated Agent request omits the Management JSON media type")
	}
	return renderTypeScriptSchema(media.Schema, "    ")
}

func operationQueryType(operation managementapi.OpenAPIOperation) string {
	properties := make(map[string]managementapi.JSONSchema)
	required := make([]string, 0)
	for _, parameter := range operation.Parameters {
		if parameter.In != "query" {
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
	schema, found := successfulJSONResponseSchema(operation)
	if !found {
		return "void"
	}
	return renderTypeScriptSchema(schema, "    ")
}

func operationRequiresOptions(
	operation managementapi.OperationContract,
	document managementapi.OpenAPIDocument,
) bool {
	if strings.Contains(operation.Path, "{") {
		return true
	}
	openAPIOperation := document.Paths[operation.Path][strings.ToLower(string(operation.Method))]
	return openAPIOperation.RequestBody != nil
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
