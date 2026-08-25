package managementapiartifact

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const componentSchemaPrefix = "#/components/schemas/"

var typeScriptIdentifier = regexp.MustCompile(`^[A-Za-z_$][A-Za-z0-9_$]*$`)

func renderManagementSchemaTypes(
	output *strings.Builder,
	document managementapi.OpenAPIDocument,
) {
	schemas := document.Components.Schemas
	names := sortedSchemaNames(schemas)

	output.WriteString("\n\n// Resource types and validators generated from every canonical OpenAPI schema.\n")
	for _, name := range names {
		typeName := name
		if name == "AgentEvent" {
			typeName = "ManagementApiAgentEvent"
		}
		fmt.Fprintf(output, "export type %s = %s\n\n", typeName, renderTypeScriptSchema(schemas[name], ""))
	}

	output.WriteString("export type AgentEventType = ManagementApiAgentEvent['type']\n")
	output.WriteString("export type AgentEvent<EventType extends AgentEventType = AgentEventType> =\n")
	output.WriteString("  Extract<ManagementApiAgentEvent, { type: EventType }>\n\n")
	output.WriteString("export type AgentEventPayloadByType = {\n")
	for _, event := range agentEventPayloadTypes() {
		fmt.Fprintf(output, "  %s: %s\n", event.EventType, event.SchemaName)
	}
	output.WriteString("}\n\n")

	for _, alias := range agentTypeAliases() {
		fmt.Fprintf(output, "export type %s = %s\n", alias.Name, alias.TypeScript)
	}

	output.WriteString("\nexport type AgentPage<Resource> = { data: Array<Resource>; page: PageInfo }\n")
	output.WriteString("export type AgentStreamEvent = AgentEvent | AgentLiveModelStepEvent\n\n")

	output.WriteString("export interface ManagementApiSchemas {\n")
	for _, name := range names {
		typeName := name
		if name == "AgentEvent" {
			typeName = "ManagementApiAgentEvent"
		}
		fmt.Fprintf(output, "  %s: %s\n", name, typeName)
	}
	output.WriteString("}\n\n")
	output.WriteString("export type ManagementApiSchemaName = keyof ManagementApiSchemas\n\n")

	encoded, err := json.Marshal(schemas)
	if err != nil {
		panic(fmt.Sprintf("marshal generated Management API schemas: %v", err))
	}
	output.WriteString("interface ManagementApiRuntimeSchema {\n")
	output.WriteString("  '$ref'?: string\n")
	output.WriteString("  type?: string\n")
	output.WriteString("  format?: string\n")
	output.WriteString("  pattern?: string\n")
	output.WriteString("  minimum?: number\n")
	output.WriteString("  maximum?: number\n")
	output.WriteString("  minLength?: number\n")
	output.WriteString("  maxLength?: number\n")
	output.WriteString("  enum?: Array<string>\n")
	output.WriteString("  required?: Array<string>\n")
	output.WriteString("  properties?: Record<string, ManagementApiRuntimeSchema>\n")
	output.WriteString("  patternProperties?: Record<string, ManagementApiRuntimeSchema>\n")
	output.WriteString("  items?: ManagementApiRuntimeSchema\n")
	output.WriteString("  minItems?: number\n")
	output.WriteString("  maxItems?: number\n")
	output.WriteString("  uniqueItems?: boolean\n")
	output.WriteString("  oneOf?: Array<ManagementApiRuntimeSchema>\n")
	output.WriteString("  additionalProperties?: boolean\n")
	output.WriteString("}\n\n")
	fmt.Fprintf(output, "const MANAGEMENT_API_SCHEMAS = JSON.parse(\n  %s,\n) as Record<ManagementApiSchemaName, ManagementApiRuntimeSchema>\n\n", tsString(string(encoded)))
	renderRuntimeSchemaValidator(output)
}

type agentTypeAlias struct {
	Name       string
	TypeScript string
}

func agentTypeAliases() []agentTypeAlias {
	return []agentTypeAlias{
		{Name: "AgentApprovalRequestPayload", TypeScript: "AgentApprovalRequestEventPayload"},
		{Name: "AgentApprovalResultPayload", TypeScript: "AgentApprovalResultEventPayload"},
		{Name: "AgentAssistantDeltaPayload", TypeScript: "AgentAssistantDeltaEventPayload"},
		{Name: "AgentCancellationPayload", TypeScript: "AgentCancellationEventPayload"},
		{Name: "AgentCheckpointPayload", TypeScript: "AgentContextCheckpointEventPayload"},
		{Name: "AgentLiveModelStepPhase", TypeScript: "AgentLiveModelStepEvent['phase']"},
		{Name: "AgentProfileInput", TypeScript: "AgentProfileCreateRequest"},
		{Name: "AgentProgressPayload", TypeScript: "AgentProgressEventPayload"},
		{Name: "AgentPublicationCommitInput", TypeScript: "AgentPublicationCommitRequest"},
		{Name: "AgentResourceStatus", TypeScript: "AgentProfile['status']"},
		{Name: "AgentSessionInput", TypeScript: "AgentSessionCreateRequest"},
		{Name: "AgentSessionMode", TypeScript: "AgentSession['mode']"},
		{Name: "AgentSessionStatus", TypeScript: "AgentSession['status']"},
		{Name: "AgentSkillInput", TypeScript: "AgentSkillCreateRequest"},
		{Name: "AgentTargetKind", TypeScript: "AgentTarget['kind']"},
		{Name: "AgentTerminalPayload", TypeScript: "AgentTerminalEventPayload"},
		{Name: "AgentToolClassification", TypeScript: "AgentToolDefinition['class']"},
		{Name: "AgentToolCredentialInput", TypeScript: "AgentToolCredentialCreateRequest"},
		{Name: "AgentToolIdempotency", TypeScript: "AgentToolDefinition['idempotency']"},
		{Name: "AgentToolRequestPayload", TypeScript: "AgentToolRequestEventPayload"},
		{Name: "AgentToolResultPayload", TypeScript: "AgentToolResultEventPayload"},
		{Name: "AgentToolSourceAvailability", TypeScript: "AgentToolSource['availability']"},
		{Name: "AgentToolSourceInput", TypeScript: "AgentToolSourceCreateRequest"},
		{Name: "AgentToolSourcePatchInput", TypeScript: "AgentToolSourcePatchRequest"},
		{Name: "AgentTurnStatus", TypeScript: "AgentTurn['status']"},
		{Name: "AgentUserInputPayload", TypeScript: "AgentUserInputEventPayload"},
	}
}

type agentEventPayloadType struct {
	EventType  string
	SchemaName string
}

func agentEventPayloadTypes() []agentEventPayloadType {
	return []agentEventPayloadType{
		{EventType: "user_input", SchemaName: "AgentUserInputEventPayload"},
		{EventType: "assistant_delta", SchemaName: "AgentAssistantDeltaEventPayload"},
		{EventType: "tool_request", SchemaName: "AgentToolRequestEventPayload"},
		{EventType: "tool_result", SchemaName: "AgentToolResultEventPayload"},
		{EventType: "model_step_summary", SchemaName: "AgentModelStepSummaryEventPayload"},
		{EventType: "progress", SchemaName: "AgentProgressEventPayload"},
		{EventType: "context_checkpoint", SchemaName: "AgentContextCheckpointEventPayload"},
		{EventType: "approval_request", SchemaName: "AgentApprovalRequestEventPayload"},
		{EventType: "approval_result", SchemaName: "AgentApprovalResultEventPayload"},
		{EventType: "cancellation", SchemaName: "AgentCancellationEventPayload"},
		{EventType: "terminal", SchemaName: "AgentTerminalEventPayload"},
	}
}

func sortedSchemaNames(schemas map[string]managementapi.JSONSchema) []string {
	names := make([]string, 0, len(schemas))
	for name := range schemas {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func renderTypeScriptSchema(schema managementapi.JSONSchema, indent string) string {
	if schema.Ref != "" {
		return strings.TrimPrefix(schema.Ref, componentSchemaPrefix)
	}
	if len(schema.OneOf) > 0 {
		branches := make([]string, 0, len(schema.OneOf))
		for _, branch := range schema.OneOf {
			branches = append(branches, renderTypeScriptSchema(branch, indent))
		}
		return strings.Join(branches, " | ")
	}
	if len(schema.Enum) > 0 {
		values := make([]string, 0, len(schema.Enum))
		for _, value := range schema.Enum {
			values = append(values, tsString(value))
		}
		return strings.Join(values, " | ")
	}

	switch schema.Type {
	case "string":
		return "string"
	case "integer", "number":
		return "number"
	case "boolean":
		return "boolean"
	case "null":
		return "null"
	case "array":
		if schema.Items == nil {
			return "Array<unknown>"
		}
		return "Array<" + renderTypeScriptSchema(*schema.Items, indent) + ">"
	case "object":
		return renderTypeScriptObject(schema, indent)
	case "":
		return "unknown"
	default:
		panic(fmt.Sprintf("unsupported JSON Schema type %q in generated Management contract", schema.Type))
	}
}

func renderTypeScriptObject(schema managementapi.JSONSchema, indent string) string {
	if len(schema.Properties) == 0 {
		if len(schema.PatternProperties) == 1 {
			for _, value := range schema.PatternProperties {
				return "Record<string, " + renderTypeScriptSchema(value, indent) + ">"
			}
		}
		if schema.AdditionalProperties != nil && *schema.AdditionalProperties {
			return "Record<string, unknown>"
		}
		return "Record<string, never>"
	}

	required := make(map[string]struct{}, len(schema.Required))
	for _, name := range schema.Required {
		required[name] = struct{}{}
	}
	names := make([]string, 0, len(schema.Properties))
	for name := range schema.Properties {
		names = append(names, name)
	}
	sort.Strings(names)

	var output strings.Builder
	output.WriteString("{\n")
	childIndent := indent + "  "
	for _, name := range names {
		optional := "?"
		if _, found := required[name]; found {
			optional = ""
		}
		fmt.Fprintf(
			&output,
			"%s%s%s: %s\n",
			childIndent,
			typeScriptPropertyName(name),
			optional,
			renderTypeScriptSchema(schema.Properties[name], childIndent),
		)
	}
	output.WriteString(indent + "}")
	if schema.AdditionalProperties != nil && *schema.AdditionalProperties {
		output.WriteString(" & Record<string, unknown>")
	}
	return output.String()
}

func typeScriptPropertyName(name string) string {
	if typeScriptIdentifier.MatchString(name) {
		return name
	}
	return tsString(name)
}

func renderRuntimeSchemaValidator(output *strings.Builder) {
	output.WriteString(`function managementApiSchemaReference(schema: ManagementApiRuntimeSchema): ManagementApiRuntimeSchema {
  if (!schema.$ref) return schema
  const name = schema.$ref.replace('#/components/schemas/', '') as ManagementApiSchemaName
  return MANAGEMENT_API_SCHEMAS[name]
}

function managementApiStringMatchesFormat(value: string, format?: string): boolean {
  if (format === 'date-time') return Number.isFinite(Date.parse(value))
  return true
}

function managementApiSchemaMatches(schema: ManagementApiRuntimeSchema, value: unknown): boolean {
  schema = managementApiSchemaReference(schema)
  if (schema.oneOf) {
    return schema.oneOf.filter((branch) => managementApiSchemaMatches(branch, value)).length === 1
  }
  if (schema.enum && !schema.enum.includes(value as string)) return false

  if (schema.type === 'null') return value === null
  if (schema.type === 'string') {
    return (
      typeof value === 'string' &&
      (schema.minLength === undefined || value.length >= schema.minLength) &&
      (schema.maxLength === undefined || value.length <= schema.maxLength) &&
      (schema.pattern === undefined || new RegExp(schema.pattern).test(value)) &&
      managementApiStringMatchesFormat(value, schema.format)
    )
  }
  if (schema.type === 'integer' || schema.type === 'number') {
    return (
      typeof value === 'number' &&
      Number.isFinite(value) &&
      (schema.type !== 'integer' || Number.isSafeInteger(value)) &&
      (schema.minimum === undefined || value >= schema.minimum) &&
      (schema.maximum === undefined || value <= schema.maximum)
    )
  }
  if (schema.type === 'boolean') return typeof value === 'boolean'
  if (schema.type === 'array') {
    return (
      Array.isArray(value) &&
      (schema.minItems === undefined || value.length >= schema.minItems) &&
      (schema.maxItems === undefined || value.length <= schema.maxItems) &&
      (!schema.uniqueItems || new Set(value.map((item) => JSON.stringify(item))).size === value.length) &&
      (!schema.items || value.every((item) => managementApiSchemaMatches(schema.items!, item)))
    )
  }
  if (schema.type === 'object') {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return false
    const record = value as Record<string, unknown>
    if (schema.required?.some((name) => !Object.prototype.hasOwnProperty.call(record, name))) {
      return false
    }
    for (const [name, item] of Object.entries(record)) {
      const property = schema.properties?.[name]
      if (property) {
        if (!managementApiSchemaMatches(property, item)) return false
        continue
      }
      const pattern = Object.entries(schema.patternProperties ?? {}).find(([expression]) =>
        new RegExp(expression).test(name),
      )
      if (pattern) {
        if (!managementApiSchemaMatches(pattern[1], item)) return false
        continue
      }
      if (schema.additionalProperties === false) return false
    }
    return true
  }
  return schema.type === undefined
}

export function assertManagementApiSchema<SchemaName extends ManagementApiSchemaName>(
  schemaName: SchemaName,
  value: unknown,
): ManagementApiSchemas[SchemaName] {
  if (!managementApiSchemaMatches(MANAGEMENT_API_SCHEMAS[schemaName], value)) {
    throw new Error('Router returned a response that does not match ' + schemaName + '.')
  }
  return value as ManagementApiSchemas[SchemaName]
}
`)
}
