// Package configschema publishes the machine-readable canonical Router
// configuration contract generated from pkg/config Go types and registries.
package configschema

import (
	"bytes"
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"strings"

	"github.com/invopop/jsonschema"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	ContractVersion    = "vllm-sr/config-schema/v1"
	ConfigVersion      = routerconfig.CanonicalConfigVersion
	SchemaID           = "https://vllm-sr.ai/schemas/router-config-v0.3.schema.json"
	SchemaEndpoint     = "/api/v1/config/schema"
	ValidationEndpoint = "/api/v1/config/validate"
)

//go:generate go run ../../../../tools/configschema/main.go --repository-root ../../../..

//go:embed router-config-v0.3.schema.json
var embeddedSchema []byte

type validationContract struct {
	Structural string `json:"structural"`
	Semantic   string `json:"semantic"`
	Endpoint   string `json:"endpoint"`
}

type signalSurface struct {
	routerconfig.SignalCatalogEntry
	SchemaRef string `json:"schema_ref"`
}

type algorithmSurface struct {
	routerconfig.AlgorithmCatalogEntry
	SchemaRef string `json:"schema_ref,omitempty"`
}

type pluginSurface struct {
	routerconfig.DecisionPluginCatalogEntry
	SchemaRef string `json:"schema_ref"`
}

type projectionSurface struct {
	Collection  string `json:"collection"`
	DisplayName string `json:"display_name"`
	SchemaRef   string `json:"schema_ref"`
}

type globalSectionSurface struct {
	Key         string   `json:"key"`
	Layer       string   `json:"layer"`
	DisplayName string   `json:"display_name"`
	Path        []string `json:"path"`
}

type schemaExtension struct {
	ContractVersion      string                 `json:"contract_version"`
	ConfigVersion        string                 `json:"config_version"`
	SchemaEndpoint       string                 `json:"schema_endpoint"`
	Validation           validationContract     `json:"validation"`
	Signals              []signalSurface        `json:"signals"`
	Algorithms           []algorithmSurface     `json:"algorithms"`
	Plugins              []pluginSurface        `json:"plugins"`
	Projections          []projectionSurface    `json:"projections"`
	ProjectionInputTypes []string               `json:"projection_input_types"`
	GlobalSections       []globalSectionSurface `json:"global_sections"`
}

// Document returns an immutable copy of the checked-in generated schema.
func Document() []byte {
	return bytes.Clone(embeddedSchema)
}

// ETag returns a strong content identity for HTTP and consumer caches.
func ETag() string {
	digest := sha256.Sum256(embeddedSchema)
	return `"sha256:` + hex.EncodeToString(digest[:]) + `"`
}

// GenerateFromSource reflects the canonical Go configuration types and adds
// descriptions from their source comments. repositoryRoot must be the project
// root, not the Go module directory.
func GenerateFromSource(repositoryRoot string) ([]byte, error) {
	reflector := &jsonschema.Reflector{
		Anonymous:                  true,
		ExpandedStruct:             true,
		FieldNameTag:               "yaml",
		RequiredFromJSONSchemaTags: true,
		Mapper:                     schemaTypeMapper,
	}
	moduleRoot := strings.TrimSuffix(repositoryRoot, "/") + "/src/semantic-router"
	if err := reflector.AddGoComments(
		"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config",
		moduleRoot+"/pkg/config",
	); err != nil {
		return nil, fmt.Errorf("load config Go comments: %w", err)
	}
	if err := reflector.AddGoComments(
		"github.com/vllm-project/semantic-router/src/semantic-router/pkg/catalog",
		moduleRoot+"/pkg/catalog",
	); err != nil {
		return nil, fmt.Errorf("load catalog Go comments: %w", err)
	}

	// Publish the steady-state Router contract. CanonicalConfigDocument also
	// carries the Dashboard's transient setup marker, which is intentionally not
	// part of the runtime or user-authored configuration schema.
	schema := reflector.Reflect(&routerconfig.CanonicalConfig{})
	schema.ID = jsonschema.ID(SchemaID)
	schema.Title = "vLLM Semantic Router canonical configuration"
	schema.Description = "Canonical v0.3 Router configuration. JSON Schema validates document structure; the Router validation endpoint applies cross-field, reference, and runtime semantic checks."

	addRecipeRoutingDefinition(schema)
	setCoreEnums(schema)
	setMemoryPersistenceBounds(schema)

	pluginRefs, err := addPluginDefinitions(reflector, schema)
	if err != nil {
		return nil, err
	}
	addPluginPayloadConditions(schema, pluginRefs)
	extension, err := buildExtension(schema, pluginRefs)
	if err != nil {
		return nil, err
	}
	if schema.Extras == nil {
		schema.Extras = map[string]any{}
	}
	schema.Extras["x-vllm-sr"] = extension

	payload, err := json.MarshalIndent(schema, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode config schema: %w", err)
	}
	return append(payload, '\n'), nil
}

// GenerateTypeScriptContract publishes a typed Dashboard entrypoint for the
// generated JSON artifact. Runtime inventories remain in the schema extension
// instead of being copied into another source file.
func GenerateTypeScriptContract(schemaDocument []byte) ([]byte, error) {
	var document map[string]json.RawMessage
	if err := json.Unmarshal(schemaDocument, &document); err != nil {
		return nil, fmt.Errorf("decode generated schema: %w", err)
	}
	if _, ok := document["x-vllm-sr"]; !ok {
		return nil, fmt.Errorf("generated schema is missing x-vllm-sr")
	}
	return []byte(`// Code generated by tools/configschema; DO NOT EDIT.
// The Dashboard imports the one repository-owned schema artifact directly;
// production discovery still uses the deployed Router schema endpoint.
import routerConfigSchema from '../../../../src/semantic-router/pkg/configschema/router-config-v0.3.schema.json'

export const ROUTER_CONFIG_SCHEMA = routerConfigSchema
export const ROUTER_CONFIG_EXTENSION = ROUTER_CONFIG_SCHEMA['x-vllm-sr']

export type SignalType = (typeof ROUTER_CONFIG_EXTENSION.signals)[number]['type']
export type AlgorithmType = (typeof ROUTER_CONFIG_EXTENSION.algorithms)[number]['type']
export type PluginType = (typeof ROUTER_CONFIG_EXTENSION.plugins)[number]['type']

export const SIGNAL_TYPES = ROUTER_CONFIG_EXTENSION.signals.map((surface) => surface.type)
export const DECISION_SIGNAL_TYPES = [
  ...ROUTER_CONFIG_EXTENSION.signals
    .filter((surface) => surface.decision_referenceable)
    .map((surface) => surface.type),
  'projection',
]
export const ALGORITHM_TYPES = ROUTER_CONFIG_EXTENSION.algorithms.map((surface) => surface.type)
export const PLUGIN_TYPES = ROUTER_CONFIG_EXTENSION.plugins.map((surface) => surface.type)
`), nil
}

func schemaTypeMapper(value reflect.Type) *jsonschema.Schema {
	if value == reflect.TypeOf(routerconfig.StructuredPayload{}) {
		return &jsonschema.Schema{
			Type:                 "object",
			AdditionalProperties: jsonschema.TrueSchema,
			Description:          "Plugin-specific structured configuration. Use the plugin surface schema selected by the sibling type field.",
		}
	}
	if value == reflect.TypeOf(routerconfig.CompressionTokenLimit{}) {
		return &jsonschema.Schema{
			Description: "A non-negative token count or the literal auto.",
			OneOf: []*jsonschema.Schema{
				{Type: "integer", Minimum: json.Number("0")},
				{Type: "string", Const: "auto"},
			},
		}
	}
	return nil
}

// CanonicalRecipe deliberately reuses CanonicalRouting in Go so normalization
// can share code with the top-level profile. The public recipe document omits
// modelCards, so publish that contextual restriction as its own schema.
func addRecipeRoutingDefinition(root *jsonschema.Schema) {
	base := root.Definitions["CanonicalRouting"]
	recipe := root.Definitions["CanonicalRecipe"]
	if base == nil || base.Properties == nil || recipe == nil || recipe.Properties == nil {
		return
	}
	properties := jsonschema.NewProperties()
	for pair := base.Properties.Oldest(); pair != nil; pair = pair.Next() {
		if pair.Key != "modelCards" {
			properties.Set(pair.Key, pair.Value)
		}
	}
	root.Definitions["CanonicalRecipeRouting"] = &jsonschema.Schema{
		Type:                 "object",
		Description:          "Recipe-owned routing profile. Model cards remain in the shared top-level catalog.",
		Properties:           properties,
		AdditionalProperties: jsonschema.FalseSchema,
	}
	if routing, ok := recipe.Properties.Get("routing"); ok {
		routing.Ref = "#/$defs/CanonicalRecipeRouting"
	}
}

func setCoreEnums(root *jsonschema.Schema) {
	setPropertyConst(root, root, "version", ConfigVersion)
	setDefinitionPropertyEnum(root, "CanonicalRouterGlobal", "config_source", []string{
		string(routerconfig.ConfigSourceFile),
		string(routerconfig.ConfigSourceKubernetes),
	})
	setDefinitionPropertyEnum(root, "AlgorithmConfig", "type", routerconfig.SupportedDecisionAlgorithmTypes())
	setDefinitionPropertyEnum(root, "DecisionPlugin", "type", routerconfig.SupportedDecisionPluginTypes())
	conditionTypes := append(routerconfig.SupportedDecisionSignalTypes(), routerconfig.SignalTypeProjection)
	setDefinitionPropertyEnum(root, "RuleNode", "type", conditionTypes)
	setDefinitionPropertyEnum(root, "RuleNode", "on_unknown", []string{
		string(routerconfig.RuleOnUnknownNoMatch),
		string(routerconfig.RuleOnUnknownMatch),
		string(routerconfig.RuleOnUnknownFailRequest),
	})
	setDefinitionPropertyEnum(root, "ProjectionScoreInput", "type", routerconfig.SupportedProjectionInputTypes())
	setDefinitionPropertyEnum(root, "InputModalityRule", "modality", routerconfig.SupportedInputModalities())
	setDefinitionPropertyEnum(root, "CanonicalRouting", "strategy", []string{
		string(routerconfig.RoutingStrategyPriority),
		string(routerconfig.RoutingStrategyConfidence),
	})
	setDefinitionPropertyEnum(root, "CanonicalRecipeRouting", "strategy", []string{
		string(routerconfig.RoutingStrategyPriority),
		string(routerconfig.RoutingStrategyConfidence),
	})
	setDefinitionPropertyEnum(root, "RouterLearningAdaptationConfig", "candidate_set", []string{
		"",
		routerconfig.RouterLearningCandidateSetDecision,
		routerconfig.RouterLearningCandidateSetTier,
		routerconfig.RouterLearningCandidateSetGlobal,
	})
	setDefinitionPropertyEnum(root, "RouterLearningAdaptationConfig", "strategy", []string{
		"",
		routerconfig.RouterLearningStrategyRoutingSampling,
	})
	setDefinitionPropertyEnum(root, "RouterLearningProtectionConfig", "scope", []string{
		"",
		routerconfig.RouterLearningScopeConversation,
		routerconfig.RouterLearningScopeSession,
	})
	setDefinitionPropertyEnum(root, "RouterLearningStateStoreConfig", "backend", []string{"", "local", "redis"})
	for _, definition := range []string{
		"DecisionAdaptationsConfig",
		"DecisionLearningAdaptationConfig",
		"DecisionLearningProtectionConfig",
	} {
		setDefinitionPropertyEnum(root, definition, "mode", []string{
			"",
			routerconfig.DecisionAdaptationModeApply,
			routerconfig.DecisionAdaptationModeObserve,
			routerconfig.DecisionAdaptationModeBypass,
		})
	}
	setDefinitionPropertyEnum(root, "DecisionLearningAdaptationConfig", "candidate_set", []string{
		"",
		routerconfig.RouterLearningCandidateSetDecision,
		routerconfig.RouterLearningCandidateSetTier,
		routerconfig.RouterLearningCandidateSetGlobal,
	})
	setDefinitionPropertyEnum(root, "DecisionAction", "type", []string{routerconfig.DecisionActionRoute})

	for _, field := range []string{"idle_timeout_seconds", "min_turns_before_switch", "switch_margin", "stability_weight"} {
		setDefinitionPropertyMinimum(root, "RouterLearningProtectionTuning", field, 0)
	}
	for _, field := range []string{"ttl_seconds", "timeout_ms"} {
		setDefinitionPropertyMinimum(root, "RouterLearningStateStoreConfig", field, 0)
	}
	setDefinitionPropertyMinimum(root, "RouterLearningRedisStateStoreConfig", "database", 0)
	for _, field := range []string{"stability_weight", "switch_margin"} {
		setDefinitionPropertyMinimum(root, "DecisionLearningProtectionConfig", field, 0)
	}
	for _, field := range []string{"session", "conversation"} {
		setDefinitionPropertyMinLength(root, "RouterLearningIdentityHeadersConfig", field, 1)
	}
	setDefinitionPropertyMinLength(root, "RouterLearningRedisStateStoreConfig", "address", 1)
	addRouterLearningStateStoreCondition(root)
}

func setPropertyConst(_ *jsonschema.Schema, owner *jsonschema.Schema, property string, value any) {
	if owner == nil || owner.Properties == nil {
		return
	}
	if field, ok := owner.Properties.Get(property); ok {
		field.Const = value
	}
}

func setDefinitionPropertyEnum(root *jsonschema.Schema, definition, property string, values []string) {
	owner := root.Definitions[definition]
	if owner == nil || owner.Properties == nil {
		return
	}
	field, ok := owner.Properties.Get(property)
	if !ok {
		return
	}
	field.Enum = make([]any, len(values))
	for index, value := range values {
		field.Enum[index] = value
	}
}

func definitionProperty(root *jsonschema.Schema, definition, property string) *jsonschema.Schema {
	owner := root.Definitions[definition]
	if owner == nil || owner.Properties == nil {
		return nil
	}
	field, ok := owner.Properties.Get(property)
	if !ok {
		return nil
	}
	return field
}

func setDefinitionPropertyMinimum(root *jsonschema.Schema, definition, property string, value int) {
	if field := definitionProperty(root, definition, property); field != nil {
		field.Minimum = json.Number(fmt.Sprintf("%d", value))
	}
}

func setDefinitionPropertyMinLength(root *jsonschema.Schema, definition, property string, value uint64) {
	if field := definitionProperty(root, definition, property); field != nil {
		field.MinLength = &value
	}
}

func addRouterLearningStateStoreCondition(root *jsonschema.Schema) {
	owner := root.Definitions["RouterLearningStateStoreConfig"]
	redis := root.Definitions["RouterLearningRedisStateStoreConfig"]
	if owner == nil || redis == nil {
		return
	}
	redis.Required = []string{"address"}
	conditionProperties := jsonschema.NewProperties()
	conditionProperties.Set("backend", &jsonschema.Schema{Const: "redis"})
	owner.AllOf = append(owner.AllOf, &jsonschema.Schema{
		If: &jsonschema.Schema{
			Properties: conditionProperties,
			Required:   []string{"backend"},
		},
		Then: &jsonschema.Schema{Required: []string{"redis"}},
	})
}

func addPluginDefinitions(
	reflector *jsonschema.Reflector,
	root *jsonschema.Schema,
) (map[string]string, error) {
	refs := make(map[string]string)
	samples := routerconfig.DecisionPluginSchemaSamples()
	for _, plugin := range routerconfig.DecisionPluginCatalog() {
		sample, ok := samples[plugin.Type]
		if !ok {
			return nil, fmt.Errorf("plugin %q has no validation payload factory", plugin.Type)
		}
		reflector.ExpandedStruct = false
		pluginSchema := reflector.Reflect(sample)
		reflector.ExpandedStruct = true
		for name, definition := range pluginSchema.Definitions {
			if root.Definitions[name] != nil {
				// A nested payload type can be reachable from several plugins. A
				// single reflector and Go package own both definitions, so the
				// first generated copy is canonical.
				continue
			}
			root.Definitions[name] = definition
		}
		refs[plugin.Type] = pluginSchema.Ref
	}
	return refs, nil
}

func addPluginPayloadConditions(root *jsonschema.Schema, pluginRefs map[string]string) {
	plugin := root.Definitions["DecisionPlugin"]
	if plugin == nil {
		return
	}
	for _, entry := range routerconfig.DecisionPluginCatalog() {
		ref := pluginRefs[entry.Type]
		if ref == "" {
			continue
		}
		conditionProperties := jsonschema.NewProperties()
		conditionProperties.Set("type", &jsonschema.Schema{Const: entry.Type})
		payloadProperties := jsonschema.NewProperties()
		payloadProperties.Set("configuration", &jsonschema.Schema{Ref: ref})
		plugin.AllOf = append(plugin.AllOf, &jsonschema.Schema{
			If: &jsonschema.Schema{
				Properties: conditionProperties,
				Required:   []string{"type"},
			},
			Then: &jsonschema.Schema{
				Properties: payloadProperties,
			},
		})
	}
}

func buildExtension(root *jsonschema.Schema, pluginRefs map[string]string) (schemaExtension, error) {
	signals := routerconfig.SignalCatalog()
	if err := validateSignalCatalog(signals); err != nil {
		return schemaExtension{}, err
	}
	extension := schemaExtension{
		ContractVersion: ContractVersion,
		ConfigVersion:   ConfigVersion,
		SchemaEndpoint:  SchemaEndpoint,
		Validation: validationContract{
			Structural: "json-schema-draft-2020-12",
			Semantic:   "router",
			Endpoint:   ValidationEndpoint,
		},
		ProjectionInputTypes: routerconfig.SupportedProjectionInputTypes(),
	}

	signalOwner := root.Definitions["CanonicalSignals"]
	signalCollections := make([]string, 0, len(signals))
	for _, entry := range signals {
		signalCollections = append(signalCollections, entry.Collection)
	}
	if err := requirePropertyCatalogCoverage(signalOwner, "signal", signalCollections); err != nil {
		return schemaExtension{}, err
	}
	for _, entry := range signals {
		ref, err := arrayItemRef(signalOwner, entry.Collection)
		if err != nil {
			return schemaExtension{}, fmt.Errorf("signal %q: %w", entry.Type, err)
		}
		extension.Signals = append(extension.Signals, signalSurface{SignalCatalogEntry: entry, SchemaRef: ref})
	}

	algorithmOwner := root.Definitions["AlgorithmConfig"]
	algorithmFields := make([]string, 0, len(routerconfig.DecisionAlgorithmCatalog()))
	for _, entry := range routerconfig.DecisionAlgorithmCatalog() {
		if entry.ConfigField != "" {
			algorithmFields = append(algorithmFields, entry.ConfigField)
		}
	}
	if err := requirePropertyCatalogCoverage(
		algorithmOwner,
		"algorithm configuration",
		algorithmFields,
		"type",
		"minimum_candidates",
		"on_error",
	); err != nil {
		return schemaExtension{}, err
	}
	for _, entry := range routerconfig.DecisionAlgorithmCatalog() {
		ref := ""
		if entry.ConfigField != "" {
			var err error
			ref, err = propertyRef(algorithmOwner, entry.ConfigField)
			if err != nil {
				return schemaExtension{}, fmt.Errorf("algorithm %q: %w", entry.Type, err)
			}
		}
		extension.Algorithms = append(extension.Algorithms, algorithmSurface{AlgorithmCatalogEntry: entry, SchemaRef: ref})
	}

	for _, entry := range routerconfig.DecisionPluginCatalog() {
		ref := pluginRefs[entry.Type]
		if ref == "" {
			return schemaExtension{}, fmt.Errorf("plugin %q has no generated schema reference", entry.Type)
		}
		extension.Plugins = append(extension.Plugins, pluginSurface{DecisionPluginCatalogEntry: entry, SchemaRef: ref})
	}

	projectionOwner := root.Definitions["CanonicalProjections"]
	projectionCollections := make([]string, 0, len(routerconfig.ProjectionCatalog()))
	for _, entry := range routerconfig.ProjectionCatalog() {
		projectionCollections = append(projectionCollections, entry.Collection)
	}
	if err := requirePropertyCatalogCoverage(projectionOwner, "projection", projectionCollections); err != nil {
		return schemaExtension{}, err
	}
	for _, entry := range routerconfig.ProjectionCatalog() {
		ref, err := arrayItemRef(projectionOwner, entry.Collection)
		if err != nil {
			return schemaExtension{}, fmt.Errorf("projection %q: %w", entry.Collection, err)
		}
		extension.Projections = append(extension.Projections, projectionSurface{
			Collection: entry.Collection, DisplayName: entry.DisplayName, SchemaRef: ref,
		})
	}

	globalSections, err := buildGlobalSections(root)
	if err != nil {
		return schemaExtension{}, err
	}
	extension.GlobalSections = globalSections
	return extension, nil
}

func validateSignalCatalog(entries []routerconfig.SignalCatalogEntry) error {
	types := make(map[string]bool, len(entries))
	collections := make(map[string]bool, len(entries))
	observationKeys := make(map[string]bool, len(entries))
	for _, entry := range entries {
		if entry.Type == "" || entry.DisplayName == "" || entry.Collection == "" {
			return fmt.Errorf("signal catalog contains an incomplete entry: %#v", entry)
		}
		if types[entry.Type] || collections[entry.Collection] {
			return fmt.Errorf("signal catalog contains duplicate identity for %q", entry.Type)
		}
		types[entry.Type] = true
		collections[entry.Collection] = true

		if entry.DecisionReferenceable && entry.ObservationKey == "" {
			return fmt.Errorf("decision-referenceable signal %q has no observation key", entry.Type)
		}
		if entry.ObservationKey != "" {
			if observationKeys[entry.ObservationKey] {
				return fmt.Errorf("signal observation key %q is registered more than once", entry.ObservationKey)
			}
			observationKeys[entry.ObservationKey] = true
		}

		switch entry.ReferenceQualifier {
		case "":
			if len(entry.ReferenceSuffixes) != 0 {
				return fmt.Errorf("signal %q declares suffixes without a reference qualifier", entry.Type)
			}
		case routerconfig.SignalReferenceQualifierFixedSuffix:
			if len(entry.ReferenceSuffixes) == 0 {
				return fmt.Errorf("signal %q has a fixed-suffix qualifier without suffixes", entry.Type)
			}
		case routerconfig.SignalReferenceQualifierLabel:
			if len(entry.ReferenceSuffixes) != 0 {
				return fmt.Errorf("label-qualified signal %q must discover labels from its payload", entry.Type)
			}
		default:
			return fmt.Errorf("signal %q has unknown reference qualifier %q", entry.Type, entry.ReferenceQualifier)
		}
	}
	return nil
}

func requirePropertyCatalogCoverage(
	owner *jsonschema.Schema,
	surface string,
	registered []string,
	ignored ...string,
) error {
	if owner == nil || owner.Properties == nil {
		return fmt.Errorf("%s owner schema is missing", surface)
	}
	accounted := make(map[string]bool, len(registered)+len(ignored))
	for _, property := range ignored {
		accounted[property] = true
	}
	for _, property := range registered {
		if accounted[property] {
			return fmt.Errorf("%s property %q is registered more than once", surface, property)
		}
		if _, ok := owner.Properties.Get(property); !ok {
			return fmt.Errorf("registered %s property %q is missing from the Go config schema", surface, property)
		}
		accounted[property] = true
	}
	missing := []string{}
	for pair := owner.Properties.Oldest(); pair != nil; pair = pair.Next() {
		if !accounted[pair.Key] {
			missing = append(missing, pair.Key)
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("%s properties missing from the Go surface catalog: %s", surface, strings.Join(missing, ", "))
	}
	return nil
}

func buildGlobalSections(root *jsonschema.Schema) ([]globalSectionSurface, error) {
	global := root.Definitions["CanonicalGlobal"]
	if global == nil || global.Properties == nil {
		return nil, fmt.Errorf("global configuration schema is missing")
	}
	sections := []globalSectionSurface{}
	for layer := global.Properties.Oldest(); layer != nil; layer = layer.Next() {
		if layer.Key == "router" {
			sections = append(sections, globalSectionSurface{
				Key: "router", Layer: "router", DisplayName: "Router Core", Path: []string{"router"},
			})
			continue
		}

		owner := referencedDefinition(root, layer.Value.Ref)
		if owner == nil || owner.Properties == nil {
			return nil, fmt.Errorf("global layer %q has no object schema", layer.Key)
		}
		for section := owner.Properties.Oldest(); section != nil; section = section.Next() {
			if layer.Key != "model_catalog" || section.Key != "modules" {
				sections = append(sections, globalSectionSurface{
					Key: section.Key, Layer: layer.Key, DisplayName: humanizeSchemaName(section.Key),
					Path: []string{layer.Key, section.Key},
				})
				continue
			}

			modules := referencedDefinition(root, section.Value.Ref)
			if modules == nil || modules.Properties == nil {
				return nil, fmt.Errorf("global model modules schema is missing")
			}
			for module := modules.Properties.Oldest(); module != nil; module = module.Next() {
				sections = append(sections, globalSectionSurface{
					Key: module.Key, Layer: layer.Key, DisplayName: humanizeSchemaName(module.Key),
					Path: []string{layer.Key, "modules", module.Key},
				})
			}
		}
	}
	return sections, nil
}

func humanizeSchemaName(value string) string {
	parts := strings.Split(value, "_")
	for index, part := range parts {
		if part == "api" || part == "kb" {
			parts[index] = strings.ToUpper(part)
			continue
		}
		if part != "" {
			parts[index] = strings.ToUpper(part[:1]) + part[1:]
		}
	}
	return strings.Join(parts, " ")
}

func arrayItemRef(owner *jsonschema.Schema, property string) (string, error) {
	if owner == nil || owner.Properties == nil {
		return "", fmt.Errorf("owner schema is missing")
	}
	field, ok := owner.Properties.Get(property)
	if !ok || field.Items == nil || field.Items.Ref == "" {
		return "", fmt.Errorf("array property %q has no item schema reference", property)
	}
	return field.Items.Ref, nil
}

func propertyRef(owner *jsonschema.Schema, property string) (string, error) {
	if owner == nil || owner.Properties == nil {
		return "", fmt.Errorf("owner schema is missing")
	}
	field, ok := owner.Properties.Get(property)
	if !ok || field.Ref == "" {
		return "", fmt.Errorf("property %q has no schema reference", property)
	}
	return field.Ref, nil
}

func referencedDefinition(root *jsonschema.Schema, ref string) *jsonschema.Schema {
	const prefix = "#/$defs/"
	if !strings.HasPrefix(ref, prefix) {
		return nil
	}
	return root.Definitions[strings.TrimPrefix(ref, prefix)]
}
