package dsl

import (
	"encoding/json"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// routingYAMLDocument is the routing fragment the DSL owns: its recipes and
// entrypoints, and the facts its MODEL declarations state on providers.models[]
// entries (each model's transport stays in the base configuration).
type routingYAMLDocument struct {
	Providers   *routingFragmentProviders    `yaml:"providers,omitempty"`
	Recipes     []config.CanonicalRecipe     `yaml:"recipes,omitempty"`
	Entrypoints []config.CanonicalEntrypoint `yaml:"entrypoints,omitempty"`
}

type routingFragmentProviders struct {
	Models []config.CanonicalProviderModel `yaml:"models"`
}

// modelFactEntries are the providers.models[] entries stating the facts of
// MODEL declarations.
func modelFactEntries(cfg *config.RouterConfig) []config.CanonicalProviderModel {
	var models []config.CanonicalProviderModel
	for _, card := range config.ModelCardsFromRouterConfig(cfg) {
		models = append(models, config.CanonicalProviderModel{Name: card.Name, ModelFacts: card.ModelFacts})
	}
	return models
}

// EmitRoutingYAML compiles DSL source and emits the v0.5 routing fragment.
func EmitRoutingYAML(input string) ([]byte, []error) {
	cfg, errs := Compile(input)
	if len(errs) > 0 {
		return nil, errs
	}
	yamlBytes, err := EmitRoutingYAMLFromConfig(cfg)
	if err != nil {
		return nil, []error{err}
	}
	return yamlBytes, nil
}

// EmitRoutingYAMLFromConfig marshals only the DSL-owned routing surface.
func EmitRoutingYAMLFromConfig(cfg *config.RouterConfig) ([]byte, error) {
	if err := config.ValidateDecisionRuleLimits(cfg); err != nil {
		return nil, err
	}
	canonical := config.CanonicalConfigFromRouterConfig(cfg)
	doc := routingYAMLDocument{Recipes: canonical.Recipes, Entrypoints: canonical.Entrypoints}
	if models := modelFactEntries(cfg); len(models) > 0 {
		doc.Providers = &routingFragmentProviders{Models: models}
	}
	return yaml.Marshal(doc)
}

// DecompileRouting converts runtime config to the routing-only DSL contract.
func DecompileRouting(cfg *config.RouterConfig) (string, error) {
	if err := config.ValidateDecisionRuleLimits(cfg); err != nil {
		return "", err
	}
	d := &decompiler{cfg: cfg}
	d.pluginTemplates = make(map[string]*pluginTemplate)
	d.extractPluginTemplates()
	d.decompileRoutingStrategy()

	d.writeSection("SIGNALS")
	d.decompileSignals()

	models := config.ModelCardsFromRouterConfig(cfg)
	if len(models) > 0 {
		d.writeSection("MODELS")
		d.decompileRoutingModels(models)
	}

	if len(d.pluginTemplates) > 0 {
		d.writeSection("PLUGINS")
		d.decompilePluginTemplates()
	}

	d.writeSection("ROUTES")
	d.decompileDecisions()

	return d.sb.String(), nil
}

// DecompileRoutingToAST converts runtime config to a routing-only AST.
func DecompileRoutingToAST(cfg *config.RouterConfig) *Program {
	if err := config.ValidateDecisionRuleLimits(cfg); err != nil {
		return nil
	}
	d := &decompiler{cfg: cfg}
	prog := &Program{Strategy: string(cfg.Strategy), ModelBindings: cloneModelBindings(cfg.RecipeBindings()), RecipePlugins: cfg.Plugins, CandidateRequirements: cfg.CandidateRequirements.Clone()}
	d.appendSignalsToProgram(prog)
	d.appendModelsToProgram(prog)
	d.appendRoutesToProgram(prog)
	return prog
}

func (d *decompiler) decompileRoutingStrategy() {
	// A recipe view's ModelBindings carry the global bindings as well; those
	// stay in global.model_catalog.bindings.
	bindings := d.cfg.RecipeBindings()
	if d.cfg.Strategy == "" && len(bindings) == 0 && d.cfg.CandidateRequirements == nil && len(d.cfg.Plugins) == 0 {
		return
	}
	d.writeSection("ROUTING PROFILE")
	d.write("ROUTING {\n")
	d.decompileRoutingPolicies()
	if d.cfg.Strategy != "" {
		d.write("  strategy: %s\n", d.cfg.Strategy)
	}
	if len(bindings) > 0 {
		// Convert the canonical structs through their JSON tags, then use the
		// DSL formatter so object keys retain the grammar's identifier syntax.
		encoded, _ := json.Marshal(bindings)
		var fields map[string]interface{}
		_ = json.Unmarshal(encoded, &fields)
		d.write("  model_bindings: %s\n", formatPluginConfigValue(fields))
	}
	if len(d.cfg.Plugins) > 0 {
		d.write("  plugins: %s\n", formatRecipePlugins(d.cfg.Plugins))
	}
	d.write("}\n\n")
}

func (d *decompiler) appendSignalsToProgram(prog *Program) {
	d.appendCoreSignals(prog)
	d.appendOperationalSignals(prog)
	d.appendSafetySignals(prog)
	d.appendProjectionPartitions(prog)
	d.appendProjectionScores(prog)
	d.appendProjectionMappings(prog)
}

func (d *decompiler) appendProjectionPartitions(prog *Program) {
	for _, partition := range d.cfg.Projections.Partitions {
		prog.ProjectionPartitions = append(prog.ProjectionPartitions, &ProjectionPartitionDecl{
			Name:        partition.Name,
			Semantics:   partition.Semantics,
			Temperature: partition.Temperature,
			Members:     partition.Members,
			Default:     partition.Default,
		})
	}
}

func (d *decompiler) appendProjectionScores(prog *Program) {
	for _, score := range d.cfg.Projections.Scores {
		decl := &ProjectionScoreDecl{
			Name:   score.Name,
			Method: score.Method,
		}
		for _, input := range score.Inputs {
			decl.Inputs = append(decl.Inputs, &ProjectionScoreInputDecl{
				SignalType:  input.Type,
				SignalName:  input.Name,
				Weight:      input.Weight,
				ValueSource: input.ValueSource,
				Match:       input.Match,
				Miss:        input.Miss,
			})
		}
		prog.ProjectionScores = append(prog.ProjectionScores, decl)
	}
}

func (d *decompiler) appendProjectionMappings(prog *Program) {
	for _, mapping := range d.cfg.Projections.Mappings {
		decl := &ProjectionMappingDecl{
			Name:   mapping.Name,
			Source: mapping.Source,
			Method: mapping.Method,
		}
		if mapping.Calibration != nil {
			decl.Calibration = &ProjectionMappingCalibrationDecl{
				Method: mapping.Calibration.Method,
				Slope:  mapping.Calibration.Slope,
			}
		}
		for _, output := range mapping.Outputs {
			decl.Outputs = append(decl.Outputs, &ProjectionMappingOutputDecl{
				Name: output.Name,
				LT:   output.LT,
				LTE:  output.LTE,
				GT:   output.GT,
				GTE:  output.GTE,
			})
		}
		prog.ProjectionMappings = append(prog.ProjectionMappings, decl)
	}
}

func (d *decompiler) appendCoreSignals(prog *Program) {
	d.appendApplicationSignals(prog, config.SignalTypeDomain)
	for _, kw := range d.cfg.KeywordRules {
		prog.Signals = append(prog.Signals, d.keywordToSignal(&kw))
	}
	d.appendApplicationSignals(prog, config.SignalTypeFactCheck)
	d.appendApplicationSignals(prog, config.SignalTypeUserFeedback)
	d.appendApplicationSignals(prog, config.SignalTypeTurnRelation)
}

func (d *decompiler) appendOperationalSignals(prog *Program) {
	for _, lang := range d.cfg.LanguageRules {
		prog.Signals = append(prog.Signals, d.languageToSignal(&lang))
	}
	for _, ctx := range d.cfg.ContextRules {
		prog.Signals = append(prog.Signals, d.contextToSignal(&ctx))
	}
	d.appendApplicationSignals(prog, config.SignalTypeComplexity)
	d.appendApplicationSignals(prog, config.SignalTypeModality)
	for _, rb := range d.cfg.RoleBindings {
		prog.Signals = append(prog.Signals, d.roleBindingToSignal(&rb))
	}
	for i := range d.cfg.ConversationRules {
		prog.Signals = append(prog.Signals, d.conversationToSignal(&d.cfg.ConversationRules[i]))
	}
}

func (d *decompiler) appendSafetySignals(prog *Program) {
	for _, signalType := range []string{
		config.SignalTypeSafety, config.SignalTypeJailbreak, config.SignalTypeHallucination,
		config.SignalTypeRefusal, config.SignalTypePII,
	} {
		d.appendApplicationSignals(prog, signalType)
	}
}

func (d *decompiler) appendModelsToProgram(prog *Program) {
	for _, model := range config.ModelCardsFromRouterConfig(d.cfg) {
		prog.Models = append(prog.Models, routingModelToDecl(model))
	}
}

func (d *decompiler) appendRoutesToProgram(prog *Program) {
	for _, dec := range d.cfg.Decisions {
		prog.Routes = append(prog.Routes, d.decisionToRoute(&dec))
	}
}

func (d *decompiler) decompileRoutingModels(models []config.RoutingModel) {
	for _, model := range models {
		d.write("MODEL %s {\n", quoteName(model.Name))
		d.writeRoutingModelFields(model)
		d.write("}\n\n")
	}
}

func (d *decompiler) writeRoutingModelFields(model config.RoutingModel) {
	d.writeOptionalRoutingModelString("param_size", model.ParamSize)
	if model.ContextWindowSize > 0 {
		d.write("  context_window_size: %d\n", model.ContextWindowSize)
	}
	if model.MaxOutputTokens > 0 {
		d.write("  max_output_tokens: %d\n", model.MaxOutputTokens)
	}
	d.writeOptionalRoutingModelString("description", model.Description)
	d.writeOptionalRoutingModelArray("capabilities", model.Capabilities)
	d.writeRoutingModelLoRAs(model.LoRAs)
	d.writeOptionalRoutingModelArray("tags", model.Tags)
	if modalities := model.Modalities; modalities != nil {
		d.write("  modalities: { input: %s, output: %s }\n", quotedStringArray(modalities.Input), quotedStringArray(modalities.Output))
	}
}

func (d *decompiler) writeOptionalRoutingModelString(key, value string) {
	if value == "" {
		return
	}
	d.write("  %s: %q\n", key, value)
}

func (d *decompiler) writeOptionalRoutingModelArray(key string, values []string) {
	if len(values) == 0 {
		return
	}
	d.write("  %s: %s\n", key, quotedStringArray(values))
}

func (d *decompiler) writeRoutingModelLoRAs(adapters []config.LoRAAdapter) {
	if len(adapters) == 0 {
		return
	}
	d.write("  loras: [\n")
	for _, adapter := range adapters {
		d.write("    { name: %q", adapter.Name)
		if adapter.Description != "" {
			d.write(", description: %q", adapter.Description)
		}
		d.write(" },\n")
	}
	d.write("  ]\n")
}

func routingModelToDecl(model config.RoutingModel) *ModelDecl {
	fields := make(map[string]Value)
	if model.ParamSize != "" {
		fields["param_size"] = StringValue{V: model.ParamSize}
	}
	if model.ContextWindowSize > 0 {
		fields["context_window_size"] = IntValue{V: model.ContextWindowSize}
	}
	if model.MaxOutputTokens > 0 {
		fields["max_output_tokens"] = IntValue{V: model.MaxOutputTokens}
	}
	if model.Description != "" {
		fields["description"] = StringValue{V: model.Description}
	}
	if len(model.Capabilities) > 0 {
		fields["capabilities"] = stringsToArray(model.Capabilities)
	}
	if len(model.LoRAs) > 0 {
		items := make([]Value, 0, len(model.LoRAs))
		for _, adapter := range model.LoRAs {
			loraFields := map[string]Value{
				"name": StringValue{V: adapter.Name},
			}
			if adapter.Description != "" {
				loraFields["description"] = StringValue{V: adapter.Description}
			}
			items = append(items, ObjectValue{Fields: loraFields})
		}
		fields["loras"] = ArrayValue{Items: items}
	}
	if len(model.Tags) > 0 {
		fields["tags"] = stringsToArray(model.Tags)
	}
	if modalities := model.Modalities; modalities != nil {
		fields["modalities"] = ObjectValue{Fields: map[string]Value{
			"input":  stringsToArray(modalities.Input),
			"output": stringsToArray(modalities.Output),
		}}
	}
	return &ModelDecl{Name: model.Name, Fields: fields}
}

func quotedStringArray(values []string) string {
	if len(values) == 0 {
		return "[]"
	}
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, strconv.Quote(value))
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}
