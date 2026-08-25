package dsl

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type routingYAMLDocument struct {
	Routing config.CanonicalRouting `yaml:"routing"`
}

// EmitRoutingYAML compiles DSL source and emits the v0.3 routing fragment.
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
	document, err := canonicalRecipeDocument(cfg)
	if err != nil {
		return nil, err
	}
	doc := routingYAMLDocument{Routing: document}
	return yaml.Marshal(doc)
}

// canonicalRecipeDocument selects exactly one DSL-owned Recipe document and
// runs it through the same strict, model-free contract accepted by the
// Management API. Keeping this projection in one place prevents the YAML and
// base-merge paths from silently diverging.
func canonicalRecipeDocument(cfg *config.RouterConfig) (config.CanonicalRouting, error) {
	document, err := selectRoutingDocument(cfg)
	if err != nil {
		return config.CanonicalRouting{}, err
	}
	raw, err := config.MarshalRoutingRecipeDocument(document)
	if err != nil {
		return config.CanonicalRouting{}, err
	}
	parsed, _, err := config.ParseRoutingRecipeDocument(raw)
	if err != nil {
		return config.CanonicalRouting{}, err
	}
	// Decision identity is publication state. Recipe DSL addresses Decisions by
	// their human name, so no generated snapshot ID belongs in authoring output.
	for index := range parsed.Decisions {
		parsed.Decisions[index].ID = ""
	}
	return config.CanonicalRoutingFromRecipeDocument(parsed), nil
}

// selectRoutingDocument keeps the routing-only helper deliberately narrow:
// one invocation emits exactly one model-free Recipe document. A narrow
// Recipe program is represented by the flat routing view; a complete manifest
// must contain exactly one Recipe to use this emitter.
func selectRoutingDocument(cfg *config.RouterConfig) (config.CanonicalRouting, error) {
	if cfg == nil {
		return config.CanonicalRouting{}, fmt.Errorf("cannot emit a nil routing config")
	}
	if len(cfg.Recipes) == 0 {
		return config.CanonicalRoutingFromRouterConfig(cfg), nil
	}
	if len(cfg.Recipes) != 1 {
		return config.CanonicalRouting{}, fmt.Errorf("routing-only export requires exactly one Recipe")
	}
	selected := &cfg.Recipes[0]
	scoped := cfg.ConfigForRecipe(selected)
	if scoped == nil {
		return config.CanonicalRouting{}, fmt.Errorf("cannot construct routing view for Recipe %q", selected.Name)
	}
	return config.CanonicalRoutingFromRouterConfig(scoped), nil
}

// DecompileRouting converts runtime config to the routing-only DSL contract.
func DecompileRouting(cfg *config.RouterConfig) (string, error) {
	d := &decompiler{cfg: cfg}
	d.pluginTemplates = make(map[string]*pluginTemplate)
	d.extractPluginTemplates()
	d.decompileRoutingStrategy()

	d.writeSection("SIGNALS")
	d.decompileSignals()

	models := canonicalDSLModels(cfg)
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
	d := &decompiler{cfg: cfg}
	prog := &Program{Strategy: string(cfg.Strategy)}
	d.appendSignalsToProgram(prog)
	d.appendModelsToProgram(prog)
	d.appendRoutesToProgram(prog)
	return prog
}

func (d *decompiler) decompileRoutingStrategy() {
	if d.cfg.Strategy == "" {
		return
	}
	d.writeSection("ROUTING PROFILE")
	d.write("ROUTING {\n  strategy: %s\n}\n\n", d.cfg.Strategy)
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
				KB:          input.KB,
				Metric:      input.Metric,
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
	for _, cat := range d.cfg.Categories {
		prog.Signals = append(prog.Signals, d.categoryToSignal(&cat))
	}
	for _, kw := range d.cfg.KeywordRules {
		prog.Signals = append(prog.Signals, d.keywordToSignal(&kw))
	}
	for _, emb := range d.cfg.EmbeddingRules {
		prog.Signals = append(prog.Signals, d.embeddingToSignal(&emb))
	}
	for _, fc := range d.cfg.FactCheckRules {
		prog.Signals = append(prog.Signals, d.factCheckToSignal(&fc))
	}
	for _, uf := range d.cfg.UserFeedbackRules {
		prog.Signals = append(prog.Signals, d.userFeedbackToSignal(&uf))
	}
	for _, rule := range d.cfg.ReaskRules {
		prog.Signals = append(prog.Signals, d.reaskToSignal(&rule))
	}
	for _, pref := range d.cfg.PreferenceRules {
		prog.Signals = append(prog.Signals, d.preferenceToSignal(&pref))
	}
}

func (d *decompiler) appendOperationalSignals(prog *Program) {
	for _, lang := range d.cfg.LanguageRules {
		prog.Signals = append(prog.Signals, d.languageToSignal(&lang))
	}
	for _, ctx := range d.cfg.ContextRules {
		prog.Signals = append(prog.Signals, d.contextToSignal(&ctx))
	}
	for _, structure := range d.cfg.StructureRules {
		prog.Signals = append(prog.Signals, d.structureToSignal(&structure))
	}
	for _, comp := range d.cfg.ComplexityRules {
		prog.Signals = append(prog.Signals, d.complexityToSignal(&comp))
	}
	for _, mod := range d.cfg.ModalityRules {
		prog.Signals = append(prog.Signals, d.modalityToSignal(&mod))
	}
	for _, rb := range d.cfg.RoleBindings {
		prog.Signals = append(prog.Signals, d.roleBindingToSignal(&rb))
	}
	for i := range d.cfg.ConversationRules {
		prog.Signals = append(prog.Signals, d.conversationToSignal(&d.cfg.ConversationRules[i]))
	}
	for i := range d.cfg.EventRules {
		prog.Signals = append(prog.Signals, d.eventRuleToDecl(&d.cfg.EventRules[i]))
	}
}

func (d *decompiler) appendSafetySignals(prog *Program) {
	for _, jb := range d.cfg.JailbreakRules {
		prog.Signals = append(prog.Signals, d.jailbreakToSignal(&jb))
	}
	for _, pii := range d.cfg.PIIRules {
		prog.Signals = append(prog.Signals, d.piiToSignal(&pii))
	}
	for _, kb := range d.cfg.KBRules {
		prog.Signals = append(prog.Signals, d.kbSignalToDecl(&kb))
	}
}

func (d *decompiler) appendModelsToProgram(prog *Program) {
	for _, model := range canonicalDSLModels(d.cfg) {
		prog.Models = append(prog.Models, routingModelToDecl(model))
	}
}

type dslModelCard struct {
	config.RoutingModel
	Aliases []string
}

func canonicalDSLModels(cfg *config.RouterConfig) []dslModelCard {
	models := append([]config.RoutingModel(nil), config.CanonicalConfigFromRouterConfig(cfg).Routing.ModelCards...)
	if len(models) == 0 {
		models = config.CanonicalRoutingFromRouterConfig(cfg).ModelCards
	}
	if len(models) == 0 && cfg != nil {
		models = make([]config.RoutingModel, 0, len(cfg.ModelConfig))
		for name, params := range cfg.ModelConfig {
			loras := make([]config.LoRAAdapter, 0, len(params.LoRAs))
			for _, adapter := range params.LoRAs {
				loras = append(loras, config.LoRAAdapter{Name: adapter.Name, Description: adapter.Description})
			}
			models = append(models, config.RoutingModel{
				Name: name, ParamSize: params.ParamSize,
				ContextWindowSize: params.ContextWindowSize, Description: params.Description,
				Capabilities: append([]string(nil), params.Capabilities...), LoRAs: loras,
				Reasoning: config.ModelReasoning{
					Type: params.Reasoning.Type, Efforts: append([]string(nil), params.Reasoning.Efforts...),
				},
				QualityScore: params.QualityScore, Modality: params.Modality,
				Tags: append([]string(nil), params.Tags...),
			})
		}
	}
	views := make([]dslModelCard, 0, len(models))
	for _, model := range models {
		view := dslModelCard{RoutingModel: model}
		if cfg != nil {
			params := cfg.ModelConfig[model.Name]
			view.Aliases = append([]string(nil), params.Aliases...)
			if view.Reasoning.Type == "" {
				view.Reasoning = config.ModelReasoning{
					Type: params.Reasoning.Type, Efforts: append([]string(nil), params.Reasoning.Efforts...),
				}
			}
		}
		views = append(views, view)
	}
	sort.SliceStable(views, func(i, j int) bool {
		return views[i].Name < views[j].Name
	})
	return views
}

func (d *decompiler) appendRoutesToProgram(prog *Program) {
	for _, dec := range d.cfg.Decisions {
		prog.Routes = append(prog.Routes, d.decisionToRoute(&dec))
	}
}

func (d *decompiler) decompileRoutingModels(models []dslModelCard) {
	for _, model := range models {
		d.write("MODEL %s {\n", quoteName(model.Name))
		d.writeRoutingModelFields(model)
		d.write("}\n\n")
	}
}

func (d *decompiler) writeRoutingModelFields(model dslModelCard) {
	d.writeOptionalRoutingModelArray("aliases", model.Aliases)
	d.writeOptionalRoutingModelString("param_size", model.ParamSize)
	if model.ContextWindowSize > 0 {
		d.write("  context_window_size: %d\n", model.ContextWindowSize)
	}
	d.writeOptionalRoutingModelString("description", model.Description)
	d.writeOptionalRoutingModelArray("capabilities", model.Capabilities)
	if model.Reasoning.Type != "" {
		d.write("  reasoning: { type: %q", model.Reasoning.Type)
		if len(model.Reasoning.Efforts) > 0 {
			d.write(", efforts: %s", quotedStringArray(model.Reasoning.Efforts))
		}
		d.write(" }\n")
	}
	if len(model.LoRAs) > 0 {
		loras := make([]string, 0, len(model.LoRAs))
		for _, adapter := range model.LoRAs {
			loras = append(loras, adapter.Name)
		}
		d.writeOptionalRoutingModelArray("loras", loras)
	}
	d.writeOptionalRoutingModelArray("tags", model.Tags)
	if model.QualityScore != 0 {
		d.write(
			"  quality_score: %s\n",
			strconv.FormatFloat(model.QualityScore, 'f', -1, 64),
		)
	}
	d.writeOptionalRoutingModelString("modality", model.Modality)
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

func routingModelToDecl(model dslModelCard) *ModelDecl {
	fields := make(map[string]Value)
	if len(model.Aliases) > 0 {
		fields["aliases"] = stringsToArray(model.Aliases)
	}
	if model.ParamSize != "" {
		fields["param_size"] = StringValue{V: model.ParamSize}
	}
	if model.ContextWindowSize > 0 {
		fields["context_window_size"] = IntValue{V: model.ContextWindowSize}
	}
	if model.Description != "" {
		fields["description"] = StringValue{V: model.Description}
	}
	if len(model.Capabilities) > 0 {
		fields["capabilities"] = stringsToArray(model.Capabilities)
	}
	if model.Reasoning.Type != "" {
		reasoningFields := map[string]Value{"type": StringValue{V: model.Reasoning.Type}}
		if len(model.Reasoning.Efforts) > 0 {
			reasoningFields["efforts"] = stringsToArray(model.Reasoning.Efforts)
		}
		fields["reasoning"] = ObjectValue{Fields: reasoningFields}
	}
	if len(model.LoRAs) > 0 {
		loras := make([]string, 0, len(model.LoRAs))
		for _, adapter := range model.LoRAs {
			loras = append(loras, adapter.Name)
		}
		fields["loras"] = stringsToArray(loras)
	}
	if len(model.Tags) > 0 {
		fields["tags"] = stringsToArray(model.Tags)
	}
	if model.QualityScore != 0 {
		fields["quality_score"] = FloatValue{V: model.QualityScore}
	}
	if model.Modality != "" {
		fields["modality"] = StringValue{V: model.Modality}
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
