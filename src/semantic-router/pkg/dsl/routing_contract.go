package dsl

import (
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
	doc := routingYAMLDocument{
		Routing: config.CanonicalRoutingFromRouterConfig(cfg),
	}
	return yaml.Marshal(doc)
}

// DecompileRouting converts runtime config to the routing-only DSL contract.
func DecompileRouting(cfg *config.RouterConfig) (string, error) {
	d := &decompiler{cfg: cfg}
	d.pluginTemplates = make(map[string]*pluginTemplate)
	d.extractPluginTemplates()

	d.writeSection("SIGNALS")
	d.decompileSignals()

	models := config.CanonicalRoutingFromRouterConfig(cfg).ModelCards
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
	prog := &Program{}
	d.appendSignalsToProgram(prog)
	d.appendModelsToProgram(prog)
	d.appendRoutesToProgram(prog)
	return prog
}

func (d *decompiler) appendSignalsToProgram(prog *Program) {
	d.appendCoreSignals(prog)
	d.appendOperationalSignals(prog)
	d.appendSafetySignals(prog)
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
	for _, comp := range d.cfg.ComplexityRules {
		prog.Signals = append(prog.Signals, d.complexityToSignal(&comp))
	}
	for _, mod := range d.cfg.ModalityRules {
		prog.Signals = append(prog.Signals, d.modalityToSignal(&mod))
	}
	for _, rb := range d.cfg.RoleBindings {
		prog.Signals = append(prog.Signals, d.roleBindingToSignal(&rb))
	}
}

func (d *decompiler) appendSafetySignals(prog *Program) {
	for _, jb := range d.cfg.JailbreakRules {
		prog.Signals = append(prog.Signals, d.jailbreakToSignal(&jb))
	}
	for _, pii := range d.cfg.PIIRules {
		prog.Signals = append(prog.Signals, d.piiToSignal(&pii))
	}
}

func (d *decompiler) appendModelsToProgram(prog *Program) {
	for _, model := range config.CanonicalRoutingFromRouterConfig(d.cfg).ModelCards {
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
	d.writeOptionalRoutingModelString("description", model.Description)
	d.writeOptionalRoutingModelArray("capabilities", model.Capabilities)
	d.writeRoutingModelLoRAs(model.LoRAs)
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
