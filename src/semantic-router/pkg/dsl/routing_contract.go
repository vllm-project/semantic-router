package dsl

import (
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

	if cfg != nil && cfg.MetaRouting.Enabled() {
		d.writeSection("META")
		d.decompileMeta()
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
	d.appendMetaToProgram(prog)
	d.appendModelsToProgram(prog)
	d.appendRoutesToProgram(prog)
	return prog
}

func (d *decompiler) appendSignalsToProgram(prog *Program) {
	d.appendCoreSignals(prog)
	d.appendOperationalSignals(prog)
	d.appendSafetySignals(prog)
	d.appendProjectionPartitions(prog)
	d.appendProjectionScores(prog)
	d.appendProjectionMappings(prog)
}

func (d *decompiler) appendMetaToProgram(prog *Program) {
	if d.cfg == nil || !d.cfg.MetaRouting.Enabled() {
		return
	}
	prog.Meta = &MetaDecl{
		Fields: metaRoutingToFields(d.cfg.MetaRouting),
	}
}

func (d *decompiler) decompileMeta() {
	fields := metaRoutingToFields(d.cfg.MetaRouting)
	if len(fields) == 0 {
		return
	}
	d.write("META {\n")
	for _, key := range sortedFieldKeys(fields) {
		d.write("  %s: %s\n", key, formatDSLValue(fields[key]))
	}
	d.write("}\n\n")
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

func metaRoutingToFields(meta config.MetaRoutingConfig) map[string]Value {
	fields := make(map[string]Value)
	if meta.Mode != "" {
		fields["mode"] = StringValue{V: meta.Mode}
	}
	if meta.MaxPasses > 0 {
		fields["max_passes"] = IntValue{V: meta.MaxPasses}
	}
	if triggerFields := metaTriggerPolicyToFields(meta.TriggerPolicy); len(triggerFields) > 0 {
		fields["trigger_policy"] = ObjectValue{Fields: triggerFields}
	}
	if items := metaAllowedActionsToValues(meta.AllowedActions); len(items) > 0 {
		fields["allowed_actions"] = ArrayValue{Items: items}
	}
	return fields
}

func metaTriggerPolicyToFields(policy *config.MetaTriggerPolicy) map[string]Value {
	if policy == nil {
		return nil
	}

	fields := make(map[string]Value)
	addMetaTriggerThresholdFields(fields, policy)
	addMetaRequiredFamiliesField(fields, policy.RequiredFamilies)
	addMetaFamilyDisagreementsField(fields, policy.FamilyDisagreements)
	return fields
}

func addMetaTriggerThresholdFields(fields map[string]Value, policy *config.MetaTriggerPolicy) {
	if policy.DecisionMarginBelow != nil {
		fields["decision_margin_below"] = FloatValue{V: *policy.DecisionMarginBelow}
	}
	if policy.ProjectionBoundaryWithin != nil {
		fields["projection_boundary_within"] = FloatValue{V: *policy.ProjectionBoundaryWithin}
	}
	if policy.PartitionConflict != nil {
		fields["partition_conflict"] = BoolValue{V: *policy.PartitionConflict}
	}
}

func addMetaRequiredFamiliesField(fields map[string]Value, families []config.MetaRequiredSignalFamily) {
	if len(families) == 0 {
		return
	}
	items := make([]Value, 0, len(families))
	for _, family := range families {
		items = append(items, ObjectValue{Fields: metaRequiredFamilyToFields(family)})
	}
	fields["required_families"] = ArrayValue{Items: items}
}

func metaRequiredFamilyToFields(family config.MetaRequiredSignalFamily) map[string]Value {
	fields := map[string]Value{
		"type": StringValue{V: family.Type},
	}
	if family.MinConfidence != nil {
		fields["min_confidence"] = FloatValue{V: *family.MinConfidence}
	}
	if family.MinMatches != nil {
		fields["min_matches"] = IntValue{V: *family.MinMatches}
	}
	return fields
}

func addMetaFamilyDisagreementsField(fields map[string]Value, disagreements []config.MetaSignalFamilyDisagreement) {
	if len(disagreements) == 0 {
		return
	}
	items := make([]Value, 0, len(disagreements))
	for _, disagreement := range disagreements {
		items = append(items, ObjectValue{Fields: map[string]Value{
			"cheap":     StringValue{V: disagreement.Cheap},
			"expensive": StringValue{V: disagreement.Expensive},
		}})
	}
	fields["family_disagreements"] = ArrayValue{Items: items}
}

func metaAllowedActionsToValues(actions []config.MetaRefinementAction) []Value {
	if len(actions) == 0 {
		return nil
	}
	items := make([]Value, 0, len(actions))
	for _, action := range actions {
		items = append(items, ObjectValue{Fields: metaAllowedActionToFields(action)})
	}
	return items
}

func metaAllowedActionToFields(action config.MetaRefinementAction) map[string]Value {
	fields := map[string]Value{
		"type": StringValue{V: action.Type},
	}
	if len(action.SignalFamilies) > 0 {
		fields["signal_families"] = stringsToArray(action.SignalFamilies)
	}
	return fields
}

func sortedFieldKeys(fields map[string]Value) []string {
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func formatDSLValue(value Value) string {
	switch v := value.(type) {
	case StringValue:
		return strconv.Quote(v.V)
	case IntValue:
		return strconv.Itoa(v.V)
	case FloatValue:
		return strconv.FormatFloat(v.V, 'f', -1, 64)
	case BoolValue:
		if v.V {
			return "true"
		}
		return "false"
	case ArrayValue:
		parts := make([]string, 0, len(v.Items))
		for _, item := range v.Items {
			parts = append(parts, formatDSLValue(item))
		}
		return "[" + strings.Join(parts, ", ") + "]"
	case ObjectValue:
		parts := make([]string, 0, len(v.Fields))
		for _, key := range sortedFieldKeys(v.Fields) {
			parts = append(parts, key+": "+formatDSLValue(v.Fields[key]))
		}
		return "{ " + strings.Join(parts, ", ") + " }"
	default:
		return "null"
	}
}
