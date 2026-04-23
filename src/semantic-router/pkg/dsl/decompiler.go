package dsl

import (
	"fmt"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decompile converts runtime config into the DSL contract.
func Decompile(cfg *config.RouterConfig) (string, error) {
	return DecompileRouting(cfg)
}

// DecompileToAST converts runtime config into a DSL AST Program.
func DecompileToAST(cfg *config.RouterConfig) *Program {
	return DecompileRoutingToAST(cfg)
}

type decompiler struct {
	cfg             *config.RouterConfig
	sb              strings.Builder
	pluginTemplates map[string]*pluginTemplate // auto-extracted templates
}

type pluginTemplate struct {
	name       string
	pluginType string
	usageCount int
}

// ---------- Session State Decompilation ----------

func (d *decompiler) decompileSessionStates() {
	for _, ss := range d.cfg.SessionStates {
		d.write("SESSION_STATE %s {\n", quoteName(ss.Name))
		for _, f := range ss.Fields {
			d.write("  %s: %s\n", f.Name, f.TypeName)
		}
		d.write("}\n\n")
	}
}

// ---------- Signal Decompilation ----------

func (d *decompiler) decompileSignals() {
	for _, cat := range d.cfg.Categories {
		d.write("SIGNAL domain %s {\n", quoteName(cat.Name))
		if cat.Description != "" {
			d.write("  description: %q\n", cat.Description)
		}
		if len(cat.MMLUCategories) > 0 {
			d.write("  mmlu_categories: %s\n", formatStringArray(cat.MMLUCategories))
		}
		d.write("}\n\n")
	}

	for _, kw := range d.cfg.KeywordRules {
		d.write("SIGNAL keyword %s {\n", quoteName(kw.Name))
		if kw.Operator != "" {
			d.write("  operator: %q\n", kw.Operator)
		}
		if len(kw.Keywords) > 0 {
			d.write("  keywords: %s\n", formatStringArray(kw.Keywords))
		}
		if kw.CaseSensitive {
			d.write("  case_sensitive: true\n")
		}
		if kw.Method != "" {
			d.write("  method: %q\n", kw.Method)
		}
		if kw.FuzzyMatch {
			d.write("  fuzzy_match: true\n")
		}
		if kw.FuzzyThreshold != 0 {
			d.write("  fuzzy_threshold: %d\n", kw.FuzzyThreshold)
		}
		if kw.BM25Threshold != 0 {
			d.write("  bm25_threshold: %v\n", kw.BM25Threshold)
		}
		if kw.NgramThreshold != 0 {
			d.write("  ngram_threshold: %v\n", kw.NgramThreshold)
		}
		if kw.NgramArity != 0 {
			d.write("  ngram_arity: %d\n", kw.NgramArity)
		}
		d.write("}\n\n")
	}

	for _, emb := range d.cfg.EmbeddingRules {
		d.write("SIGNAL embedding %s {\n", quoteName(emb.Name))
		if emb.SimilarityThreshold != 0 {
			d.write("  threshold: %v\n", emb.SimilarityThreshold)
		}
		if len(emb.Candidates) > 0 {
			d.write("  candidates: %s\n", formatStringArray(emb.Candidates))
		}
		if emb.AggregationMethodConfiged != "" {
			d.write("  aggregation_method: %q\n", string(emb.AggregationMethodConfiged))
		}
		d.write("}\n\n")
	}

	for _, fc := range d.cfg.FactCheckRules {
		d.write("SIGNAL fact_check %s {\n", quoteName(fc.Name))
		if fc.Description != "" {
			d.write("  description: %q\n", fc.Description)
		}
		d.write("}\n\n")
	}

	for _, uf := range d.cfg.UserFeedbackRules {
		d.write("SIGNAL user_feedback %s {\n", quoteName(uf.Name))
		if uf.Description != "" {
			d.write("  description: %q\n", uf.Description)
		}
		d.write("}\n\n")
	}

	for _, rule := range d.cfg.ReaskRules {
		d.write("SIGNAL reask %s {\n", quoteName(rule.Name))
		if rule.Description != "" {
			d.write("  description: %q\n", rule.Description)
		}
		if rule.Threshold != 0 {
			d.write("  threshold: %g\n", rule.Threshold)
		}
		if rule.LookbackTurns != 0 {
			d.write("  lookback_turns: %d\n", rule.LookbackTurns)
		}
		d.write("}\n\n")
	}

	for _, pref := range d.cfg.PreferenceRules {
		d.write("SIGNAL preference %s {\n", quoteName(pref.Name))
		if pref.Description != "" {
			d.write("  description: %q\n", pref.Description)
		}
		if len(pref.Examples) > 0 {
			d.write("  examples: %s\n", formatStringArray(pref.Examples))
		}
		if pref.Threshold != 0 {
			d.write("  threshold: %g\n", pref.Threshold)
		}
		d.write("}\n\n")
	}

	for _, lang := range d.cfg.LanguageRules {
		d.write("SIGNAL language %s {\n", quoteName(lang.Name))
		if lang.Description != "" {
			d.write("  description: %q\n", lang.Description)
		}
		d.write("}\n\n")
	}

	for _, ctx := range d.cfg.ContextRules {
		d.write("SIGNAL context %s {\n", quoteName(ctx.Name))
		if ctx.MinTokens != "" {
			d.write("  min_tokens: %q\n", string(ctx.MinTokens))
		}
		if ctx.MaxTokens != "" {
			d.write("  max_tokens: %q\n", string(ctx.MaxTokens))
		}
		d.write("}\n\n")
	}

	for _, structure := range d.cfg.StructureRules {
		d.write("SIGNAL structure %s {\n", quoteName(structure.Name))
		if structure.Description != "" {
			d.write("  description: %q\n", structure.Description)
		}
		d.write("  feature: %s\n", formatPluginConfigValue(structureFeatureToMap(structure.Feature)))
		if structure.Predicate != nil {
			d.write("  predicate: %s\n", formatPluginConfigValue(structurePredicateToMap(structure.Predicate)))
		}
		d.write("}\n\n")
	}

	for _, conv := range d.cfg.ConversationRules {
		d.write("SIGNAL conversation %s {\n", quoteName(conv.Name))
		if conv.Description != "" {
			d.write("  description: %q\n", conv.Description)
		}
		d.write("  feature: %s\n", formatPluginConfigValue(conversationFeatureToMap(conv.Feature)))
		if conv.Predicate != nil {
			d.write("  predicate: %s\n", formatPluginConfigValue(structurePredicateToMap(conv.Predicate)))
		}
		d.write("}\n\n")
	}

	for _, comp := range d.cfg.ComplexityRules {
		d.write("SIGNAL complexity %s {\n", quoteName(comp.Name))
		if comp.Threshold != 0 {
			d.write("  threshold: %v\n", comp.Threshold)
		}
		if comp.Description != "" {
			d.write("  description: %q\n", comp.Description)
		}
		if comp.Composer != nil {
			d.write("  composer: %s\n", decompileComposerObj(comp.Composer))
		}
		if len(comp.Hard.Candidates) > 0 {
			d.write("  hard: { candidates: %s }\n", formatStringArray(comp.Hard.Candidates))
		}
		if len(comp.Easy.Candidates) > 0 {
			d.write("  easy: { candidates: %s }\n", formatStringArray(comp.Easy.Candidates))
		}
		d.write("}\n\n")
	}

	for _, mod := range d.cfg.ModalityRules {
		d.write("SIGNAL modality %s {\n", quoteName(mod.Name))
		if mod.Description != "" {
			d.write("  description: %q\n", mod.Description)
		}
		d.write("}\n\n")
	}

	for _, rb := range d.cfg.RoleBindings {
		d.write("SIGNAL authz %s {\n", quoteName(rb.Name))
		if rb.Role != "" {
			d.write("  role: %q\n", rb.Role)
		}
		if len(rb.Subjects) > 0 {
			d.write("  subjects: [")
			for i, subj := range rb.Subjects {
				if i > 0 {
					d.write(", ")
				}
				d.write("{ kind: %q, name: %q }", subj.Kind, subj.Name)
			}
			d.write("]\n")
		}
		d.write("}\n\n")
	}

	for _, jb := range d.cfg.JailbreakRules {
		d.write("SIGNAL jailbreak %s {\n", quoteName(jb.Name))
		if jb.Method != "" {
			d.write("  method: %q\n", jb.Method)
		}
		if jb.Threshold != 0 {
			d.write("  threshold: %v\n", jb.Threshold)
		}
		if jb.IncludeHistory {
			d.write("  include_history: true\n")
		}
		if jb.Description != "" {
			d.write("  description: %q\n", jb.Description)
		}
		if len(jb.JailbreakPatterns) > 0 {
			d.write("  jailbreak_patterns: %s\n", formatStringArray(jb.JailbreakPatterns))
		}
		if len(jb.BenignPatterns) > 0 {
			d.write("  benign_patterns: %s\n", formatStringArray(jb.BenignPatterns))
		}
		d.write("}\n\n")
	}

	for _, pii := range d.cfg.PIIRules {
		d.write("SIGNAL pii %s {\n", quoteName(pii.Name))
		if pii.Threshold != 0 {
			d.write("  threshold: %v\n", pii.Threshold)
		}
		if len(pii.PIITypesAllowed) > 0 {
			d.write("  pii_types_allowed: %s\n", formatStringArray(pii.PIITypesAllowed))
		}
		if pii.IncludeHistory {
			d.write("  include_history: true\n")
		}
		if pii.Description != "" {
			d.write("  description: %q\n", pii.Description)
		}
		d.write("}\n\n")
	}

	for _, kb := range d.cfg.KBRules {
		d.write("SIGNAL kb %s {\n", quoteName(kb.Name))
		if kb.KB != "" {
			d.write("  kb: %q\n", kb.KB)
		}
		d.write("  target: { kind: %q, value: %q }\n", kb.Target.Kind, kb.Target.Value)
		if kb.Match != "" {
			d.write("  match: %q\n", kb.Match)
		}
		d.write("}\n\n")
	}

	for _, partition := range d.cfg.Projections.Partitions {
		d.write("PROJECTION partition %s {\n", quoteName(partition.Name))
		if partition.Semantics != "" {
			d.write("  semantics: %q\n", partition.Semantics)
		}
		if partition.Temperature != 0 {
			d.write("  temperature: %v\n", partition.Temperature)
		}
		if len(partition.Members) > 0 {
			d.write("  members: %s\n", formatStringArray(partition.Members))
		}
		if partition.Default != "" {
			d.write("  default: %q\n", partition.Default)
		}
		d.write("}\n\n")
	}

	for _, score := range d.cfg.Projections.Scores {
		d.write("PROJECTION score %s {\n", quoteName(score.Name))
		if score.Method != "" {
			d.write("  method: %q\n", score.Method)
		}
		if len(score.Inputs) > 0 {
			d.write("  inputs: %s\n", formatProjectionScoreInputs(score.Inputs))
		}
		d.write("}\n\n")
	}

	for _, mapping := range d.cfg.Projections.Mappings {
		d.write("PROJECTION mapping %s {\n", quoteName(mapping.Name))
		if mapping.Source != "" {
			d.write("  source: %q\n", mapping.Source)
		}
		if mapping.Method != "" {
			d.write("  method: %q\n", mapping.Method)
		}
		if mapping.Calibration != nil {
			d.write("  calibration: %s\n", formatProjectionMappingCalibration(mapping.Calibration))
		}
		if len(mapping.Outputs) > 0 {
			d.write("  outputs: %s\n", formatProjectionMappingOutputs(mapping.Outputs))
		}
		d.write("}\n\n")
	}
}

// ---------- Plugin Template Extraction ----------

func (d *decompiler) extractPluginTemplates() {
	// Count plugin usage across decisions to find repeated plugins
	type pluginKey struct {
		pluginType string
	}
	seen := make(map[pluginKey]*pluginTemplate)

	for _, dec := range d.cfg.Decisions {
		for _, p := range dec.Plugins {
			key := pluginKey{pluginType: p.Type}
			// Use a simple fingerprint: type
			if _, exists := seen[key]; !exists {
				name := sanitizeName(p.Type)
				seen[key] = &pluginTemplate{
					name:       name,
					pluginType: p.Type,
					usageCount: 1,
				}
			} else {
				seen[key].usageCount++
			}
		}
	}

	// Only extract templates that are used 2+ times
	for _, tmpl := range seen {
		if tmpl.usageCount >= 2 {
			d.pluginTemplates[tmpl.pluginType] = tmpl
		}
	}
}

func (d *decompiler) decompilePluginTemplates() {
	// Sort by plugin type for deterministic output
	keys := sortedKeys(d.pluginTemplates)
	for _, key := range keys {
		tmpl := d.pluginTemplates[key]
		d.write("PLUGIN %s %s {}\n\n", tmpl.name, sanitizeName(tmpl.pluginType))
	}
}

// ---------- Decision/Route Decompilation ----------

func (d *decompiler) decompileDecisions() {
	for _, dec := range d.cfg.Decisions {
		if dec.Description != "" {
			d.write("ROUTE %s (description = %q) {\n", quoteName(dec.Name), dec.Description)
		} else {
			d.write("ROUTE %s {\n", quoteName(dec.Name))
		}

		d.write("  PRIORITY %d\n", dec.Priority)
		if dec.Tier != 0 {
			d.write("  TIER %d\n", dec.Tier)
		}

		// WHEN expression
		ruleExpr := decompileRuleNode(&dec.Rules)
		if ruleExpr != "" {
			d.write("  WHEN %s\n", ruleExpr)
		}

		omitModelList := candidateIterationsCoverModelRefs(dec)

		// MODEL list
		if len(dec.ModelRefs) > 0 && !omitModelList {
			d.write("  MODEL ")
			for i, mr := range dec.ModelRefs {
				if i > 0 {
					d.write(",\n        ")
				}
				d.write("%q", mr.Model)
				opts := modelRefOptions(&mr, d.cfg.ModelConfig)
				if opts != "" {
					d.write(" (%s)", opts)
				}
			}
			d.write("\n")
		}

		// Bounded candidate iteration
		for _, iter := range dec.CandidateIterations {
			d.decompileCandidateIteration(iter)
		}

		// ALGORITHM
		if dec.Algorithm != nil && dec.Algorithm.Type != "" {
			d.write("  ALGORITHM %s", dec.Algorithm.Type)
			algoFields := d.decompileAlgorithmFields(dec.Algorithm)
			if algoFields != "" {
				d.write(" {\n%s  }\n", algoFields)
			} else {
				d.write("\n")
			}
		}

		// PLUGINs
		for _, p := range dec.Plugins {
			pluginFields := decompilePluginConfig(&p)
			if pluginFields != "" {
				d.write("  PLUGIN %s {\n%s  }\n", sanitizeName(p.Type), pluginFields)
			} else {
				d.write("  PLUGIN %s\n", sanitizeName(p.Type))
			}
		}

		d.write("}\n\n")
	}
}

func (d *decompiler) decompileCandidateIteration(iter config.CandidateIterationConfig) {
	source := iter.Source
	if source == "models" {
		source = decompileCandidateIterationModelSource(iter.Models)
	}
	d.write("  FOR %s IN %s {\n", sanitizeName(iter.Variable), source)
	for _, output := range iter.Outputs {
		if output.Type == "model" {
			d.write("    MODEL %s\n", sanitizeName(output.Value))
		}
	}
	d.write("  }\n")
}

func decompileCandidateIterationModelSource(models []config.ModelRef) string {
	if len(models) == 0 {
		return "[]"
	}
	var sb strings.Builder
	sb.WriteString("[")
	for i, model := range models {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(strconv.Quote(model.Model))
		opts := candidateIterationModelRefOptions(&model)
		if opts != "" {
			sb.WriteString(" (")
			sb.WriteString(opts)
			sb.WriteString(")")
		}
	}
	sb.WriteString("]")
	return sb.String()
}

func candidateIterationModelRefOptions(model *config.ModelRef) string {
	var opts []string
	if model.UseReasoning != nil {
		opts = append(opts, fmt.Sprintf("reasoning = %t", *model.UseReasoning))
	}
	if model.ReasoningEffort != "" {
		opts = append(opts, fmt.Sprintf("effort = %q", model.ReasoningEffort))
	}
	if model.LoRAName != "" {
		opts = append(opts, fmt.Sprintf("lora = %q", model.LoRAName))
	}
	if model.Weight != 0 {
		opts = append(opts, fmt.Sprintf("weight = %s", strconv.FormatFloat(model.Weight, 'f', -1, 64)))
	}
	return strings.Join(opts, ", ")
}

func candidateIterationsCoverModelRefs(dec config.Decision) bool {
	// The MODEL omission optimization is only proven for one explicit-model
	// iteration that emits MODEL <iterator>. Multiple iterations require a
	// merge/order/dedup contract before they can safely cover ModelRefs.
	if len(dec.CandidateIterations) != 1 {
		return false
	}
	iter := dec.CandidateIterations[0]
	if iter.Source != "models" || !iterEmitsVariable(iter) {
		return false
	}
	if len(dec.ModelRefs) != len(iter.Models) {
		return false
	}
	for i := range dec.ModelRefs {
		if dec.ModelRefs[i].Model != iter.Models[i].Model ||
			dec.ModelRefs[i].LoRAName != iter.Models[i].LoRAName {
			return false
		}
	}
	return true
}

func decompileRuleNode(node *config.RuleCombination) string {
	if node == nil {
		return ""
	}

	// Leaf node — signal reference
	if node.Type != "" {
		return fmt.Sprintf("%s(%q)", node.Type, node.Name)
	}

	switch node.Operator {
	case "AND":
		// Flatten nested ANDs into a flat list: a AND b AND c
		parts := flattenRuleNode(node, "AND")
		return strings.Join(parts, " AND ")
	case "OR":
		// Flatten nested ORs into a flat list: (a OR b OR c)
		parts := flattenRuleNode(node, "OR")
		return "(" + strings.Join(parts, " OR ") + ")"
	case "NOT":
		if len(node.Conditions) == 1 {
			inner := decompileRuleNode(&node.Conditions[0])
			return "NOT " + inner
		}
	}

	// Fallback: join with operator
	parts := make([]string, 0, len(node.Conditions))
	for _, c := range node.Conditions {
		parts = append(parts, decompileRuleNode(&c))
	}
	if node.Operator != "" {
		return strings.Join(parts, " "+node.Operator+" ")
	}
	return strings.Join(parts, " AND ")
}

// flattenRuleNode collects all children of same-operator nested nodes into a flat list.
// e.g. OR(OR(a, b), c) → [a, b, c]
func flattenRuleNode(node *config.RuleCombination, op string) []string {
	if node.Operator == op {
		var parts []string
		for i := range node.Conditions {
			parts = append(parts, flattenRuleNode(&node.Conditions[i], op)...)
		}
		return parts
	}
	return []string{decompileRuleNode(node)}
}

// decompilePluginConfig emits field lines for a DecisionPlugin's Configuration.
func decompilePluginConfig(p *config.DecisionPlugin) string {
	var sb strings.Builder
	switch p.Type {
	case "system_prompt":
		cfg, ok := decodePluginConfig[config.SystemPromptPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled != nil {
			if *cfg.Enabled {
				fmt.Fprintf(&sb, "    enabled: true\n")
			} else {
				fmt.Fprintf(&sb, "    enabled: false\n")
			}
		}
		if cfg.SystemPrompt != "" {
			fmt.Fprintf(&sb, "    system_prompt: %q\n", cfg.SystemPrompt)
		}
		if cfg.Mode != "" {
			fmt.Fprintf(&sb, "    mode: %q\n", cfg.Mode)
		}
	case "semantic-cache":
		cfg, ok := decodePluginConfig[config.SemanticCachePluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.SimilarityThreshold != nil {
			fmt.Fprintf(&sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
		}
	case "router_replay":
		cfg, ok := decodePluginConfig[config.RouterReplayPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.MaxRecords != 0 {
			fmt.Fprintf(&sb, "    max_records: %d\n", cfg.MaxRecords)
		}
		if cfg.CaptureRequestBody {
			fmt.Fprintf(&sb, "    capture_request_body: true\n")
		}
		if cfg.CaptureResponseBody {
			fmt.Fprintf(&sb, "    capture_response_body: true\n")
		}
		if cfg.MaxBodyBytes != 0 {
			fmt.Fprintf(&sb, "    max_body_bytes: %d\n", cfg.MaxBodyBytes)
		}
	case "memory":
		cfg, ok := decodePluginConfig[config.MemoryPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.RetrievalLimit != nil {
			fmt.Fprintf(&sb, "    retrieval_limit: %d\n", *cfg.RetrievalLimit)
		}
		if cfg.SimilarityThreshold != nil {
			fmt.Fprintf(&sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
		}
		if cfg.AutoStore != nil {
			fmt.Fprintf(&sb, "    auto_store: %v\n", *cfg.AutoStore)
		}
	case "hallucination":
		cfg, ok := decodePluginConfig[config.HallucinationPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.UseNLI {
			fmt.Fprintf(&sb, "    use_nli: true\n")
		}
		if cfg.HallucinationAction != "" {
			fmt.Fprintf(&sb, "    hallucination_action: %q\n", cfg.HallucinationAction)
		}
	case "image_gen":
		cfg, ok := decodePluginConfig[config.ImageGenPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.Backend != "" {
			fmt.Fprintf(&sb, "    backend: %q\n", cfg.Backend)
		}
	case "fast_response":
		cfg, ok := decodePluginConfig[config.FastResponsePluginConfig](p)
		if !ok {
			break
		}
		if cfg.Message != "" {
			fmt.Fprintf(&sb, "    message: %q\n", cfg.Message)
		}
	case "request_params":
		cfg, ok := decodePluginConfig[config.RequestParamsPluginConfig](p)
		if !ok {
			break
		}
		if len(cfg.BlockedParams) > 0 {
			fmt.Fprintf(&sb, "    blocked_params: %s\n", formatStringArray(cfg.BlockedParams))
		}
		if cfg.MaxTokensLimit != nil {
			fmt.Fprintf(&sb, "    max_tokens_limit: %d\n", *cfg.MaxTokensLimit)
		}
		if cfg.MaxN != nil {
			fmt.Fprintf(&sb, "    max_n: %d\n", *cfg.MaxN)
		}
		if cfg.StripUnknown {
			fmt.Fprintf(&sb, "    strip_unknown: true\n")
		}
	case "tools":
		cfg, ok := decodePluginConfig[config.ToolsPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.Mode != "" {
			fmt.Fprintf(&sb, "    mode: %q\n", cfg.Mode)
		}
		if cfg.SemanticSelection != nil {
			fmt.Fprintf(&sb, "    semantic_selection: %v\n", *cfg.SemanticSelection)
		}
		if len(cfg.AllowTools) > 0 {
			fmt.Fprintf(&sb, "    allow_tools: %s\n", formatStringArray(cfg.AllowTools))
		}
		if len(cfg.BlockTools) > 0 {
			fmt.Fprintf(&sb, "    block_tools: %s\n", formatStringArray(cfg.BlockTools))
		}
	case "rag":
		cfg, ok := decodePluginConfig[config.RAGPluginConfig](p)
		if !ok {
			break
		}
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.Backend != "" {
			fmt.Fprintf(&sb, "    backend: %q\n", cfg.Backend)
		}
		if cfg.SimilarityThreshold != nil {
			fmt.Fprintf(&sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
		}
		if cfg.TopK != nil {
			fmt.Fprintf(&sb, "    top_k: %d\n", *cfg.TopK)
		}
		if cfg.MaxContextLength != nil {
			fmt.Fprintf(&sb, "    max_context_length: %d\n", *cfg.MaxContextLength)
		}
		if cfg.InjectionMode != "" {
			fmt.Fprintf(&sb, "    injection_mode: %q\n", cfg.InjectionMode)
		}
	case "header_mutation":
		cfg, ok := decodePluginConfig[config.HeaderMutationPluginConfig](p)
		if !ok {
			break
		}
		if len(cfg.Add) > 0 {
			fmt.Fprintf(&sb, "    add: [")
			for i, h := range cfg.Add {
				if i > 0 {
					fmt.Fprintf(&sb, ", ")
				}
				fmt.Fprintf(&sb, "{ name: %q, value: %q }", h.Name, h.Value)
			}
			fmt.Fprintf(&sb, "]\n")
		}
		if len(cfg.Update) > 0 {
			fmt.Fprintf(&sb, "    update: [")
			for i, h := range cfg.Update {
				if i > 0 {
					fmt.Fprintf(&sb, ", ")
				}
				fmt.Fprintf(&sb, "{ name: %q, value: %q }", h.Name, h.Value)
			}
			fmt.Fprintf(&sb, "]\n")
		}
		if len(cfg.Delete) > 0 {
			fmt.Fprintf(&sb, "    delete: %s\n", formatStringArray(cfg.Delete))
		}
	}
	// Preserve raw-only plugin keys, but do not duplicate typed fields that were
	// already emitted above for known plugin contracts.
	if rawMap, ok := normalizePluginConfigMap(p.Configuration); ok {
		extraFields := filterPluginConfigMap(rawMap, knownPluginConfigKeys(p))
		if len(extraFields) > 0 {
			writePluginConfigMap(&sb, extraFields, "    ")
		}
	}
	return sb.String()
}

func knownPluginConfigKeys(p *config.DecisionPlugin) map[string]struct{} {
	fields := pluginConfigToFields(p)
	if len(fields) == 0 {
		return nil
	}
	keys := make(map[string]struct{}, len(fields))
	for key := range fields {
		keys[key] = struct{}{}
	}
	return keys
}

func filterPluginConfigMap(raw map[string]interface{}, omit map[string]struct{}) map[string]interface{} {
	if len(raw) == 0 {
		return nil
	}
	filtered := make(map[string]interface{}, len(raw))
	for key, value := range raw {
		if _, skip := omit[key]; skip {
			continue
		}
		filtered[key] = value
	}
	return filtered
}

func decodePluginConfig[T any](p *config.DecisionPlugin) (*T, bool) {
	if p == nil || p.Configuration == nil {
		return nil, false
	}
	var result T
	if err := p.Configuration.DecodeInto(&result); err != nil {
		return nil, false
	}
	return &result, true
}

func writePluginConfigMap(sb *strings.Builder, raw map[string]interface{}, indent string) {
	keys := make([]string, 0, len(raw))
	for key := range raw {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		fmt.Fprintf(sb, "%s%s: %s\n", indent, key, formatPluginConfigValue(raw[key]))
	}
}

func normalizePluginConfigMap(raw *config.StructuredPayload) (map[string]interface{}, bool) {
	if raw == nil {
		return nil, false
	}
	typed, err := raw.AsStringMap()
	if err != nil {
		return nil, false
	}
	normalized := make(map[string]interface{}, len(typed))
	for key, value := range typed {
		normalized[key] = normalizePluginConfigValue(value)
	}
	return normalized, true
}

func normalizePluginConfigValue(raw interface{}) interface{} {
	switch typed := raw.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, value := range typed {
			normalized[key] = normalizePluginConfigValue(value)
		}
		return normalized
	case map[interface{}]interface{}:
		normalized := make(map[string]interface{}, len(typed))
		for key, value := range typed {
			normalized[fmt.Sprintf("%v", key)] = normalizePluginConfigValue(value)
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, len(typed))
		for index, value := range typed {
			normalized[index] = normalizePluginConfigValue(value)
		}
		return normalized
	default:
		return normalizePluginConfigReflectValue(raw)
	}
}

func normalizePluginConfigReflectValue(raw interface{}) interface{} {
	if raw == nil {
		return nil
	}
	value := reflect.ValueOf(raw)
	switch value.Kind() {
	case reflect.Array, reflect.Slice:
		normalized := make([]interface{}, value.Len())
		for index := 0; index < value.Len(); index++ {
			normalized[index] = normalizePluginConfigValue(value.Index(index).Interface())
		}
		return normalized
	case reflect.Map:
		normalized := make(map[string]interface{}, value.Len())
		iter := value.MapRange()
		for iter.Next() {
			normalized[fmt.Sprintf("%v", iter.Key().Interface())] = normalizePluginConfigValue(iter.Value().Interface())
		}
		return normalized
	default:
		return raw
	}
}

func formatPluginConfigValue(raw interface{}) string {
	normalized := normalizePluginConfigValue(raw)
	if scalar, ok := formatPluginConfigScalar(normalized); ok {
		return scalar
	}

	switch typed := normalized.(type) {
	case []interface{}:
		return formatPluginConfigList(typed)
	case map[string]interface{}:
		return formatPluginConfigMap(typed)
	case nil:
		return "null"
	default:
		return fmt.Sprintf("%v", normalized)
	}
}

func formatPluginConfigScalar(raw interface{}) (string, bool) {
	switch typed := raw.(type) {
	case string:
		return fmt.Sprintf("%q", typed), true
	case bool:
		return strconv.FormatBool(typed), true
	case int:
		return fmt.Sprintf("%d", typed), true
	case int64:
		return fmt.Sprintf("%d", typed), true
	case float32:
		return fmt.Sprintf("%v", typed), true
	case float64:
		return fmt.Sprintf("%v", typed), true
	default:
		return "", false
	}
}

func formatPluginConfigList(items []interface{}) string {
	parts := make([]string, 0, len(items))
	for _, item := range items {
		parts = append(parts, formatPluginConfigValue(item))
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatPluginConfigMap(values map[string]interface{}) string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	parts := make([]string, 0, len(keys))
	for _, key := range keys {
		parts = append(parts, fmt.Sprintf("%s: %s", key, formatPluginConfigValue(values[key])))
	}
	return "{ " + strings.Join(parts, ", ") + " }"
}

func modelRefOptions(mr *config.ModelRef, modelConfig map[string]config.ModelParams) string {
	var opts []string
	if mr.UseReasoning != nil {
		if *mr.UseReasoning {
			opts = append(opts, "reasoning = true")
		} else {
			opts = append(opts, "reasoning = false")
		}
	}
	if mr.ReasoningEffort != "" {
		opts = append(opts, fmt.Sprintf("effort = %q", mr.ReasoningEffort))
	}
	if mr.LoRAName != "" {
		opts = append(opts, fmt.Sprintf("lora = %q", mr.LoRAName))
	}
	if mr.Weight != 0 {
		opts = append(opts, fmt.Sprintf("weight = %g", mr.Weight))
	}
	// Pull param_size from model_config.
	if mc, ok := modelConfig[mr.Model]; ok {
		if mc.ParamSize != "" {
			opts = append(opts, fmt.Sprintf("param_size = %q", mc.ParamSize))
		}
	}
	return strings.Join(opts, ", ")
}

func (d *decompiler) decompileAlgorithmFields(algo *config.AlgorithmConfig) string {
	var sb strings.Builder
	// Emit top-level on_error only for types that don't have their own sub-config on_error
	switch algo.Type {
	case "confidence", "ratings", "remom":
		// on_error lives inside the sub-config
	default:
		if algo.OnError != "" {
			fmt.Fprintf(&sb, "    on_error: %q\n", algo.OnError)
		}
	}

	switch algo.Type {
	case "confidence":
		if c := algo.Confidence; c != nil {
			if c.ConfidenceMethod != "" {
				fmt.Fprintf(&sb, "    confidence_method: %q\n", c.ConfidenceMethod)
			}
			if c.Threshold != 0 {
				fmt.Fprintf(&sb, "    threshold: %v\n", c.Threshold)
			}
			if c.OnError != "" {
				fmt.Fprintf(&sb, "    on_error: %q\n", c.OnError)
			}
			if c.EscalationOrder != "" {
				fmt.Fprintf(&sb, "    escalation_order: %q\n", c.EscalationOrder)
			}
			if c.CostQualityTradeoff != 0 {
				fmt.Fprintf(&sb, "    cost_quality_tradeoff: %v\n", c.CostQualityTradeoff)
			}
			if c.HybridWeights != nil {
				fmt.Fprintf(&sb, "    hybrid_weights: { logprob_weight: %v, margin_weight: %v }\n",
					c.HybridWeights.LogprobWeight, c.HybridWeights.MarginWeight)
			}
		}
	case "ratings":
		if r := algo.Ratings; r != nil {
			if r.MaxConcurrent != 0 {
				fmt.Fprintf(&sb, "    max_concurrent: %d\n", r.MaxConcurrent)
			}
		}
	case "remom":
		if r := algo.ReMoM; r != nil {
			if len(r.BreadthSchedule) > 0 {
				fmt.Fprintf(&sb, "    breadth_schedule: %s\n", formatIntArray(r.BreadthSchedule))
			}
			if r.ModelDistribution != "" {
				fmt.Fprintf(&sb, "    model_distribution: %q\n", r.ModelDistribution)
			}
			if r.Temperature != 0 {
				fmt.Fprintf(&sb, "    temperature: %v\n", r.Temperature)
			}
			if r.IncludeReasoning {
				fmt.Fprintf(&sb, "    include_reasoning: true\n")
			}
			if r.CompactionStrategy != "" {
				fmt.Fprintf(&sb, "    compaction_strategy: %q\n", r.CompactionStrategy)
			}
			if r.CompactionTokens != 0 {
				fmt.Fprintf(&sb, "    compaction_tokens: %d\n", r.CompactionTokens)
			}
			if r.SynthesisTemplate != "" {
				fmt.Fprintf(&sb, "    synthesis_template: %q\n", r.SynthesisTemplate)
			}
			if r.MaxConcurrent != 0 {
				fmt.Fprintf(&sb, "    max_concurrent: %d\n", r.MaxConcurrent)
			}
			if r.OnError != "" {
				fmt.Fprintf(&sb, "    on_error: %q\n", r.OnError)
			}
		}
	case "elo":
		if e := algo.Elo; e != nil {
			if e.InitialRating != 0 {
				fmt.Fprintf(&sb, "    initial_rating: %v\n", e.InitialRating)
			}
			if e.KFactor != 0 {
				fmt.Fprintf(&sb, "    k_factor: %v\n", e.KFactor)
			}
			if e.CategoryWeighted {
				fmt.Fprintf(&sb, "    category_weighted: true\n")
			}
			if e.DecayFactor != 0 {
				fmt.Fprintf(&sb, "    decay_factor: %v\n", e.DecayFactor)
			}
			if e.StoragePath != "" {
				fmt.Fprintf(&sb, "    storage_path: %q\n", e.StoragePath)
			}
		}
	case "router_dc":
		if r := algo.RouterDC; r != nil {
			if r.Temperature != 0 {
				fmt.Fprintf(&sb, "    temperature: %v\n", r.Temperature)
			}
			if r.DimensionSize != 0 {
				fmt.Fprintf(&sb, "    dimension_size: %d\n", r.DimensionSize)
			}
			if r.MinSimilarity != 0 {
				fmt.Fprintf(&sb, "    min_similarity: %v\n", r.MinSimilarity)
			}
		}
	case "automix":
		if a := algo.AutoMix; a != nil {
			if a.VerificationThreshold != 0 {
				fmt.Fprintf(&sb, "    verification_threshold: %v\n", a.VerificationThreshold)
			}
			if a.MaxEscalations != 0 {
				fmt.Fprintf(&sb, "    max_escalations: %d\n", a.MaxEscalations)
			}
		}
	case "latency_aware":
		if l := algo.LatencyAware; l != nil {
			if l.TPOTPercentile != 0 {
				fmt.Fprintf(&sb, "    tpot_percentile: %d\n", l.TPOTPercentile)
			}
			if l.TTFTPercentile != 0 {
				fmt.Fprintf(&sb, "    ttft_percentile: %d\n", l.TTFTPercentile)
			}
		}
	}

	return sb.String()
}

// ---------- AST Building Helpers (for DecompileToAST) ----------

func (d *decompiler) categoryToSignal(cat *config.Category) *SignalDecl {
	fields := make(map[string]Value)
	if cat.Description != "" {
		fields["description"] = StringValue{V: cat.Description}
	}
	if len(cat.MMLUCategories) > 0 {
		fields["mmlu_categories"] = stringsToArray(cat.MMLUCategories)
	}
	return &SignalDecl{SignalType: "domain", Name: cat.Name, Fields: fields}
}

func (d *decompiler) keywordToSignal(kw *config.KeywordRule) *SignalDecl {
	fields := make(map[string]Value)
	if kw.Operator != "" {
		fields["operator"] = StringValue{V: kw.Operator}
	}
	if len(kw.Keywords) > 0 {
		fields["keywords"] = stringsToArray(kw.Keywords)
	}
	if kw.CaseSensitive {
		fields["case_sensitive"] = BoolValue{V: true}
	}
	if kw.Method != "" {
		fields["method"] = StringValue{V: kw.Method}
	}
	return &SignalDecl{SignalType: "keyword", Name: kw.Name, Fields: fields}
}

func (d *decompiler) embeddingToSignal(emb *config.EmbeddingRule) *SignalDecl {
	fields := make(map[string]Value)
	if emb.SimilarityThreshold != 0 {
		fields["threshold"] = FloatValue{V: float64(emb.SimilarityThreshold)}
	}
	if len(emb.Candidates) > 0 {
		fields["candidates"] = stringsToArray(emb.Candidates)
	}
	if emb.AggregationMethodConfiged != "" {
		fields["aggregation_method"] = StringValue{V: string(emb.AggregationMethodConfiged)}
	}
	return &SignalDecl{SignalType: "embedding", Name: emb.Name, Fields: fields}
}

func (d *decompiler) factCheckToSignal(fc *config.FactCheckRule) *SignalDecl {
	fields := make(map[string]Value)
	if fc.Description != "" {
		fields["description"] = StringValue{V: fc.Description}
	}
	return &SignalDecl{SignalType: "fact_check", Name: fc.Name, Fields: fields}
}

func (d *decompiler) userFeedbackToSignal(uf *config.UserFeedbackRule) *SignalDecl {
	fields := make(map[string]Value)
	if uf.Description != "" {
		fields["description"] = StringValue{V: uf.Description}
	}
	return &SignalDecl{SignalType: "user_feedback", Name: uf.Name, Fields: fields}
}

func (d *decompiler) reaskToSignal(rule *config.ReaskRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	if rule.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(rule.Threshold)}
	}
	if rule.LookbackTurns != 0 {
		fields["lookback_turns"] = IntValue{V: rule.LookbackTurns}
	}
	return &SignalDecl{SignalType: "reask", Name: rule.Name, Fields: fields}
}

func (d *decompiler) preferenceToSignal(pref *config.PreferenceRule) *SignalDecl {
	fields := make(map[string]Value)
	if pref.Description != "" {
		fields["description"] = StringValue{V: pref.Description}
	}
	if len(pref.Examples) > 0 {
		fields["examples"] = stringsToArray(pref.Examples)
	}
	if pref.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(pref.Threshold)}
	}
	return &SignalDecl{SignalType: "preference", Name: pref.Name, Fields: fields}
}

func (d *decompiler) languageToSignal(lang *config.LanguageRule) *SignalDecl {
	fields := make(map[string]Value)
	if lang.Description != "" {
		fields["description"] = StringValue{V: lang.Description}
	}
	return &SignalDecl{SignalType: "language", Name: lang.Name, Fields: fields}
}

func (d *decompiler) contextToSignal(ctx *config.ContextRule) *SignalDecl {
	fields := make(map[string]Value)
	if ctx.MinTokens != "" {
		fields["min_tokens"] = StringValue{V: string(ctx.MinTokens)}
	}
	if ctx.MaxTokens != "" {
		fields["max_tokens"] = StringValue{V: string(ctx.MaxTokens)}
	}
	return &SignalDecl{SignalType: "context", Name: ctx.Name, Fields: fields}
}

func (d *decompiler) structureToSignal(rule *config.StructureRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	fields["feature"] = structureFeatureValue(rule.Feature)
	if rule.Predicate != nil {
		fields["predicate"] = structurePredicateValue(rule.Predicate)
	}
	return &SignalDecl{SignalType: "structure", Name: rule.Name, Fields: fields}
}

func (d *decompiler) conversationToSignal(rule *config.ConversationRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.Description != "" {
		fields["description"] = StringValue{V: rule.Description}
	}
	fields["feature"] = conversationFeatureValue(rule.Feature)
	if rule.Predicate != nil {
		fields["predicate"] = structurePredicateValue(rule.Predicate)
	}
	return &SignalDecl{SignalType: "conversation", Name: rule.Name, Fields: fields}
}

func (d *decompiler) complexityToSignal(comp *config.ComplexityRule) *SignalDecl {
	fields := make(map[string]Value)
	if comp.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(comp.Threshold)}
	}
	if comp.Description != "" {
		fields["description"] = StringValue{V: comp.Description}
	}
	if len(comp.Hard.Candidates) > 0 {
		fields["hard"] = ObjectValue{Fields: map[string]Value{
			"candidates": stringsToArray(comp.Hard.Candidates),
		}}
	}
	if len(comp.Easy.Candidates) > 0 {
		fields["easy"] = ObjectValue{Fields: map[string]Value{
			"candidates": stringsToArray(comp.Easy.Candidates),
		}}
	}
	return &SignalDecl{SignalType: "complexity", Name: comp.Name, Fields: fields}
}

func (d *decompiler) modalityToSignal(mod *config.ModalityRule) *SignalDecl {
	fields := make(map[string]Value)
	if mod.Description != "" {
		fields["description"] = StringValue{V: mod.Description}
	}
	return &SignalDecl{SignalType: "modality", Name: mod.Name, Fields: fields}
}

func (d *decompiler) roleBindingToSignal(rb *config.RoleBinding) *SignalDecl {
	fields := make(map[string]Value)
	if rb.Role != "" {
		fields["role"] = StringValue{V: rb.Role}
	}
	if len(rb.Subjects) > 0 {
		var items []Value
		for _, subj := range rb.Subjects {
			items = append(items, ObjectValue{Fields: map[string]Value{
				"kind": StringValue{V: subj.Kind},
				"name": StringValue{V: subj.Name},
			}})
		}
		fields["subjects"] = ArrayValue{Items: items}
	}
	return &SignalDecl{SignalType: "authz", Name: rb.Name, Fields: fields}
}

func (d *decompiler) jailbreakToSignal(jb *config.JailbreakRule) *SignalDecl {
	fields := make(map[string]Value)
	if jb.Method != "" {
		fields["method"] = StringValue{V: jb.Method}
	}
	if jb.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(jb.Threshold)}
	}
	if jb.IncludeHistory {
		fields["include_history"] = BoolValue{V: true}
	}
	if jb.Description != "" {
		fields["description"] = StringValue{V: jb.Description}
	}
	if len(jb.JailbreakPatterns) > 0 {
		fields["jailbreak_patterns"] = stringsToArray(jb.JailbreakPatterns)
	}
	if len(jb.BenignPatterns) > 0 {
		fields["benign_patterns"] = stringsToArray(jb.BenignPatterns)
	}
	return &SignalDecl{SignalType: "jailbreak", Name: jb.Name, Fields: fields}
}

func (d *decompiler) piiToSignal(pii *config.PIIRule) *SignalDecl {
	fields := make(map[string]Value)
	if pii.Threshold != 0 {
		fields["threshold"] = FloatValue{V: float64(pii.Threshold)}
	}
	if len(pii.PIITypesAllowed) > 0 {
		fields["pii_types_allowed"] = stringsToArray(pii.PIITypesAllowed)
	}
	if pii.IncludeHistory {
		fields["include_history"] = BoolValue{V: true}
	}
	if pii.Description != "" {
		fields["description"] = StringValue{V: pii.Description}
	}
	return &SignalDecl{SignalType: "pii", Name: pii.Name, Fields: fields}
}

func (d *decompiler) kbSignalToDecl(rule *config.KBSignalRule) *SignalDecl {
	fields := make(map[string]Value)
	if rule.KB != "" {
		fields["kb"] = StringValue{V: rule.KB}
	}
	fields["target"] = ObjectValue{Fields: map[string]Value{
		"kind":  StringValue{V: rule.Target.Kind},
		"value": StringValue{V: rule.Target.Value},
	}}
	if rule.Match != "" {
		fields["match"] = StringValue{V: rule.Match}
	}
	return &SignalDecl{SignalType: "kb", Name: rule.Name, Fields: fields}
}

func (d *decompiler) decisionToRoute(dec *config.Decision) *RouteDecl {
	route := &RouteDecl{
		Name:        dec.Name,
		Description: dec.Description,
		Priority:    dec.Priority,
		Tier:        dec.Tier,
	}

	// WHEN
	route.When = decompileRuleNodeToExpr(&dec.Rules)

	// MODEL
	for _, mr := range dec.ModelRefs {
		ref := &ModelRef{
			Model:     mr.Model,
			Reasoning: mr.UseReasoning,
			Effort:    mr.ReasoningEffort,
			LoRA:      mr.LoRAName,
			Weight:    mr.Weight,
		}
		// Pull param_size from model_config.
		if mc, ok := d.cfg.ModelConfig[mr.Model]; ok {
			ref.ParamSize = mc.ParamSize
		}
		route.Models = append(route.Models, ref)
	}

	// ALGORITHM
	if dec.Algorithm != nil && dec.Algorithm.Type != "" {
		algoSpec := &AlgoSpec{AlgoType: dec.Algorithm.Type}
		algoSpec.Fields = d.algorithmToFields(dec.Algorithm)
		route.Algorithm = algoSpec
	}

	// PLUGINs
	for _, p := range dec.Plugins {
		ref := &PluginRef{Name: sanitizeName(p.Type)}
		fields := pluginConfigToFields(&p)
		if len(fields) > 0 {
			ref.Fields = fields
		}
		route.Plugins = append(route.Plugins, ref)
	}

	for _, iter := range dec.CandidateIterations {
		route.CandidateIterations = append(route.CandidateIterations, candidateIterationConfigToDecl(iter))
	}

	return route
}

func candidateIterationConfigToDecl(iter config.CandidateIterationConfig) *CandidateIterationDecl {
	decl := &CandidateIterationDecl{
		Variable: iter.Variable,
		Source:   iter.Source,
	}
	for _, model := range iter.Models {
		decl.Models = append(decl.Models, configModelRefToDSLModelRef(model))
	}
	for _, output := range iter.Outputs {
		decl.Outputs = append(decl.Outputs, &CandidateIterationOutputDecl{
			Type:  output.Type,
			Value: output.Value,
		})
	}
	return decl
}

func configModelRefToDSLModelRef(model config.ModelRef) *ModelRef {
	return &ModelRef{
		Model:     model.Model,
		Reasoning: model.UseReasoning,
		Effort:    model.ReasoningEffort,
		LoRA:      model.LoRAName,
		Weight:    model.Weight,
	}
}

func decompileRuleNodeToExpr(node *config.RuleCombination) BoolExpr {
	if node == nil {
		return nil
	}
	if node.Type != "" {
		return &SignalRefExpr{SignalType: node.Type, SignalName: node.Name}
	}
	switch node.Operator {
	case "AND":
		exprs := flattenRuleNodeToExprs(node, "AND")
		if len(exprs) == 0 {
			return nil
		}
		result := exprs[0]
		for i := 1; i < len(exprs); i++ {
			result = &BoolAnd{Left: result, Right: exprs[i]}
		}
		return result
	case "OR":
		exprs := flattenRuleNodeToExprs(node, "OR")
		if len(exprs) == 0 {
			return nil
		}
		result := exprs[0]
		for i := 1; i < len(exprs); i++ {
			result = &BoolOr{Left: result, Right: exprs[i]}
		}
		return result
	case "NOT":
		if len(node.Conditions) == 1 {
			return &BoolNot{Expr: decompileRuleNodeToExpr(&node.Conditions[0])}
		}
	}
	return nil
}

// flattenRuleNodeToExprs collects leaves of nested same-operator nodes into a flat slice.
func flattenRuleNodeToExprs(node *config.RuleCombination, op string) []BoolExpr {
	if node.Operator == op {
		var exprs []BoolExpr
		for i := range node.Conditions {
			exprs = append(exprs, flattenRuleNodeToExprs(&node.Conditions[i], op)...)
		}
		return exprs
	}
	return []BoolExpr{decompileRuleNodeToExpr(node)}
}

// ---------- Formatting Helpers ----------

func (d *decompiler) write(format string, args ...interface{}) {
	fmt.Fprintf(&d.sb, format, args...)
}

func (d *decompiler) writeSection(name string) {
	d.write("# =============================================================================\n")
	d.write("# %s\n", name)
	d.write("# =============================================================================\n\n")
}

func formatStringArray(items []string) string {
	quoted := make([]string, len(items))
	for i, item := range items {
		quoted[i] = fmt.Sprintf("%q", item)
	}
	return "[" + strings.Join(quoted, ", ") + "]"
}

func formatIntArray(items []int) string {
	parts := make([]string, len(items))
	for i, item := range items {
		parts[i] = fmt.Sprintf("%d", item)
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func stringsToArray(items []string) ArrayValue {
	vals := make([]Value, len(items))
	for i, s := range items {
		vals[i] = StringValue{V: s}
	}
	return ArrayValue{Items: vals}
}

func structureFeatureValue(feature config.StructureFeature) ObjectValue {
	fields := map[string]Value{
		"type":   StringValue{V: feature.Type},
		"source": structureSourceValue(feature.Source),
	}
	return ObjectValue{Fields: fields}
}

func structureSourceValue(source config.StructureSource) ObjectValue {
	fields := map[string]Value{
		"type": StringValue{V: source.Type},
	}
	if source.Pattern != "" {
		fields["pattern"] = StringValue{V: source.Pattern}
	}
	if len(source.Keywords) > 0 {
		fields["keywords"] = stringsToArray(source.Keywords)
	}
	if source.CaseSensitive {
		fields["case_sensitive"] = BoolValue{V: true}
	}
	if len(source.Sequences) > 0 {
		items := make([]Value, 0, len(source.Sequences))
		for _, sequence := range source.Sequences {
			items = append(items, stringsToArray(sequence))
		}
		fields["sequences"] = ArrayValue{Items: items}
	}
	return ObjectValue{Fields: fields}
}

func structurePredicateValue(predicate *config.NumericPredicate) ObjectValue {
	fields := make(map[string]Value)
	if predicate.GT != nil {
		fields["gt"] = FloatValue{V: *predicate.GT}
	}
	if predicate.GTE != nil {
		fields["gte"] = FloatValue{V: *predicate.GTE}
	}
	if predicate.LT != nil {
		fields["lt"] = FloatValue{V: *predicate.LT}
	}
	if predicate.LTE != nil {
		fields["lte"] = FloatValue{V: *predicate.LTE}
	}
	return ObjectValue{Fields: fields}
}

func structureFeatureToMap(feature config.StructureFeature) map[string]interface{} {
	values := map[string]interface{}{
		"type":   feature.Type,
		"source": structureSourceToMap(feature.Source),
	}
	return values
}

func structureSourceToMap(source config.StructureSource) map[string]interface{} {
	values := map[string]interface{}{
		"type": source.Type,
	}
	if source.Pattern != "" {
		values["pattern"] = source.Pattern
	}
	if len(source.Keywords) > 0 {
		values["keywords"] = source.Keywords
	}
	if source.CaseSensitive {
		values["case_sensitive"] = true
	}
	if len(source.Sequences) > 0 {
		values["sequences"] = source.Sequences
	}
	return values
}

func conversationFeatureValue(feature config.ConversationFeature) ObjectValue {
	fields := map[string]Value{
		"type":   StringValue{V: feature.Type},
		"source": conversationSourceValue(feature.Source),
	}
	return ObjectValue{Fields: fields}
}

func conversationSourceValue(source config.ConversationSource) ObjectValue {
	fields := map[string]Value{
		"type": StringValue{V: source.Type},
	}
	if source.Role != "" {
		fields["role"] = StringValue{V: source.Role}
	}
	return ObjectValue{Fields: fields}
}

func conversationFeatureToMap(feature config.ConversationFeature) map[string]interface{} {
	return map[string]interface{}{
		"type":   feature.Type,
		"source": conversationSourceToMap(feature.Source),
	}
}

func conversationSourceToMap(source config.ConversationSource) map[string]interface{} {
	values := map[string]interface{}{
		"type": source.Type,
	}
	if source.Role != "" {
		values["role"] = source.Role
	}
	return values
}

func structurePredicateToMap(predicate *config.NumericPredicate) map[string]interface{} {
	values := make(map[string]interface{})
	if predicate == nil {
		return values
	}
	if predicate.GT != nil {
		values["gt"] = *predicate.GT
	}
	if predicate.GTE != nil {
		values["gte"] = *predicate.GTE
	}
	if predicate.LT != nil {
		values["lt"] = *predicate.LT
	}
	if predicate.LTE != nil {
		values["lte"] = *predicate.LTE
	}
	return values
}

func formatProjectionScoreInputs(inputs []config.ProjectionScoreInput) string {
	parts := make([]string, 0, len(inputs))
	for _, input := range inputs {
		fields := []string{
			fmt.Sprintf("type: %q", input.Type),
			fmt.Sprintf("weight: %g", input.Weight),
		}
		if input.Name != "" {
			fields = append(fields, fmt.Sprintf("name: %q", input.Name))
		}
		if input.KB != "" {
			fields = append(fields, fmt.Sprintf("kb: %q", input.KB))
		}
		if input.Metric != "" {
			fields = append(fields, fmt.Sprintf("metric: %q", input.Metric))
		}
		if input.ValueSource != "" {
			fields = append(fields, fmt.Sprintf("value_source: %q", input.ValueSource))
		}
		if input.Match != 0 {
			fields = append(fields, fmt.Sprintf("match: %g", input.Match))
		}
		if input.Miss != 0 {
			fields = append(fields, fmt.Sprintf("miss: %g", input.Miss))
		}
		parts = append(parts, "{ "+strings.Join(fields, ", ")+" }")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

func formatProjectionMappingCalibration(calibration *config.ProjectionMappingCalibration) string {
	if calibration == nil {
		return "{}"
	}
	fields := make([]string, 0, 2)
	if calibration.Method != "" {
		fields = append(fields, fmt.Sprintf("method: %q", calibration.Method))
	}
	if calibration.Slope != 0 {
		fields = append(fields, fmt.Sprintf("slope: %g", calibration.Slope))
	}
	return "{ " + strings.Join(fields, ", ") + " }"
}

func formatProjectionMappingOutputs(outputs []config.ProjectionMappingOutput) string {
	parts := make([]string, 0, len(outputs))
	for _, output := range outputs {
		fields := []string{fmt.Sprintf("name: %q", output.Name)}
		if output.GT != nil {
			fields = append(fields, fmt.Sprintf("gt: %g", *output.GT))
		}
		if output.GTE != nil {
			fields = append(fields, fmt.Sprintf("gte: %g", *output.GTE))
		}
		if output.LT != nil {
			fields = append(fields, fmt.Sprintf("lt: %g", *output.LT))
		}
		if output.LTE != nil {
			fields = append(fields, fmt.Sprintf("lte: %g", *output.LTE))
		}
		parts = append(parts, "{ "+strings.Join(fields, ", ")+" }")
	}
	return "[" + strings.Join(parts, ", ") + "]"
}

// algorithmToFields converts an AlgorithmConfig into a map[string]Value for the AST.
func (d *decompiler) algorithmToFields(algo *config.AlgorithmConfig) map[string]Value {
	fields := make(map[string]Value)
	// Emit top-level on_error only for types that don't carry their own sub-config on_error
	switch algo.Type {
	case "confidence", "ratings", "remom":
		// on_error lives in the sub-config
	default:
		if algo.OnError != "" {
			fields["on_error"] = StringValue{V: algo.OnError}
		}
	}
	switch algo.Type {
	case "remom":
		if r := algo.ReMoM; r != nil {
			if len(r.BreadthSchedule) > 0 {
				var items []Value
				for _, v := range r.BreadthSchedule {
					items = append(items, IntValue{V: v})
				}
				fields["breadth_schedule"] = ArrayValue{Items: items}
			}
			if r.ModelDistribution != "" {
				fields["model_distribution"] = StringValue{V: r.ModelDistribution}
			}
			if r.CompactionStrategy != "" {
				fields["compaction_strategy"] = StringValue{V: r.CompactionStrategy}
			}
			if r.OnError != "" {
				fields["on_error"] = StringValue{V: r.OnError}
			}
			if r.Temperature != 0 {
				fields["temperature"] = FloatValue{V: r.Temperature}
			}
			if r.MaxConcurrent != 0 {
				fields["max_concurrent"] = IntValue{V: r.MaxConcurrent}
			}
		}
	case "latency_aware":
		if l := algo.LatencyAware; l != nil {
			if l.TPOTPercentile != 0 {
				fields["tpot_percentile"] = IntValue{V: l.TPOTPercentile}
			}
			if l.TTFTPercentile != 0 {
				fields["ttft_percentile"] = IntValue{V: l.TTFTPercentile}
			}
		}
	case "confidence":
		if c := algo.Confidence; c != nil {
			if c.ConfidenceMethod != "" {
				fields["confidence_method"] = StringValue{V: c.ConfidenceMethod}
			}
			if c.Threshold != 0 {
				fields["threshold"] = FloatValue{V: c.Threshold}
			}
			if c.OnError != "" {
				fields["on_error"] = StringValue{V: c.OnError}
			}
			if c.EscalationOrder != "" {
				fields["escalation_order"] = StringValue{V: c.EscalationOrder}
			}
		}
	case "elo":
		if e := algo.Elo; e != nil {
			if e.InitialRating != 0 {
				fields["initial_rating"] = FloatValue{V: e.InitialRating}
			}
			if e.KFactor != 0 {
				fields["k_factor"] = FloatValue{V: e.KFactor}
			}
			if e.StoragePath != "" {
				fields["storage_path"] = StringValue{V: e.StoragePath}
			}
		}
	}
	return fields
}

// pluginConfigToFields converts a DecisionPlugin's typed Configuration into
// a map[string]Value suitable for the AST PluginRef.Fields.
func pluginConfigToFields(p *config.DecisionPlugin) map[string]Value {
	fields := make(map[string]Value)
	switch p.Type {
	case "system_prompt":
		cfg, ok := decodePluginConfig[config.SystemPromptPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled != nil {
			fields["enabled"] = BoolValue{V: *cfg.Enabled}
		}
		if cfg.SystemPrompt != "" {
			fields["system_prompt"] = StringValue{V: cfg.SystemPrompt}
		}
		if cfg.Mode != "" {
			fields["mode"] = StringValue{V: cfg.Mode}
		}
	case "semantic-cache":
		cfg, ok := decodePluginConfig[config.SemanticCachePluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.SimilarityThreshold != nil {
			fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
		}
	case "router_replay":
		cfg, ok := decodePluginConfig[config.RouterReplayPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.MaxRecords != 0 {
			fields["max_records"] = IntValue{V: cfg.MaxRecords}
		}
		if cfg.CaptureRequestBody {
			fields["capture_request_body"] = BoolValue{V: true}
		}
		if cfg.CaptureResponseBody {
			fields["capture_response_body"] = BoolValue{V: true}
		}
		if cfg.MaxBodyBytes != 0 {
			fields["max_body_bytes"] = IntValue{V: cfg.MaxBodyBytes}
		}
	case "memory":
		cfg, ok := decodePluginConfig[config.MemoryPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.RetrievalLimit != nil {
			fields["retrieval_limit"] = IntValue{V: *cfg.RetrievalLimit}
		}
		if cfg.SimilarityThreshold != nil {
			fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
		}
		if cfg.AutoStore != nil {
			fields["auto_store"] = BoolValue{V: *cfg.AutoStore}
		}
	case "hallucination":
		cfg, ok := decodePluginConfig[config.HallucinationPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
	case "image_gen":
		cfg, ok := decodePluginConfig[config.ImageGenPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.Backend != "" {
			fields["backend"] = StringValue{V: cfg.Backend}
		}
	case "fast_response":
		cfg, ok := decodePluginConfig[config.FastResponsePluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Message != "" {
			fields["message"] = StringValue{V: cfg.Message}
		}
	case "request_params":
		cfg, ok := decodePluginConfig[config.RequestParamsPluginConfig](p)
		if !ok {
			return fields
		}
		if len(cfg.BlockedParams) > 0 {
			var items []Value
			for _, s := range cfg.BlockedParams {
				items = append(items, StringValue{V: s})
			}
			fields["blocked_params"] = ArrayValue{Items: items}
		}
		if cfg.MaxTokensLimit != nil {
			fields["max_tokens_limit"] = IntValue{V: *cfg.MaxTokensLimit}
		}
		if cfg.MaxN != nil {
			fields["max_n"] = IntValue{V: *cfg.MaxN}
		}
		if cfg.StripUnknown {
			fields["strip_unknown"] = BoolValue{V: true}
		}
	case "tools":
		cfg, ok := decodePluginConfig[config.ToolsPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.Mode != "" {
			fields["mode"] = StringValue{V: cfg.Mode}
		}
		if cfg.SemanticSelection != nil {
			fields["semantic_selection"] = BoolValue{V: *cfg.SemanticSelection}
		}
		if len(cfg.AllowTools) > 0 {
			fields["allow_tools"] = stringsToArray(cfg.AllowTools)
		}
		if len(cfg.BlockTools) > 0 {
			fields["block_tools"] = stringsToArray(cfg.BlockTools)
		}
	case "rag":
		cfg, ok := decodePluginConfig[config.RAGPluginConfig](p)
		if !ok {
			return fields
		}
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.Backend != "" {
			fields["backend"] = StringValue{V: cfg.Backend}
		}
		if cfg.SimilarityThreshold != nil {
			fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
		}
		if cfg.TopK != nil {
			fields["top_k"] = IntValue{V: *cfg.TopK}
		}
		if cfg.MaxContextLength != nil {
			fields["max_context_length"] = IntValue{V: *cfg.MaxContextLength}
		}
		if cfg.InjectionMode != "" {
			fields["injection_mode"] = StringValue{V: cfg.InjectionMode}
		}
	case "header_mutation":
		cfg, ok := decodePluginConfig[config.HeaderMutationPluginConfig](p)
		if !ok {
			return fields
		}
		if len(cfg.Add) > 0 {
			var items []Value
			for _, h := range cfg.Add {
				items = append(items, ObjectValue{Fields: map[string]Value{
					"name":  StringValue{V: h.Name},
					"value": StringValue{V: h.Value},
				}})
			}
			fields["add"] = ArrayValue{Items: items}
		}
		if len(cfg.Update) > 0 {
			var items []Value
			for _, h := range cfg.Update {
				items = append(items, ObjectValue{Fields: map[string]Value{
					"name":  StringValue{V: h.Name},
					"value": StringValue{V: h.Value},
				}})
			}
			fields["update"] = ArrayValue{Items: items}
		}
		if len(cfg.Delete) > 0 {
			fields["delete"] = stringsToArray(cfg.Delete)
		}
	}
	return fields
}

// decompileComposerObj outputs a RuleCombination as an inline DSL object
// that matches the YAML structure: { operator: "AND", conditions: [...] }
func decompileComposerObj(node *config.RuleCombination) string {
	if node == nil {
		return "{}"
	}
	if node.Type != "" {
		return fmt.Sprintf("{ type: %q, name: %q }", node.Type, node.Name)
	}
	var parts []string
	for i := range node.Conditions {
		parts = append(parts, decompileComposerObj(&node.Conditions[i]))
	}
	return fmt.Sprintf("{ operator: %q, conditions: [%s] }", node.Operator, strings.Join(parts, ", "))
}

func sanitizeName(name string) string {
	return strings.ReplaceAll(name, "-", "_")
}

// quoteName returns the name quoted if it contains characters that are not
// valid in a bare DSL identifier (e.g. spaces), otherwise returns it as-is.
func quoteName(name string) string {
	for _, ch := range name {
		if !isIdentPart(ch) {
			return fmt.Sprintf("%q", name)
		}
	}
	return name
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// Formatter formats DSL input into canonical routing-only DSL text.
func Format(input string) (string, error) {
	prog, errs := Parse(input)
	if len(errs) > 0 {
		return "", fmt.Errorf("parse errors: %v", errs)
	}

	cfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		return "", fmt.Errorf("compile errors: %v", compileErrs)
	}

	formatted, err := DecompileRouting(cfg)
	if err != nil {
		return "", err
	}
	if len(prog.TestBlocks) == 0 {
		return formatted, nil
	}
	return appendFormattedTestBlocks(formatted, prog.TestBlocks), nil
}
