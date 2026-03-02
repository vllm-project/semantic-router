package dsl

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Decompile converts a RouterConfig back into DSL source text.
func Decompile(cfg *config.RouterConfig) (string, error) {
	d := &decompiler{cfg: cfg}
	return d.decompile()
}

// DecompileToAST converts a RouterConfig into a DSL AST Program.
func DecompileToAST(cfg *config.RouterConfig) *Program {
	d := &decompiler{cfg: cfg}
	return d.buildAST()
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

// ---------- Main Flow ----------

func (d *decompiler) decompile() (string, error) {
	d.pluginTemplates = make(map[string]*pluginTemplate)
	d.extractPluginTemplates()

	d.writeSection("SIGNALS")
	d.decompileSignals()

	if len(d.pluginTemplates) > 0 {
		d.writeSection("PLUGINS")
		d.decompilePluginTemplates()
	}

	d.writeSection("ROUTES")
	d.decompileDecisions()

	d.writeSection("BACKENDS")
	d.decompileBackends()

	d.writeSection("GLOBAL")
	d.decompileGlobal()

	return d.sb.String(), nil
}

func (d *decompiler) buildAST() *Program {
	prog := &Program{}

	// Signals
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
	for _, jb := range d.cfg.JailbreakRules {
		prog.Signals = append(prog.Signals, d.jailbreakToSignal(&jb))
	}
	for _, pii := range d.cfg.PIIRules {
		prog.Signals = append(prog.Signals, d.piiToSignal(&pii))
	}

	// Routes
	for _, dec := range d.cfg.Decisions {
		prog.Routes = append(prog.Routes, d.decisionToRoute(&dec))
	}

	// Backends
	for _, ep := range d.cfg.VLLMEndpoints {
		prog.Backends = append(prog.Backends, d.vllmEndpointToBackend(&ep))
	}
	if d.cfg.ProviderProfiles != nil {
		names := sortedKeys(d.cfg.ProviderProfiles)
		for _, name := range names {
			pp := d.cfg.ProviderProfiles[name]
			prog.Backends = append(prog.Backends, d.providerProfileToBackend(name, &pp))
		}
	}
	if d.cfg.EmbeddingModels.MmBertModelPath != "" || d.cfg.EmbeddingModels.BertModelPath != "" {
		prog.Backends = append(prog.Backends, d.embeddingModelToBackend())
	}
	if d.cfg.SemanticCache.Enabled {
		prog.Backends = append(prog.Backends, d.semanticCacheToBackend())
	}
	if d.cfg.Memory.Enabled {
		prog.Backends = append(prog.Backends, d.memoryToBackend())
	}
	if d.cfg.ResponseAPI.Enabled {
		prog.Backends = append(prog.Backends, d.responseAPIToBackend())
	}

	// Global
	prog.Global = d.buildGlobalDecl()

	return prog
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

	for _, pref := range d.cfg.PreferenceRules {
		d.write("SIGNAL preference %s {\n", quoteName(pref.Name))
		if pref.Description != "" {
			d.write("  description: %q\n", pref.Description)
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

		// WHEN expression
		ruleExpr := decompileRuleNode(&dec.Rules)
		if ruleExpr != "" {
			d.write("  WHEN %s\n", ruleExpr)
		}

		// MODEL list
		if len(dec.ModelRefs) > 0 {
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
	switch cfg := p.Configuration.(type) {
	case config.SystemPromptPluginConfig:
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
	case config.SemanticCachePluginConfig:
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.SimilarityThreshold != nil {
			fmt.Fprintf(&sb, "    similarity_threshold: %v\n", *cfg.SimilarityThreshold)
		}
	case config.RouterReplayPluginConfig:
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
	case config.MemoryPluginConfig:
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
	case config.HallucinationPluginConfig:
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.UseNLI {
			fmt.Fprintf(&sb, "    use_nli: true\n")
		}
		if cfg.HallucinationAction != "" {
			fmt.Fprintf(&sb, "    hallucination_action: %q\n", cfg.HallucinationAction)
		}
	case config.ImageGenPluginConfig:
		if cfg.Enabled {
			fmt.Fprintf(&sb, "    enabled: true\n")
		}
		if cfg.Backend != "" {
			fmt.Fprintf(&sb, "    backend: %q\n", cfg.Backend)
		}
	case config.FastResponsePluginConfig:
		if cfg.Message != "" {
			fmt.Fprintf(&sb, "    message: %q\n", cfg.Message)
		}
	case config.RAGPluginConfig:
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
	case config.HeaderMutationPluginConfig:
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
	// Also handle map[string]interface{} from raw YAML deserialization
	if m, ok := p.Configuration.(map[string]interface{}); ok {
		for k, v := range m {
			switch val := v.(type) {
			case bool:
				fmt.Fprintf(&sb, "    %s: %v\n", k, val)
			case string:
				if val != "" {
					fmt.Fprintf(&sb, "    %s: %q\n", k, val)
				}
			case int:
				if val != 0 {
					fmt.Fprintf(&sb, "    %s: %d\n", k, val)
				}
			case float64:
				if val != 0 {
					fmt.Fprintf(&sb, "    %s: %v\n", k, val)
				}
			}
		}
	}
	return sb.String()
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
	// Pull param_size and reasoning_family from model_config
	if mc, ok := modelConfig[mr.Model]; ok {
		if mc.ParamSize != "" {
			opts = append(opts, fmt.Sprintf("param_size = %q", mc.ParamSize))
		}
		if mc.ReasoningFamily != "" {
			opts = append(opts, fmt.Sprintf("reasoning_family = %q", mc.ReasoningFamily))
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

// ---------- Backend Decompilation ----------

func (d *decompiler) decompileBackends() {
	for _, ep := range d.cfg.VLLMEndpoints {
		d.write("BACKEND vllm_endpoint %s {\n", quoteName(ep.Name))
		if ep.Address != "" {
			d.write("  address: %q\n", ep.Address)
		}
		if ep.Port != 0 {
			d.write("  port: %d\n", ep.Port)
		}
		if ep.Weight != 0 {
			d.write("  weight: %d\n", ep.Weight)
		}
		if ep.Type != "" {
			d.write("  type: %q\n", ep.Type)
		}
		if ep.APIKey != "" {
			d.write("  api_key: %q\n", ep.APIKey)
		}
		if ep.ProviderProfileName != "" {
			d.write("  provider_profile: %q\n", ep.ProviderProfileName)
		}
		if ep.Model != "" {
			d.write("  model: %q\n", ep.Model)
		}
		if ep.Protocol != "" {
			d.write("  protocol: %q\n", ep.Protocol)
		}
		d.write("}\n\n")
	}

	if d.cfg.ProviderProfiles != nil {
		names := sortedKeys(d.cfg.ProviderProfiles)
		for _, name := range names {
			pp := d.cfg.ProviderProfiles[name]
			d.write("BACKEND provider_profile %s {\n", quoteName(name))
			if pp.Type != "" {
				d.write("  type: %q\n", pp.Type)
			}
			if pp.BaseURL != "" {
				d.write("  base_url: %q\n", pp.BaseURL)
			}
			if pp.AuthHeader != "" {
				d.write("  auth_header: %q\n", pp.AuthHeader)
			}
			if pp.APIVersion != "" {
				d.write("  api_version: %q\n", pp.APIVersion)
			}
			if pp.ChatPath != "" {
				d.write("  chat_path: %q\n", pp.ChatPath)
			}
			if len(pp.ExtraHeaders) > 0 {
				d.write("  extra_headers: {\n")
				for k, v := range pp.ExtraHeaders {
					d.write("    %s: %q\n", k, v)
				}
				d.write("  }\n")
			}
			d.write("}\n\n")
		}
	}

	if d.cfg.EmbeddingModels.MmBertModelPath != "" || d.cfg.EmbeddingModels.BertModelPath != "" {
		d.write("BACKEND embedding_model main {\n")
		if d.cfg.EmbeddingModels.MmBertModelPath != "" {
			d.write("  mmbert_model_path: %q\n", d.cfg.EmbeddingModels.MmBertModelPath)
		}
		if d.cfg.EmbeddingModels.BertModelPath != "" {
			d.write("  bert_model_path: %q\n", d.cfg.EmbeddingModels.BertModelPath)
		}
		if d.cfg.EmbeddingModels.UseCPU {
			d.write("  use_cpu: true\n")
		}
		hnsw := d.cfg.EmbeddingModels.HNSWConfig
		if hnsw.ModelType != "" || hnsw.PreloadEmbeddings || hnsw.TargetDimension != 0 {
			d.write("  hnsw_config: {\n")
			if hnsw.ModelType != "" {
				d.write("    model_type: %q\n", hnsw.ModelType)
			}
			if hnsw.PreloadEmbeddings {
				d.write("    preload_embeddings: true\n")
			}
			if hnsw.TargetDimension != 0 {
				d.write("    target_dimension: %d\n", hnsw.TargetDimension)
			}
			if hnsw.MinScoreThreshold != 0 {
				d.write("    min_score_threshold: %v\n", hnsw.MinScoreThreshold)
			}
			d.write("  }\n")
		}
		d.write("}\n\n")
	}

	if d.cfg.SemanticCache.Enabled {
		d.write("BACKEND semantic_cache main {\n")
		d.write("  enabled: true\n")
		if d.cfg.SemanticCache.BackendType != "" {
			d.write("  backend_type: %q\n", d.cfg.SemanticCache.BackendType)
		}
		if d.cfg.SemanticCache.SimilarityThreshold != nil {
			d.write("  similarity_threshold: %v\n", *d.cfg.SemanticCache.SimilarityThreshold)
		}
		if d.cfg.SemanticCache.MaxEntries != 0 {
			d.write("  max_entries: %d\n", d.cfg.SemanticCache.MaxEntries)
		}
		if d.cfg.SemanticCache.TTLSeconds != 0 {
			d.write("  ttl_seconds: %d\n", d.cfg.SemanticCache.TTLSeconds)
		}
		if d.cfg.SemanticCache.EvictionPolicy != "" {
			d.write("  eviction_policy: %q\n", d.cfg.SemanticCache.EvictionPolicy)
		}
		d.write("}\n\n")
	}

	if d.cfg.Memory.Enabled {
		d.write("BACKEND memory mem {\n")
		d.write("  enabled: true\n")
		if d.cfg.Memory.AutoStore {
			d.write("  auto_store: true\n")
		}
		if d.cfg.Memory.DefaultRetrievalLimit != 0 {
			d.write("  default_retrieval_limit: %d\n", d.cfg.Memory.DefaultRetrievalLimit)
		}
		if d.cfg.Memory.DefaultSimilarityThreshold != 0 {
			d.write("  default_similarity_threshold: %v\n", d.cfg.Memory.DefaultSimilarityThreshold)
		}
		d.write("}\n\n")
	}

	if d.cfg.ResponseAPI.Enabled {
		d.write("BACKEND response_api resp {\n")
		d.write("  enabled: true\n")
		if d.cfg.ResponseAPI.StoreBackend != "" {
			d.write("  store_backend: %q\n", d.cfg.ResponseAPI.StoreBackend)
		}
		if d.cfg.ResponseAPI.TTLSeconds != 0 {
			d.write("  ttl_seconds: %d\n", d.cfg.ResponseAPI.TTLSeconds)
		}
		if d.cfg.ResponseAPI.MaxResponses != 0 {
			d.write("  max_responses: %d\n", d.cfg.ResponseAPI.MaxResponses)
		}
		d.write("}\n\n")
	}
}

// ---------- Global Decompilation ----------

func (d *decompiler) decompileGlobal() {
	d.write("GLOBAL {\n")
	if d.cfg.DefaultModel != "" {
		d.write("  default_model: %q\n", d.cfg.DefaultModel)
	}
	if d.cfg.Strategy != "" {
		d.write("  strategy: %q\n", d.cfg.Strategy)
	}
	if d.cfg.DefaultReasoningEffort != "" {
		d.write("  default_reasoning_effort: %q\n", d.cfg.DefaultReasoningEffort)
	}

	// reasoning_families
	if len(d.cfg.ReasoningFamilies) > 0 {
		d.write("\n  reasoning_families: {\n")
		for name, rfc := range d.cfg.ReasoningFamilies {
			d.write("    %s: { type: %q, parameter: %q }\n", name, rfc.Type, rfc.Parameter)
		}
		d.write("  }\n")
	}

	// prompt_guard
	if d.cfg.PromptGuard.Enabled || d.cfg.PromptGuard.Threshold != 0 {
		d.write("\n  prompt_guard: {\n")
		if d.cfg.PromptGuard.Enabled {
			d.write("    enabled: true\n")
		}
		if d.cfg.PromptGuard.Threshold != 0 {
			d.write("    threshold: %v\n", d.cfg.PromptGuard.Threshold)
		}
		d.write("  }\n")
	}

	// hallucination_mitigation
	hm := d.cfg.HallucinationMitigation
	if hm.Enabled || hm.FactCheckModel.Threshold != 0 || hm.HallucinationModel.Threshold != 0 {
		d.write("\n  hallucination_mitigation: {\n")
		if hm.Enabled {
			d.write("    enabled: true\n")
		}
		if hm.OnHallucinationDetected != "" {
			d.write("    on_hallucination_detected: %q\n", hm.OnHallucinationDetected)
		}
		if hm.FactCheckModel.Threshold != 0 {
			d.write("    fact_check_model: { threshold: %v }\n", hm.FactCheckModel.Threshold)
		}
		if hm.HallucinationModel.Threshold != 0 {
			d.write("    hallucination_model: { threshold: %v }\n", hm.HallucinationModel.Threshold)
		}
		if hm.NLIModel.Threshold != 0 {
			d.write("    nli_model: { threshold: %v }\n", hm.NLIModel.Threshold)
		}
		d.write("  }\n")
	}

	// model_selection
	if d.cfg.ModelSelection.Enabled || d.cfg.ModelSelection.Method != "" {
		d.write("\n  model_selection: {\n")
		if d.cfg.ModelSelection.Enabled {
			d.write("    enabled: true\n")
		}
		if d.cfg.ModelSelection.Method != "" {
			d.write("    method: %q\n", d.cfg.ModelSelection.Method)
		}
		d.write("  }\n")
	}

	// looper
	if d.cfg.Looper.Endpoint != "" {
		d.write("\n  looper: {\n")
		d.write("    endpoint: %q\n", d.cfg.Looper.Endpoint)
		if d.cfg.Looper.TimeoutSeconds != 0 {
			d.write("    timeout_seconds: %d\n", d.cfg.Looper.TimeoutSeconds)
		}
		d.write("  }\n")
	}

	// observability
	if d.cfg.Observability.Tracing.Enabled || (d.cfg.Observability.Metrics.Enabled != nil && *d.cfg.Observability.Metrics.Enabled) {
		d.write("\n  observability: {\n")
		if d.cfg.Observability.Metrics.Enabled != nil && *d.cfg.Observability.Metrics.Enabled {
			d.write("    metrics: { enabled: true }\n")
		}
		if d.cfg.Observability.Tracing.Enabled {
			d.write("    tracing: {\n")
			d.write("      enabled: true\n")
			if d.cfg.Observability.Tracing.Provider != "" {
				d.write("      provider: %q\n", d.cfg.Observability.Tracing.Provider)
			}
			exp := d.cfg.Observability.Tracing.Exporter
			if exp.Type != "" || exp.Endpoint != "" {
				d.write("      exporter: {\n")
				if exp.Type != "" {
					d.write("        type: %q\n", exp.Type)
				}
				if exp.Endpoint != "" {
					d.write("        endpoint: %q\n", exp.Endpoint)
				}
				if exp.Insecure {
					d.write("        insecure: true\n")
				}
				d.write("      }\n")
			}
			samp := d.cfg.Observability.Tracing.Sampling
			if samp.Type != "" {
				d.write("      sampling: { type: %q, rate: %v }\n", samp.Type, samp.Rate)
			}
			res := d.cfg.Observability.Tracing.Resource
			if res.ServiceName != "" {
				d.write("      resource: {\n")
				d.write("        service_name: %q\n", res.ServiceName)
				if res.ServiceVersion != "" {
					d.write("        service_version: %q\n", res.ServiceVersion)
				}
				if res.DeploymentEnvironment != "" {
					d.write("        deployment_environment: %q\n", res.DeploymentEnvironment)
				}
				d.write("      }\n")
			}
			d.write("    }\n")
		}
		d.write("  }\n")
	}

	// authz
	if d.cfg.Authz.FailOpen {
		d.write("\n  authz: {\n")
		d.write("    fail_open: true\n")
		d.write("  }\n")
	}

	// ratelimit
	if d.cfg.RateLimit.FailOpen {
		d.write("\n  ratelimit: {\n")
		d.write("    fail_open: true\n")
		d.write("  }\n")
	}

	d.write("}\n")
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

func (d *decompiler) preferenceToSignal(pref *config.PreferenceRule) *SignalDecl {
	fields := make(map[string]Value)
	if pref.Description != "" {
		fields["description"] = StringValue{V: pref.Description}
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

func (d *decompiler) decisionToRoute(dec *config.Decision) *RouteDecl {
	route := &RouteDecl{
		Name:        dec.Name,
		Description: dec.Description,
		Priority:    dec.Priority,
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
		// Pull param_size and reasoning_family from model_config
		if mc, ok := d.cfg.ModelConfig[mr.Model]; ok {
			ref.ParamSize = mc.ParamSize
			ref.ReasoningFamily = mc.ReasoningFamily
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

	return route
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

// ---------- Backend AST Helpers ----------

func (d *decompiler) vllmEndpointToBackend(ep *config.VLLMEndpoint) *BackendDecl {
	fields := make(map[string]Value)
	if ep.Address != "" {
		fields["address"] = StringValue{V: ep.Address}
	}
	if ep.Port != 0 {
		fields["port"] = IntValue{V: ep.Port}
	}
	if ep.Weight != 0 {
		fields["weight"] = IntValue{V: ep.Weight}
	}
	if ep.Type != "" {
		fields["type"] = StringValue{V: ep.Type}
	}
	if ep.Model != "" {
		fields["model"] = StringValue{V: ep.Model}
	}
	if ep.Protocol != "" {
		fields["protocol"] = StringValue{V: ep.Protocol}
	}
	return &BackendDecl{BackendType: "vllm_endpoint", Name: ep.Name, Fields: fields}
}

func (d *decompiler) providerProfileToBackend(name string, pp *config.ProviderProfile) *BackendDecl {
	fields := make(map[string]Value)
	if pp.Type != "" {
		fields["type"] = StringValue{V: pp.Type}
	}
	if pp.BaseURL != "" {
		fields["base_url"] = StringValue{V: pp.BaseURL}
	}
	return &BackendDecl{BackendType: "provider_profile", Name: name, Fields: fields}
}

func (d *decompiler) embeddingModelToBackend() *BackendDecl {
	fields := make(map[string]Value)
	em := d.cfg.EmbeddingModels
	if em.MmBertModelPath != "" {
		fields["mmbert_model_path"] = StringValue{V: em.MmBertModelPath}
	}
	if em.BertModelPath != "" {
		fields["bert_model_path"] = StringValue{V: em.BertModelPath}
	}
	if em.UseCPU {
		fields["use_cpu"] = BoolValue{V: true}
	}
	return &BackendDecl{BackendType: "embedding_model", Name: "main", Fields: fields}
}

func (d *decompiler) semanticCacheToBackend() *BackendDecl {
	fields := make(map[string]Value)
	fields["enabled"] = BoolValue{V: true}
	if d.cfg.SemanticCache.BackendType != "" {
		fields["backend_type"] = StringValue{V: d.cfg.SemanticCache.BackendType}
	}
	return &BackendDecl{BackendType: "semantic_cache", Name: "main", Fields: fields}
}

func (d *decompiler) memoryToBackend() *BackendDecl {
	fields := make(map[string]Value)
	fields["enabled"] = BoolValue{V: true}
	return &BackendDecl{BackendType: "memory", Name: "mem", Fields: fields}
}

func (d *decompiler) responseAPIToBackend() *BackendDecl {
	fields := make(map[string]Value)
	fields["enabled"] = BoolValue{V: true}
	return &BackendDecl{BackendType: "response_api", Name: "resp", Fields: fields}
}

func (d *decompiler) buildGlobalDecl() *GlobalDecl {
	fields := make(map[string]Value)
	if d.cfg.DefaultModel != "" {
		fields["default_model"] = StringValue{V: d.cfg.DefaultModel}
	}
	if d.cfg.Strategy != "" {
		fields["strategy"] = StringValue{V: d.cfg.Strategy}
	}
	if d.cfg.DefaultReasoningEffort != "" {
		fields["default_reasoning_effort"] = StringValue{V: d.cfg.DefaultReasoningEffort}
	}

	// reasoning_families
	if len(d.cfg.ReasoningFamilies) > 0 {
		familyFields := make(map[string]Value)
		for name, rfc := range d.cfg.ReasoningFamilies {
			entryFields := make(map[string]Value)
			if rfc.Type != "" {
				entryFields["type"] = StringValue{V: rfc.Type}
			}
			if rfc.Parameter != "" {
				entryFields["parameter"] = StringValue{V: rfc.Parameter}
			}
			familyFields[name] = ObjectValue{Fields: entryFields}
		}
		fields["reasoning_families"] = ObjectValue{Fields: familyFields}
	}

	// hallucination_mitigation
	hm := d.cfg.HallucinationMitigation
	if hm.Enabled || hm.FactCheckModel.Threshold != 0 || hm.HallucinationModel.Threshold != 0 {
		hmFields := make(map[string]Value)
		if hm.Enabled {
			hmFields["enabled"] = BoolValue{V: true}
		}
		if hm.OnHallucinationDetected != "" {
			hmFields["on_hallucination_detected"] = StringValue{V: hm.OnHallucinationDetected}
		}
		if hm.FactCheckModel.Threshold != 0 {
			hmFields["fact_check_model"] = ObjectValue{Fields: map[string]Value{
				"threshold": FloatValue{V: float64(hm.FactCheckModel.Threshold)},
			}}
		}
		if hm.HallucinationModel.Threshold != 0 {
			hmFields["hallucination_model"] = ObjectValue{Fields: map[string]Value{
				"threshold": FloatValue{V: float64(hm.HallucinationModel.Threshold)},
			}}
		}
		if hm.NLIModel.Threshold != 0 {
			hmFields["nli_model"] = ObjectValue{Fields: map[string]Value{
				"threshold": FloatValue{V: float64(hm.NLIModel.Threshold)},
			}}
		}
		fields["hallucination_mitigation"] = ObjectValue{Fields: hmFields}
	}

	// authz
	if d.cfg.Authz.FailOpen {
		fields["authz"] = ObjectValue{Fields: map[string]Value{
			"fail_open": BoolValue{V: true},
		}}
	}

	// ratelimit
	if d.cfg.RateLimit.FailOpen {
		fields["ratelimit"] = ObjectValue{Fields: map[string]Value{
			"fail_open": BoolValue{V: true},
		}}
	}

	return &GlobalDecl{Fields: fields}
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
	switch cfg := p.Configuration.(type) {
	case config.SystemPromptPluginConfig:
		if cfg.Enabled != nil {
			fields["enabled"] = BoolValue{V: *cfg.Enabled}
		}
		if cfg.SystemPrompt != "" {
			fields["system_prompt"] = StringValue{V: cfg.SystemPrompt}
		}
		if cfg.Mode != "" {
			fields["mode"] = StringValue{V: cfg.Mode}
		}
	case config.SemanticCachePluginConfig:
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.SimilarityThreshold != nil {
			fields["similarity_threshold"] = FloatValue{V: float64(*cfg.SimilarityThreshold)}
		}
	case config.RouterReplayPluginConfig:
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
	case config.MemoryPluginConfig:
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
	case config.HallucinationPluginConfig:
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
	case config.ImageGenPluginConfig:
		if cfg.Enabled {
			fields["enabled"] = BoolValue{V: true}
		}
		if cfg.Backend != "" {
			fields["backend"] = StringValue{V: cfg.Backend}
		}
	case config.FastResponsePluginConfig:
		if cfg.Message != "" {
			fields["message"] = StringValue{V: cfg.Message}
		}
	case config.RAGPluginConfig:
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
	case config.HeaderMutationPluginConfig:
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

// Formatter formats a DSL AST into canonical DSL text with consistent indentation.
func Format(input string) (string, error) {
	prog, errs := Parse(input)
	if len(errs) > 0 {
		return "", fmt.Errorf("parse errors: %v", errs)
	}

	cfg, compileErrs := CompileAST(prog)
	if len(compileErrs) > 0 {
		return "", fmt.Errorf("compile errors: %v", compileErrs)
	}

	return Decompile(cfg)
}
