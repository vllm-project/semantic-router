package dsl

import (
	"fmt"
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Compiler transforms a DSL AST into a RouterConfig.
type Compiler struct {
	prog            *Program
	config          *config.RouterConfig
	pluginTemplates map[string]*PluginDecl // name → template
	errors          []error
}

// Compile parses a DSL source string and compiles it to a RouterConfig.
func Compile(input string) (*config.RouterConfig, []error) {
	prog, parseErrors := Parse(input)
	if len(parseErrors) > 0 {
		return nil, parseErrors
	}
	return CompileAST(prog)
}

// CompileAST compiles a parsed Program AST to a RouterConfig.
func CompileAST(prog *Program) (*config.RouterConfig, []error) {
	c := &Compiler{
		prog:            prog,
		config:          &config.RouterConfig{},
		pluginTemplates: make(map[string]*PluginDecl),
	}
	c.compile()
	if len(c.errors) > 0 {
		return nil, c.errors
	}
	return c.config, nil
}

func (c *Compiler) compile() {
	// 1. Register plugin templates
	for _, p := range c.prog.Plugins {
		c.pluginTemplates[p.Name] = p
	}

	// 2. Compile signals
	c.compileSignals()

	// 3. Compile projection partitions
	c.compileProjectionPartitions()

	// 4. Compile projections
	c.compileProjectionScores()
	c.compileProjectionMappings()

	// 5. Compile top-level model catalog
	c.compileModels()

	// 6. Compile routes (decisions)
	c.compileRoutes()
}

func (c *Compiler) compileProjectionPartitions() {
	for _, partitionDecl := range c.prog.ProjectionPartitions {
		c.validateSoftmaxDomainProjectionPartition(partitionDecl)
		partition := config.ProjectionPartition{
			Name:        partitionDecl.Name,
			Semantics:   partitionDecl.Semantics,
			Temperature: partitionDecl.Temperature,
			Members:     partitionDecl.Members,
			Default:     partitionDecl.Default,
		}
		c.config.Projections.Partitions = append(c.config.Projections.Partitions, partition)
	}
}

func (c *Compiler) compileProjectionScores() {
	for _, scoreDecl := range c.prog.ProjectionScores {
		score := config.ProjectionScore{
			Name:   scoreDecl.Name,
			Method: scoreDecl.Method,
			Inputs: make([]config.ProjectionScoreInput, 0, len(scoreDecl.Inputs)),
		}
		for _, inputDecl := range scoreDecl.Inputs {
			score.Inputs = append(score.Inputs, config.ProjectionScoreInput{
				Type:        inputDecl.SignalType,
				Name:        inputDecl.SignalName,
				KB:          inputDecl.KB,
				Metric:      inputDecl.Metric,
				Weight:      inputDecl.Weight,
				ValueSource: inputDecl.ValueSource,
				Match:       inputDecl.Match,
				Miss:        inputDecl.Miss,
			})
		}
		c.config.Projections.Scores = append(c.config.Projections.Scores, score)
	}
}

func (c *Compiler) compileProjectionMappings() {
	for _, mappingDecl := range c.prog.ProjectionMappings {
		mapping := config.ProjectionMapping{
			Name:   mappingDecl.Name,
			Source: mappingDecl.Source,
			Method: mappingDecl.Method,
		}
		if mappingDecl.Calibration != nil {
			mapping.Calibration = &config.ProjectionMappingCalibration{
				Method: mappingDecl.Calibration.Method,
				Slope:  mappingDecl.Calibration.Slope,
			}
		}
		for _, outputDecl := range mappingDecl.Outputs {
			mapping.Outputs = append(mapping.Outputs, config.ProjectionMappingOutput{
				Name: outputDecl.Name,
				LT:   outputDecl.LT,
				LTE:  outputDecl.LTE,
				GT:   outputDecl.GT,
				GTE:  outputDecl.GTE,
			})
		}
		c.config.Projections.Mappings = append(c.config.Projections.Mappings, mapping)
	}
}

// ---------- Signals ----------

//nolint:cyclop // Flat dispatch keeps signal-type ownership explicit at the compile entrypoint.
func (c *Compiler) compileSignals() {
	for _, s := range c.prog.Signals {
		switch s.SignalType {
		case "keyword":
			c.compileKeywordSignal(s)
		case "embedding":
			c.compileEmbeddingSignal(s)
		case "domain":
			c.compileDomainSignal(s)
		case "fact_check":
			c.compileFactCheckSignal(s)
		case "user_feedback":
			c.compileUserFeedbackSignal(s)
		case "reask":
			c.compileReaskSignal(s)
		case "preference":
			c.compilePreferenceSignal(s)
		case "language":
			c.compileLanguageSignal(s)
		case "context":
			c.compileContextSignal(s)
		case "structure":
			c.compileStructureSignal(s)
		case "complexity":
			c.compileComplexitySignal(s)
		case "modality":
			c.compileModalitySignal(s)
		case "session":
			c.compileSessionSignal(s)
		case "authz":
			c.compileAuthzSignal(s)
		case "jailbreak":
			c.compileJailbreakSignal(s)
		case "pii":
			c.compilePIISignal(s)
		case "kb":
			c.compileKBSignal(s)
		default:
			c.addError(s.Pos, "unknown signal type %q", s.SignalType)
		}
	}
}

func (c *Compiler) compileKeywordSignal(s *SignalDecl) {
	rule := config.KeywordRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "operator"); ok {
		rule.Operator = v
	}
	if v, ok := getStringArrayField(s.Fields, "keywords"); ok {
		rule.Keywords = v
	}
	if v, ok := getBoolField(s.Fields, "case_sensitive"); ok {
		rule.CaseSensitive = v
	}
	if v, ok := getStringField(s.Fields, "method"); ok {
		rule.Method = v
	}
	if v, ok := getBoolField(s.Fields, "fuzzy_match"); ok {
		rule.FuzzyMatch = v
	}
	if v, ok := getIntField(s.Fields, "fuzzy_threshold"); ok {
		rule.FuzzyThreshold = v
	}
	if v, ok := getFloat32Field(s.Fields, "bm25_threshold"); ok {
		rule.BM25Threshold = v
	}
	if v, ok := getFloat32Field(s.Fields, "ngram_threshold"); ok {
		rule.NgramThreshold = v
	}
	if v, ok := getIntField(s.Fields, "ngram_arity"); ok {
		rule.NgramArity = v
	}
	c.config.KeywordRules = append(c.config.KeywordRules, rule)
}

func (c *Compiler) compileEmbeddingSignal(s *SignalDecl) {
	rule := config.EmbeddingRule{Name: s.Name}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.SimilarityThreshold = v
	}
	if v, ok := getStringArrayField(s.Fields, "candidates"); ok {
		rule.Candidates = v
	}
	if v, ok := getStringField(s.Fields, "aggregation_method"); ok {
		rule.AggregationMethodConfiged = config.AggregationMethod(v)
	}
	c.config.EmbeddingRules = append(c.config.EmbeddingRules, rule)
}

func (c *Compiler) compileDomainSignal(s *SignalDecl) {
	cat := config.Category{}
	cat.Name = s.Name
	if v, ok := getStringField(s.Fields, "description"); ok {
		cat.Description = v
	}
	if v, ok := getStringArrayField(s.Fields, "mmlu_categories"); ok {
		for _, mmluCategory := range v {
			if config.IsSupportedRoutingDomainName(mmluCategory) {
				continue
			}
			c.addError(
				s.Pos,
				"SIGNAL domain %q has unsupported mmlu_categories value %q (supported: %s)%s",
				s.Name,
				mmluCategory,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				compileDomainSuggestionSuffix(mmluCategory),
			)
			return
		}
		cat.MMLUCategories = v
	}
	c.config.Categories = append(c.config.Categories, cat)
}

func (c *Compiler) compileFactCheckSignal(s *SignalDecl) {
	rule := config.FactCheckRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.FactCheckRules = append(c.config.FactCheckRules, rule)
}

func (c *Compiler) compileUserFeedbackSignal(s *SignalDecl) {
	rule := config.UserFeedbackRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.UserFeedbackRules = append(c.config.UserFeedbackRules, rule)
}

func (c *Compiler) compileReaskSignal(s *SignalDecl) {
	rule := config.ReaskRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
	}
	if v, ok := getIntField(s.Fields, "lookback_turns"); ok {
		rule.LookbackTurns = v
	}
	c.config.ReaskRules = append(c.config.ReaskRules, rule)
}

func (c *Compiler) compilePreferenceSignal(s *SignalDecl) {
	rule := config.PreferenceRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	if v, ok := getStringArrayField(s.Fields, "examples"); ok {
		rule.Examples = v
	}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
	}
	c.config.PreferenceRules = append(c.config.PreferenceRules, rule)
}

func (c *Compiler) compileLanguageSignal(s *SignalDecl) {
	rule := config.LanguageRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.LanguageRules = append(c.config.LanguageRules, rule)
}

func (c *Compiler) compileContextSignal(s *SignalDecl) {
	rule := config.ContextRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "min_tokens"); ok {
		rule.MinTokens = config.TokenCount(v)
	}
	if v, ok := getStringField(s.Fields, "max_tokens"); ok {
		rule.MaxTokens = config.TokenCount(v)
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.ContextRules = append(c.config.ContextRules, rule)
}

func (c *Compiler) compileStructureSignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name

	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode structure signal %q: %v", s.Name, err)
		return
	}

	var rule config.StructureRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode structure signal %q: %v", s.Name, err)
		return
	}
	rule.Name = s.Name
	if err := config.ValidateStructureRuleContract(rule); err != nil {
		c.addError(s.Pos, "%v", err)
		return
	}
	c.config.StructureRules = append(c.config.StructureRules, rule)
}

func (c *Compiler) compileComplexitySignal(s *SignalDecl) {
	rule := config.ComplexityRule{Name: s.Name}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	if obj, ok := s.Fields["composer"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			rc := compileComposerObj(ov)
			rule.Composer = &rc
		}
	}
	if obj, ok := s.Fields["hard"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			if candidates, ok := getStringArrayField(ov.Fields, "candidates"); ok {
				rule.Hard = config.ComplexityCandidates{Candidates: candidates}
			}
		}
	}
	if obj, ok := s.Fields["easy"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			if candidates, ok := getStringArrayField(ov.Fields, "candidates"); ok {
				rule.Easy = config.ComplexityCandidates{Candidates: candidates}
			}
		}
	}
	c.config.ComplexityRules = append(c.config.ComplexityRules, rule)
}

func (c *Compiler) compileModalitySignal(s *SignalDecl) {
	rule := config.ModalityRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.ModalityRules = append(c.config.ModalityRules, rule)
}

func (c *Compiler) compileSessionSignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name

	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode session signal %q: %v", s.Name, err)
		return
	}

	var rule config.SessionRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode session signal %q: %v", s.Name, err)
		return
	}
	c.config.SessionRules = append(c.config.SessionRules, rule)
}

func (c *Compiler) compileJailbreakSignal(s *SignalDecl) {
	rule := config.JailbreakRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "method"); ok {
		rule.Method = v
	}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
	}
	if v, ok := getBoolField(s.Fields, "include_history"); ok {
		rule.IncludeHistory = v
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	if v, ok := getStringArrayField(s.Fields, "jailbreak_patterns"); ok {
		rule.JailbreakPatterns = v
	}
	if v, ok := getStringArrayField(s.Fields, "benign_patterns"); ok {
		rule.BenignPatterns = v
	}
	c.config.JailbreakRules = append(c.config.JailbreakRules, rule)
}

func (c *Compiler) compilePIISignal(s *SignalDecl) {
	rule := config.PIIRule{Name: s.Name}
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
	}
	if v, ok := getStringArrayField(s.Fields, "pii_types_allowed"); ok {
		rule.PIITypesAllowed = v
	}
	if v, ok := getBoolField(s.Fields, "include_history"); ok {
		rule.IncludeHistory = v
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.PIIRules = append(c.config.PIIRules, rule)
}

func (c *Compiler) compileKBSignal(s *SignalDecl) {
	rule := config.KBSignalRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "kb"); ok {
		rule.KB = v
	}
	if target, ok := s.Fields["target"].(ObjectValue); ok {
		if kind, ok := getStringField(target.Fields, "kind"); ok {
			rule.Target.Kind = kind
		}
		if value, ok := getStringField(target.Fields, "value"); ok {
			rule.Target.Value = value
		}
	}
	if v, ok := getStringField(s.Fields, "match"); ok {
		rule.Match = v
	}
	c.config.KBRules = append(c.config.KBRules, rule)
}

//nolint:gocognit,nestif // Legacy schema decoding stays localized until the authz signal surface is extracted.
func (c *Compiler) compileAuthzSignal(s *SignalDecl) {
	rb := config.RoleBinding{Name: s.Name}
	if v, ok := getStringField(s.Fields, "role"); ok {
		rb.Role = v
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rb.Description = v
	}
	if arr, ok := s.Fields["subjects"]; ok {
		if av, ok := arr.(ArrayValue); ok {
			for _, item := range av.Items {
				if obj, ok := item.(ObjectValue); ok {
					subj := config.Subject{}
					if kind, ok := getStringField(obj.Fields, "kind"); ok {
						subj.Kind = kind
					}
					if name, ok := getStringField(obj.Fields, "name"); ok {
						subj.Name = name
					}
					rb.Subjects = append(rb.Subjects, subj)
				}
			}
		}
	}
	c.config.RoleBindings = append(c.config.RoleBindings, rb)
}

// ---------- Routes (Decisions) ----------

//nolint:gocognit,cyclop // Route lowering owns the DSL-to-config translation in one place.
func (c *Compiler) compileRoutes() {
	for _, r := range c.prog.Routes {
		decision := config.Decision{
			Name:        r.Name,
			Description: r.Description,
			Priority:    r.Priority,
			Tier:        r.Tier,
		}

		// Compile WHEN expression → RuleNode tree.
		// Python CLI requires rules.operator + rules.conditions at top level,
		// so we always ensure the top-level node is a combination (AND/OR/NOT),
		// never a bare leaf or empty struct.
		if r.When != nil {
			rules := c.compileBoolExpr(r.When)
			// If compileBoolExpr returned a leaf node (single signal ref),
			// wrap it in AND([leaf]) so Python validation passes.
			if rules.Operator == "" && rules.Type != "" {
				rules = config.RuleCombination{
					Operator:   "AND",
					Conditions: []config.RuleNode{rules},
				}
			}
			decision.Rules = rules
		} else {
			// No WHEN clause → match-all. Use AND with empty conditions.
			decision.Rules = config.RuleCombination{
				Operator:   "AND",
				Conditions: []config.RuleNode{},
			}
		}

		// Compile MODEL list
		for _, m := range r.Models {
			ref := config.ModelRef{
				Model:    m.Model,
				LoRAName: m.LoRA,
				Weight:   m.Weight,
			}
			if m.Reasoning != nil {
				ref.UseReasoning = m.Reasoning
			}
			if m.Effort != "" {
				ref.ReasoningEffort = m.Effort
			}
			decision.ModelRefs = append(decision.ModelRefs, ref)

			// Populate model_config for route-local model metadata fields.
			if m.ParamSize != "" {
				if c.config.ModelConfig == nil {
					c.config.ModelConfig = make(map[string]config.ModelParams)
				}
				mc := c.config.ModelConfig[m.Model]
				if m.ParamSize != "" {
					mc.ParamSize = m.ParamSize
				}
				c.config.ModelConfig[m.Model] = mc
			}
		}

		// Compile ALGORITHM
		if r.Algorithm != nil {
			decision.Algorithm = c.compileAlgorithm(r.Algorithm)
		}

		// Compile PLUGINs
		for _, pr := range r.Plugins {
			dp := c.compilePluginRef(pr)
			if dp != nil {
				decision.Plugins = append(decision.Plugins, *dp)
			}
		}

		c.config.Decisions = append(c.config.Decisions, decision)
	}
}

func (c *Compiler) compileBoolExpr(expr BoolExpr) config.RuleCombination {
	switch e := expr.(type) {
	case *BoolAnd:
		// Flatten nested ANDs: (a AND b) AND c → AND(a, b, c)
		conditions := c.flattenBoolExpr(expr, func(ex BoolExpr) bool {
			_, ok := ex.(*BoolAnd)
			return ok
		}, func(ex BoolExpr) (BoolExpr, BoolExpr) {
			a := ex.(*BoolAnd)
			return a.Left, a.Right
		})
		return config.RuleCombination{
			Operator:   "AND",
			Conditions: conditions,
		}
	case *BoolOr:
		// Flatten nested ORs: (a OR b) OR c → OR(a, b, c)
		conditions := c.flattenBoolExpr(expr, func(ex BoolExpr) bool {
			_, ok := ex.(*BoolOr)
			return ok
		}, func(ex BoolExpr) (BoolExpr, BoolExpr) {
			o := ex.(*BoolOr)
			return o.Left, o.Right
		})
		return config.RuleCombination{
			Operator:   "OR",
			Conditions: conditions,
		}
	case *BoolNot:
		return config.RuleCombination{
			Operator: "NOT",
			Conditions: []config.RuleNode{
				c.compileBoolExpr(e.Expr),
			},
		}
	case *SignalRefExpr:
		return config.RuleCombination{
			Type: e.SignalType,
			Name: e.SignalName,
		}
	default:
		c.addError(Position{}, "unknown bool expression type %T", expr)
		return config.RuleCombination{}
	}
}

// flattenBoolExpr collects all leaves of same-type nested binary expressions into a flat list.
func (c *Compiler) flattenBoolExpr(
	expr BoolExpr,
	isSameType func(BoolExpr) bool,
	getChildren func(BoolExpr) (BoolExpr, BoolExpr),
) []config.RuleNode {
	if isSameType(expr) {
		left, right := getChildren(expr)
		var result []config.RuleNode
		result = append(result, c.flattenBoolExpr(left, isSameType, getChildren)...)
		result = append(result, c.flattenBoolExpr(right, isSameType, getChildren)...)
		return result
	}
	return []config.RuleNode{c.compileBoolExpr(expr)}
}

//nolint:cyclop // The supported algorithm surface is intentionally enumerated here.
func (c *Compiler) compileAlgorithm(spec *AlgoSpec) *config.AlgorithmConfig {
	algo := &config.AlgorithmConfig{
		Type: spec.AlgoType,
	}

	switch spec.AlgoType {
	case "confidence":
		algo.Confidence = c.compileConfidenceAlgo(spec.Fields)
	case "ratings":
		algo.Ratings = c.compileRatingsAlgo(spec.Fields)
	case "remom":
		algo.ReMoM = c.compileReMoMAlgo(spec.Fields)
	case "elo":
		algo.Elo = c.compileEloAlgo(spec.Fields)
	case "router_dc":
		algo.RouterDC = c.compileRouterDCAlgo(spec.Fields)
	case "automix":
		algo.AutoMix = c.compileAutoMixAlgo(spec.Fields)
	case "hybrid":
		algo.Hybrid = c.compileHybridAlgo(spec.Fields)
	case "session_aware":
		algo.SessionAware = c.compileSessionAwareAlgo(spec.Fields)
	case "rl_driven":
		algo.RLDriven = c.compileRLDrivenAlgo(spec.Fields)
	case "gmtrouter":
		algo.GMTRouter = c.compileGMTRouterAlgo(spec.Fields)
	case "latency_aware":
		algo.LatencyAware = c.compileLatencyAwareAlgo(spec.Fields)
	case "static", "knn", "kmeans", "svm":
		// These types have no sub-config or use model_selection
	default:
		c.addError(spec.Pos, "unknown algorithm type %q", spec.AlgoType)
	}

	// Set on_error at the top level only for algorithm types that don't carry
	// their own on_error in a sub-config (confidence, ratings, remom already
	// set on_error inside their respective sub-struct).
	switch spec.AlgoType {
	case "confidence", "ratings", "remom":
		// on_error is handled in the sub-config
	default:
		if v, ok := getStringField(spec.Fields, "on_error"); ok {
			algo.OnError = v
		}
	}

	return algo
}

//nolint:nestif // Nested object decoding mirrors the DSL object shape.
func (c *Compiler) compileConfidenceAlgo(fields map[string]Value) *config.ConfidenceAlgorithmConfig {
	cfg := &config.ConfidenceAlgorithmConfig{}
	if v, ok := getStringField(fields, "confidence_method"); ok {
		cfg.ConfidenceMethod = v
	}
	if v, ok := getFloat64Field(fields, "threshold"); ok {
		cfg.Threshold = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
	if v, ok := getStringField(fields, "escalation_order"); ok {
		cfg.EscalationOrder = v
	}
	if v, ok := getFloat64Field(fields, "cost_quality_tradeoff"); ok {
		cfg.CostQualityTradeoff = v
	}
	if obj, ok := fields["hybrid_weights"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			hw := &config.HybridWeightsConfig{}
			if v, ok := getFloat64Field(ov.Fields, "logprob_weight"); ok {
				hw.LogprobWeight = v
			}
			if v, ok := getFloat64Field(ov.Fields, "margin_weight"); ok {
				hw.MarginWeight = v
			}
			cfg.HybridWeights = hw
		}
	}
	return cfg
}

func (c *Compiler) compileRatingsAlgo(fields map[string]Value) *config.RatingsAlgorithmConfig {
	cfg := &config.RatingsAlgorithmConfig{}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
	return cfg
}

func (c *Compiler) compileReMoMAlgo(fields map[string]Value) *config.ReMoMAlgorithmConfig {
	cfg := &config.ReMoMAlgorithmConfig{}
	if v, ok := getIntArrayField(fields, "breadth_schedule"); ok {
		cfg.BreadthSchedule = v
	}
	if v, ok := getStringField(fields, "model_distribution"); ok {
		cfg.ModelDistribution = v
	}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = v
	}
	if v, ok := getBoolField(fields, "include_reasoning"); ok {
		cfg.IncludeReasoning = v
	}
	if v, ok := getStringField(fields, "compaction_strategy"); ok {
		cfg.CompactionStrategy = v
	}
	if v, ok := getIntField(fields, "compaction_tokens"); ok {
		cfg.CompactionTokens = v
	}
	if v, ok := getStringField(fields, "synthesis_template"); ok {
		cfg.SynthesisTemplate = v
	}
	if v, ok := getIntField(fields, "max_concurrent"); ok {
		cfg.MaxConcurrent = v
	}
	if v, ok := getStringField(fields, "on_error"); ok {
		cfg.OnError = v
	}
	if v, ok := getBoolField(fields, "include_intermediate_responses"); ok {
		cfg.IncludeIntermediateResponses = v
	}
	return cfg
}

func (c *Compiler) compileEloAlgo(fields map[string]Value) *config.EloSelectionConfig {
	cfg := &config.EloSelectionConfig{}
	if v, ok := getFloat64Field(fields, "initial_rating"); ok {
		cfg.InitialRating = v
	}
	if v, ok := getFloat64Field(fields, "k_factor"); ok {
		cfg.KFactor = v
	}
	if v, ok := getBoolField(fields, "category_weighted"); ok {
		cfg.CategoryWeighted = v
	}
	if v, ok := getFloat64Field(fields, "decay_factor"); ok {
		cfg.DecayFactor = v
	}
	if v, ok := getIntField(fields, "min_comparisons"); ok {
		cfg.MinComparisons = v
	}
	if v, ok := getFloat64Field(fields, "cost_scaling_factor"); ok {
		cfg.CostScalingFactor = v
	}
	if v, ok := getStringField(fields, "storage_path"); ok {
		cfg.StoragePath = v
	}
	return cfg
}

func (c *Compiler) compileRouterDCAlgo(fields map[string]Value) *config.RouterDCSelectionConfig {
	cfg := &config.RouterDCSelectionConfig{}
	if v, ok := getFloat64Field(fields, "temperature"); ok {
		cfg.Temperature = v
	}
	if v, ok := getIntField(fields, "dimension_size"); ok {
		cfg.DimensionSize = v
	}
	if v, ok := getFloat64Field(fields, "min_similarity"); ok {
		cfg.MinSimilarity = v
	}
	if v, ok := getBoolField(fields, "use_query_contrastive"); ok {
		cfg.UseQueryContrastive = v
	}
	if v, ok := getBoolField(fields, "use_model_contrastive"); ok {
		cfg.UseModelContrastive = v
	}
	return cfg
}

func (c *Compiler) compileAutoMixAlgo(fields map[string]Value) *config.AutoMixSelectionConfig {
	cfg := &config.AutoMixSelectionConfig{}
	if v, ok := getFloat64Field(fields, "verification_threshold"); ok {
		cfg.VerificationThreshold = v
	}
	if v, ok := getIntField(fields, "max_escalations"); ok {
		cfg.MaxEscalations = v
	}
	if v, ok := getBoolField(fields, "cost_aware_routing"); ok {
		cfg.CostAwareRouting = v
	}
	return cfg
}

func (c *Compiler) compileHybridAlgo(fields map[string]Value) *config.HybridSelectionConfig {
	cfg := &config.HybridSelectionConfig{}
	if v, ok := getFloat64Field(fields, "elo_weight"); ok {
		cfg.EloWeight = v
	}
	if v, ok := getFloat64Field(fields, "router_dc_weight"); ok {
		cfg.RouterDCWeight = v
	}
	if v, ok := getFloat64Field(fields, "automix_weight"); ok {
		cfg.AutoMixWeight = v
	}
	if v, ok := getFloat64Field(fields, "cost_weight"); ok {
		cfg.CostWeight = v
	}
	return cfg
}

func (c *Compiler) compileSessionAwareAlgo(fields map[string]Value) *config.SessionAwareSelectionConfig {
	cfg := &config.SessionAwareSelectionConfig{}
	if v, ok := getStringField(fields, "fallback_method"); ok {
		cfg.FallbackMethod = v
	}
	if v, ok := getIntField(fields, "min_turns_before_switch"); ok {
		cfg.MinTurnsBeforeSwitch = v
	}
	if v, ok := getFloat64Field(fields, "stay_bias"); ok {
		cfg.StayBias = v
	}
	if v, ok := getFloat64Field(fields, "quality_gap_multiplier"); ok {
		cfg.QualityGapMultiplier = v
	}
	if v, ok := getFloat64Field(fields, "handoff_penalty_weight"); ok {
		cfg.HandoffPenaltyWeight = v
	}
	if v, ok := getFloat64Field(fields, "remaining_turn_weight"); ok {
		cfg.RemainingTurnWeight = v
	}
	return cfg
}

func (c *Compiler) compileRLDrivenAlgo(fields map[string]Value) *config.RLDrivenSelectionConfig {
	cfg := &config.RLDrivenSelectionConfig{}
	if v, ok := getFloat64Field(fields, "exploration_rate"); ok {
		cfg.ExplorationRate = v
	}
	if v, ok := getBoolField(fields, "use_thompson_sampling"); ok {
		cfg.UseThompsonSampling = v
	}
	if v, ok := getBoolField(fields, "enable_personalization"); ok {
		cfg.EnablePersonalization = v
	}
	return cfg
}

func (c *Compiler) compileGMTRouterAlgo(fields map[string]Value) *config.GMTRouterSelectionConfig {
	cfg := &config.GMTRouterSelectionConfig{}
	if v, ok := getBoolField(fields, "enable_personalization"); ok {
		cfg.EnablePersonalization = v
	}
	if v, ok := getIntField(fields, "history_sample_size"); ok {
		cfg.HistorySampleSize = v
	}
	if v, ok := getStringField(fields, "model_path"); ok {
		cfg.ModelPath = v
	}
	return cfg
}

func (c *Compiler) compileLatencyAwareAlgo(fields map[string]Value) *config.LatencyAwareAlgorithmConfig {
	cfg := &config.LatencyAwareAlgorithmConfig{}
	if v, ok := getIntField(fields, "tpot_percentile"); ok {
		cfg.TPOTPercentile = v
	}
	if v, ok := getIntField(fields, "ttft_percentile"); ok {
		cfg.TTFTPercentile = v
	}
	return cfg
}

// ---------- Plugin Ref Resolution ----------

func (c *Compiler) compilePluginRef(ref *PluginRef) *config.DecisionPlugin {
	// Check if it's a template reference
	if tmpl, ok := c.pluginTemplates[ref.Name]; ok {
		// Merge template fields with override fields
		mergedFields := make(map[string]Value)
		for k, v := range tmpl.Fields {
			mergedFields[k] = v
		}
		if ref.Fields != nil {
			for k, v := range ref.Fields {
				mergedFields[k] = v
			}
		}
		return c.buildDecisionPlugin(tmpl.PluginType, mergedFields)
	}

	// Inline plugin — ref.Name is the plugin type
	fields := ref.Fields
	if fields == nil {
		fields = make(map[string]Value)
	}
	return c.buildDecisionPlugin(ref.Name, fields)
}

//nolint:gocognit,cyclop,funlen // Plugin decoding remains centralized so type-specific payload wiring stays consistent.
func (c *Compiler) buildDecisionPlugin(pluginType string, fields map[string]Value) *config.DecisionPlugin {
	dp := &config.DecisionPlugin{Type: pluginType}
	setPluginConfig := func(value interface{}) {
		payload, err := config.NewStructuredPayload(value)
		if err != nil {
			c.addError(Position{}, "failed to encode plugin %q configuration: %v", pluginType, err)
			return
		}
		dp.Configuration = payload
	}

	switch pluginType {
	case "system_prompt":
		cfg := config.SystemPromptPluginConfig{}
		if v, ok := getStringField(fields, "system_prompt"); ok {
			cfg.SystemPrompt = v
		}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = &v
		}
		if v, ok := getStringField(fields, "mode"); ok {
			cfg.Mode = v
		}
		setPluginConfig(cfg)

	case "semantic_cache", "semantic-cache":
		dp.Type = "semantic-cache" // normalize to config convention
		cfg := config.SemanticCachePluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
			cfg.SimilarityThreshold = &v
		}
		setPluginConfig(cfg)

	case "hallucination":
		cfg := config.HallucinationPluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getBoolField(fields, "use_nli"); ok {
			cfg.UseNLI = v
		}
		if v, ok := getStringField(fields, "hallucination_action"); ok {
			cfg.HallucinationAction = v
		}
		setPluginConfig(cfg)

	case "memory":
		cfg := config.MemoryPluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getIntField(fields, "retrieval_limit"); ok {
			cfg.RetrievalLimit = &v
		}
		if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
			cfg.SimilarityThreshold = &v
		}
		if v, ok := getBoolField(fields, "auto_store"); ok {
			cfg.AutoStore = &v
		}
		setPluginConfig(cfg)

	case "rag":
		cfg := c.compileRAGPlugin(fields)
		setPluginConfig(cfg)

	case "header_mutation":
		cfg := config.HeaderMutationPluginConfig{}
		setPluginConfig(cfg)

	case "router_replay":
		cfg := config.RouterReplayPluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getIntField(fields, "max_records"); ok {
			cfg.MaxRecords = v
		}
		if v, ok := getBoolField(fields, "capture_request_body"); ok {
			cfg.CaptureRequestBody = v
		}
		if v, ok := getBoolField(fields, "capture_response_body"); ok {
			cfg.CaptureResponseBody = v
		}
		if v, ok := getIntField(fields, "max_body_bytes"); ok {
			cfg.MaxBodyBytes = v
		}
		setPluginConfig(cfg)

	case "image_gen":
		cfg := config.ImageGenPluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getStringField(fields, "backend"); ok {
			cfg.Backend = v
		}
		setPluginConfig(cfg)

	case "fast_response":
		cfg := config.FastResponsePluginConfig{}
		if v, ok := getStringField(fields, "message"); ok {
			cfg.Message = v
		}
		setPluginConfig(cfg)

	case "request_params":
		cfg := config.RequestParamsPluginConfig{}
		if v, ok := getStringArrayField(fields, "blocked_params"); ok {
			cfg.BlockedParams = v
		}
		if v, ok := getIntField(fields, "max_tokens_limit"); ok {
			cfg.MaxTokensLimit = &v
		}
		if v, ok := getIntField(fields, "max_n"); ok {
			cfg.MaxN = &v
		}
		if v, ok := getBoolField(fields, "strip_unknown"); ok {
			cfg.StripUnknown = v
		}
		setPluginConfig(cfg)

	case "tools":
		cfg := config.ToolsPluginConfig{}
		if v, ok := getBoolField(fields, "enabled"); ok {
			cfg.Enabled = v
		}
		if v, ok := getStringField(fields, "mode"); ok {
			cfg.Mode = v
		}
		if v, ok := getBoolField(fields, "semantic_selection"); ok {
			cfg.SemanticSelection = &v
		}
		if values, ok := getStringArrayField(fields, "allow_tools"); ok {
			cfg.AllowTools = values
		}
		if values, ok := getStringArrayField(fields, "block_tools"); ok {
			cfg.BlockTools = values
		}
		setPluginConfig(cfg)

	default:
		c.addError(Position{}, "unknown plugin type %q", pluginType)
		return nil
	}

	return dp
}

//nolint:nestif // Nested backend_config decoding mirrors the DSL object shape.
func (c *Compiler) compileRAGPlugin(fields map[string]Value) config.RAGPluginConfig {
	cfg := config.RAGPluginConfig{}
	if v, ok := getBoolField(fields, "enabled"); ok {
		cfg.Enabled = v
	}
	if v, ok := getStringField(fields, "backend"); ok {
		cfg.Backend = v
	}
	if v, ok := getIntField(fields, "top_k"); ok {
		cfg.TopK = &v
	}
	if v, ok := getFloat32Field(fields, "similarity_threshold"); ok {
		cfg.SimilarityThreshold = &v
	}
	if v, ok := getStringField(fields, "injection_mode"); ok {
		cfg.InjectionMode = v
	}
	if v, ok := getStringField(fields, "on_failure"); ok {
		cfg.OnFailure = v
	}
	// backend_config is a nested object
	if obj, ok := fields["backend_config"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			payload, err := config.NewStructuredPayload(fieldsToMap(ov.Fields))
			if err != nil {
				c.addError(Position{}, "failed to encode rag backend_config: %v", err)
			} else {
				cfg.BackendConfig = payload
			}
		}
	}
	return cfg
}

// compileComposerObj converts an ObjectValue representing a composer (RuleCombination)
// back into a config.RuleCombination. Handles both leaf nodes { type, name } and
// composite nodes { operator, conditions: [...] }.
func compileComposerObj(ov ObjectValue) config.RuleCombination {
	rc := config.RuleCombination{}
	if t, ok := getStringField(ov.Fields, "type"); ok {
		rc.Type = t
	}
	if n, ok := getStringField(ov.Fields, "name"); ok {
		rc.Name = n
	}
	if op, ok := getStringField(ov.Fields, "operator"); ok {
		rc.Operator = op
	}
	if arr, ok := ov.Fields["conditions"]; ok {
		if av, ok := arr.(ArrayValue); ok {
			for _, item := range av.Items {
				if obj, ok := item.(ObjectValue); ok {
					rc.Conditions = append(rc.Conditions, compileComposerObj(obj))
				}
			}
		}
	}
	return rc
}

// ---------- Error helpers ----------

func (c *Compiler) addError(pos Position, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	c.errors = append(c.errors, fmt.Errorf("%s: %s", pos, msg))
}

// ---------- Field extraction helpers ----------

func getStringField(fields map[string]Value, key string) (string, bool) {
	if v, ok := fields[key]; ok {
		if sv, ok := v.(StringValue); ok {
			return sv.V, true
		}
	}
	return "", false
}

func getIntField(fields map[string]Value, key string) (int, bool) {
	if v, ok := fields[key]; ok {
		if iv, ok := v.(IntValue); ok {
			return iv.V, true
		}
		// Also accept float as int
		if fv, ok := v.(FloatValue); ok {
			return int(fv.V), true
		}
	}
	return 0, false
}

func getFloat32Field(fields map[string]Value, key string) (float32, bool) {
	if v, ok := fields[key]; ok {
		if fv, ok := v.(FloatValue); ok {
			return float32(fv.V), true
		}
		if iv, ok := v.(IntValue); ok {
			return float32(iv.V), true
		}
	}
	return 0, false
}

func getFloat64Field(fields map[string]Value, key string) (float64, bool) {
	if v, ok := fields[key]; ok {
		if fv, ok := v.(FloatValue); ok {
			return fv.V, true
		}
		if iv, ok := v.(IntValue); ok {
			return float64(iv.V), true
		}
	}
	return 0, false
}

func getBoolField(fields map[string]Value, key string) (bool, bool) {
	if v, ok := fields[key]; ok {
		if bv, ok := v.(BoolValue); ok {
			return bv.V, true
		}
	}
	return false, false
}

func getStringArrayField(fields map[string]Value, key string) ([]string, bool) {
	if v, ok := fields[key]; ok {
		if av, ok := v.(ArrayValue); ok {
			var result []string
			for _, item := range av.Items {
				if sv, ok := item.(StringValue); ok {
					result = append(result, sv.V)
				}
			}
			return result, true
		}
	}
	return nil, false
}

func compileDomainSuggestionSuffix(value string) string {
	suggestion := config.SuggestSupportedRoutingDomainName(value)
	if suggestion == "" || suggestion == value {
		return ""
	}
	return fmt.Sprintf("; did you mean %q?", suggestion)
}

func (c *Compiler) validateSoftmaxDomainProjectionPartition(
	partition *ProjectionPartitionDecl,
) {
	if partition.Semantics != "softmax_exclusive" {
		return
	}
	for _, member := range partition.Members {
		signal := c.findSignalDeclByName(member)
		if signal == nil || signal.SignalType != "domain" {
			continue
		}
		mmluCategories := getMMLUCategories(signal)
		if len(mmluCategories) == 0 {
			if config.IsSupportedRoutingDomainName(signal.Name) {
				continue
			}
			c.addError(
				partition.Pos,
				"PROJECTION partition %q member %q must use a supported routing domain name (%s) or declare mmlu_categories explicitly%s",
				partition.Name,
				member,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				compileDomainSuggestionSuffix(member),
			)
			continue
		}
		for _, value := range mmluCategories {
			if config.IsSupportedRoutingDomainName(value) {
				continue
			}
			c.addError(
				partition.Pos,
				"PROJECTION partition %q member %q has unsupported mmlu_categories value %q (supported: %s)%s",
				partition.Name,
				member,
				value,
				strings.Join(config.SupportedRoutingDomainNames(), ", "),
				compileDomainSuggestionSuffix(value),
			)
		}
	}
}

func (c *Compiler) findSignalDeclByName(name string) *SignalDecl {
	for _, signal := range c.prog.Signals {
		if signal.Name == name {
			return signal
		}
	}
	return nil
}

func getIntArrayField(fields map[string]Value, key string) ([]int, bool) {
	if v, ok := fields[key]; ok {
		if av, ok := v.(ArrayValue); ok {
			var result []int
			for _, item := range av.Items {
				if iv, ok := item.(IntValue); ok {
					result = append(result, iv.V)
				}
			}
			return result, true
		}
	}
	return nil, false
}

func fieldsToMap(fields map[string]Value) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range fields {
		result[k] = valueToInterface(v)
	}
	return result
}

func valueToInterface(v Value) interface{} {
	switch val := v.(type) {
	case StringValue:
		return val.V
	case IntValue:
		return val.V
	case FloatValue:
		return val.V
	case BoolValue:
		return val.V
	case ArrayValue:
		var arr []interface{}
		for _, item := range val.Items {
			arr = append(arr, valueToInterface(item))
		}
		return arr
	case ObjectValue:
		return fieldsToMap(val.Fields)
	default:
		return nil
	}
}
