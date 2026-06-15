package dsl

import (
	"strings"

	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Compiler) compileSignals() {
	for _, s := range c.prog.Signals {
		fn, ok := signalCompilerByType[s.SignalType]
		if !ok {
			c.addError(s.Pos, "unknown signal type %q", s.SignalType)
			continue
		}
		fn(c, s)
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
	if v, ok := getStringField(s.Fields, "query_modality"); ok {
		rule.QueryModality = config.QueryModality(v)
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
	if scores, ok := getModelScoresField(s.Fields, "model_scores"); ok {
		cat.ModelScores = scores
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
	if v, ok := getFloat32Field(s.Fields, "threshold"); ok {
		rule.Threshold = v
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
			rule.Hard = compileComplexityCandidates(ov.Fields)
		}
	}
	if obj, ok := s.Fields["easy"]; ok {
		if ov, ok := obj.(ObjectValue); ok {
			rule.Easy = compileComplexityCandidates(ov.Fields)
		}
	}
	c.config.ComplexityRules = append(c.config.ComplexityRules, rule)
}

func compileComplexityCandidates(fields map[string]Value) config.ComplexityCandidates {
	var candidates config.ComplexityCandidates
	if values, ok := getStringArrayField(fields, "candidates"); ok {
		candidates.Candidates = values
	}
	if values, ok := getStringArrayField(fields, "image_candidates"); ok {
		candidates.ImageCandidates = values
	}
	return candidates
}

func (c *Compiler) compileModalitySignal(s *SignalDecl) {
	rule := config.ModalityRule{Name: s.Name}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rule.Description = v
	}
	c.config.ModalityRules = append(c.config.ModalityRules, rule)
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

func (c *Compiler) compileConversationSignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name

	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode conversation signal %q: %v", s.Name, err)
		return
	}

	var rule config.ConversationRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode conversation signal %q: %v", s.Name, err)
		return
	}
	rule.Name = s.Name
	if err := config.ValidateConversationRuleContract(rule); err != nil {
		c.addError(s.Pos, "%v", err)
		return
	}
	c.config.ConversationRules = append(c.config.ConversationRules, rule)
}

func (c *Compiler) compileEventSignal(s *SignalDecl) {
	rule := config.EventRule{Name: s.Name}
	if v, ok := getStringArrayField(s.Fields, "event_types"); ok {
		rule.EventTypes = v
	}
	if v, ok := getStringArrayField(s.Fields, "severities"); ok {
		rule.Severities = v
	}
	if v, ok := getStringArrayField(s.Fields, "action_codes"); ok {
		rule.ActionCodes = v
	}
	if v, ok := getBoolField(s.Fields, "temporal"); ok {
		rule.Temporal = v
	}
	c.config.EventRules = append(c.config.EventRules, rule)
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

func (c *Compiler) compileAuthzSignal(s *SignalDecl) {
	rb := config.RoleBinding{Name: s.Name}
	if v, ok := getStringField(s.Fields, "role"); ok {
		rb.Role = v
	}
	if v, ok := getStringField(s.Fields, "description"); ok {
		rb.Description = v
	}
	if arr, ok := s.Fields["subjects"]; ok {
		rb.Subjects = parseAuthzSubjects(arr)
	}
	c.config.RoleBindings = append(c.config.RoleBindings, rb)
}

func parseAuthzSubjects(v Value) []config.Subject {
	arr, ok := v.(ArrayValue)
	if !ok {
		return nil
	}
	subjects := make([]config.Subject, 0, len(arr.Items))
	for _, item := range arr.Items {
		obj, ok := item.(ObjectValue)
		if !ok {
			continue
		}
		subj := config.Subject{}
		if kind, ok := getStringField(obj.Fields, "kind"); ok {
			subj.Kind = kind
		}
		if name, ok := getStringField(obj.Fields, "name"); ok {
			subj.Name = name
		}
		subjects = append(subjects, subj)
	}
	return subjects
}
