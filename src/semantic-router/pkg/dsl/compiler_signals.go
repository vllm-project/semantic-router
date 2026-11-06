package dsl

import (
	"gopkg.in/yaml.v2"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func (c *Compiler) compileSignals() {
	for _, s := range c.prog.Signals {
		if signal, ok := applicationSignals[s.SignalType]; ok {
			signal.compile(c, s)
			continue
		}
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
	if v, ok := getBoolField(s.Fields, "include_history"); ok {
		rule.IncludeHistory = v
	}
	c.config.KeywordRules = append(c.config.KeywordRules, rule)
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

func (c *Compiler) compileMetadataSignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name
	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode metadata signal %q: %v", s.Name, err)
		return
	}
	var rule config.MetadataRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode metadata signal %q: %v", s.Name, err)
		return
	}
	c.config.MetadataRules = append(c.config.MetadataRules, rule)
}

func (c *Compiler) compileInputModalitySignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name
	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode input_modality signal %q: %v", s.Name, err)
		return
	}
	var rule config.InputModalityRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode input_modality signal %q: %v", s.Name, err)
		return
	}
	if err := config.ValidateInputModalityRuleContract(rule); err != nil {
		c.addError(s.Pos, "%v", err)
		return
	}
	c.config.InputModalityRules = append(c.config.InputModalityRules, rule)
}

func (c *Compiler) compileSystemOneSignal(s *SignalDecl) {
	payload := fieldsToMap(s.Fields)
	payload["name"] = s.Name
	raw, err := yaml.Marshal(payload)
	if err != nil {
		c.addError(s.Pos, "failed to encode systemone signal %q: %v", s.Name, err)
		return
	}
	var rule config.SystemOneRule
	if err := yaml.Unmarshal(raw, &rule); err != nil {
		c.addError(s.Pos, "failed to decode systemone signal %q: %v", s.Name, err)
		return
	}
	if err := config.ValidateSystemOneRuleContract(rule); err != nil {
		c.addError(s.Pos, "%v", err)
		return
	}
	c.config.SystemOneRules = append(c.config.SystemOneRules, rule)
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
