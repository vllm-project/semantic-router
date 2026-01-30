package mcpserver

import (
	"fmt"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func normalizeKind(kind string) string {
	k := strings.ToLower(strings.TrimSpace(kind))
	k = strings.ReplaceAll(k, "-", "_")
	return k
}

func listNames(kind string, cfg *config.RouterConfig) ([]string, error) {
	switch normalizeKind(kind) {
	case KindDecision:
		n := make([]string, 0, len(cfg.Decisions))
		for _, d := range cfg.Decisions {
			if d.Name != "" {
				n = append(n, d.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindKeywordRule:
		n := make([]string, 0, len(cfg.KeywordRules))
		for _, r := range cfg.KeywordRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindEmbeddingRule:
		n := make([]string, 0, len(cfg.EmbeddingRules))
		for _, r := range cfg.EmbeddingRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindFactCheckRule:
		n := make([]string, 0, len(cfg.FactCheckRules))
		for _, r := range cfg.FactCheckRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindUserFeedbackRule:
		n := make([]string, 0, len(cfg.UserFeedbackRules))
		for _, r := range cfg.UserFeedbackRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindPreferenceRule:
		n := make([]string, 0, len(cfg.PreferenceRules))
		for _, r := range cfg.PreferenceRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindLanguageRule:
		n := make([]string, 0, len(cfg.LanguageRules))
		for _, r := range cfg.LanguageRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindContextRule:
		n := make([]string, 0, len(cfg.ContextRules))
		for _, r := range cfg.ContextRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindLatencyRule:
		n := make([]string, 0, len(cfg.LatencyRules))
		for _, r := range cfg.LatencyRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindComplexityRule:
		n := make([]string, 0, len(cfg.ComplexityRules))
		for _, r := range cfg.ComplexityRules {
			if r.Name != "" {
				n = append(n, r.Name)
			}
		}
		sort.Strings(n)
		return n, nil
	case KindModel:
		n := make([]string, 0, len(cfg.ModelConfig))
		for name := range cfg.ModelConfig {
			n = append(n, name)
		}
		sort.Strings(n)
		return n, nil
	default:
		return nil, fmt.Errorf("unsupported kind %q", kind)
	}
}

func getEntityYAML(kind, name string, cfg *config.RouterConfig) (any, error) {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil, fmt.Errorf("name is required")
	}

	switch normalizeKind(kind) {
	case KindDecision:
		for _, d := range cfg.Decisions {
			if d.Name == name {
				return d, nil
			}
		}
		return nil, fmt.Errorf("decision %q not found", name)
	case KindKeywordRule:
		for _, r := range cfg.KeywordRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("keyword_rule %q not found", name)
	case KindEmbeddingRule:
		for _, r := range cfg.EmbeddingRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("embedding_rule %q not found", name)
	case KindFactCheckRule:
		for _, r := range cfg.FactCheckRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("fact_check_rule %q not found", name)
	case KindUserFeedbackRule:
		for _, r := range cfg.UserFeedbackRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("user_feedback_rule %q not found", name)
	case KindPreferenceRule:
		for _, r := range cfg.PreferenceRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("preference_rule %q not found", name)
	case KindLanguageRule:
		for _, r := range cfg.LanguageRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("language_rule %q not found", name)
	case KindContextRule:
		for _, r := range cfg.ContextRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("context_rule %q not found", name)
	case KindLatencyRule:
		for _, r := range cfg.LatencyRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("latency_rule %q not found", name)
	case KindComplexityRule:
		for _, r := range cfg.ComplexityRules {
			if r.Name == name {
				return r, nil
			}
		}
		return nil, fmt.Errorf("complexity_rule %q not found", name)
	case KindModel:
		if cfg.ModelConfig == nil {
			return nil, fmt.Errorf("model_config is nil")
		}
		mp, ok := cfg.ModelConfig[name]
		if !ok {
			return nil, fmt.Errorf("model %q not found", name)
		}
		return mp, nil
	default:
		return nil, fmt.Errorf("unsupported kind %q", kind)
	}
}

func addEntity(kind string, cfg *config.RouterConfig, itemYAML string) (string, error) {
	switch normalizeKind(kind) {
	case KindDecision:
		item, err := parseYAMLStrict[config.Decision](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("decision.name is required")
		}
		for _, d := range cfg.Decisions {
			if d.Name == item.Name {
				return "", fmt.Errorf("decision %q already exists", item.Name)
			}
		}
		cfg.Decisions = append(cfg.Decisions, *item)
		return item.Name, nil
	case KindKeywordRule:
		item, err := parseYAMLStrict[config.KeywordRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("keyword_rule.name is required")
		}
		for _, r := range cfg.KeywordRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("keyword_rule %q already exists", item.Name)
			}
		}
		cfg.KeywordRules = append(cfg.KeywordRules, *item)
		return item.Name, nil
	case KindEmbeddingRule:
		item, err := parseYAMLStrict[config.EmbeddingRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("embedding_rule.name is required")
		}
		for _, r := range cfg.EmbeddingRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("embedding_rule %q already exists", item.Name)
			}
		}
		cfg.EmbeddingRules = append(cfg.EmbeddingRules, *item)
		return item.Name, nil
	case KindFactCheckRule:
		item, err := parseYAMLStrict[config.FactCheckRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("fact_check_rule.name is required")
		}
		for _, r := range cfg.FactCheckRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("fact_check_rule %q already exists", item.Name)
			}
		}
		cfg.FactCheckRules = append(cfg.FactCheckRules, *item)
		return item.Name, nil
	case KindUserFeedbackRule:
		item, err := parseYAMLStrict[config.UserFeedbackRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("user_feedback_rule.name is required")
		}
		for _, r := range cfg.UserFeedbackRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("user_feedback_rule %q already exists", item.Name)
			}
		}
		cfg.UserFeedbackRules = append(cfg.UserFeedbackRules, *item)
		return item.Name, nil
	case KindPreferenceRule:
		item, err := parseYAMLStrict[config.PreferenceRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("preference_rule.name is required")
		}
		for _, r := range cfg.PreferenceRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("preference_rule %q already exists", item.Name)
			}
		}
		cfg.PreferenceRules = append(cfg.PreferenceRules, *item)
		return item.Name, nil
	case KindLanguageRule:
		item, err := parseYAMLStrict[config.LanguageRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("language_rule.name is required")
		}
		for _, r := range cfg.LanguageRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("language_rule %q already exists", item.Name)
			}
		}
		cfg.LanguageRules = append(cfg.LanguageRules, *item)
		return item.Name, nil
	case KindContextRule:
		item, err := parseYAMLStrict[config.ContextRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("context_rule.name is required")
		}
		for _, r := range cfg.ContextRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("context_rule %q already exists", item.Name)
			}
		}
		cfg.ContextRules = append(cfg.ContextRules, *item)
		return item.Name, nil
	case KindLatencyRule:
		item, err := parseYAMLStrict[config.LatencyRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("latency_rule.name is required")
		}
		for _, r := range cfg.LatencyRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("latency_rule %q already exists", item.Name)
			}
		}
		cfg.LatencyRules = append(cfg.LatencyRules, *item)
		return item.Name, nil
	case KindComplexityRule:
		item, err := parseYAMLStrict[config.ComplexityRule](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("complexity_rule.name is required")
		}
		for _, r := range cfg.ComplexityRules {
			if r.Name == item.Name {
				return "", fmt.Errorf("complexity_rule %q already exists", item.Name)
			}
		}
		cfg.ComplexityRules = append(cfg.ComplexityRules, *item)
		return item.Name, nil
	case KindModel:
		// For models, the payload is a small object with name + params.
		type modelPayload struct {
			Name   string             `yaml:"name"`
			Params config.ModelParams `yaml:"params"`
		}
		item, err := parseYAMLStrict[modelPayload](itemYAML)
		if err != nil {
			return "", err
		}
		if item.Name == "" {
			return "", fmt.Errorf("model.name is required")
		}
		if cfg.ModelConfig == nil {
			cfg.ModelConfig = map[string]config.ModelParams{}
		}
		if _, ok := cfg.ModelConfig[item.Name]; ok {
			return "", fmt.Errorf("model %q already exists", item.Name)
		}
		cfg.ModelConfig[item.Name] = item.Params
		return item.Name, nil
	default:
		return "", fmt.Errorf("unsupported kind %q", kind)
	}
}

func updateEntity(kind, name string, cfg *config.RouterConfig, itemYAML string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("name is required")
	}

	switch normalizeKind(kind) {
	case KindDecision:
		item, err := parseYAMLStrict[config.Decision](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("decision.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.Decisions {
			if cfg.Decisions[i].Name == name {
				cfg.Decisions[i] = *item
				return nil
			}
		}
		return fmt.Errorf("decision %q not found", name)
	case KindKeywordRule:
		item, err := parseYAMLStrict[config.KeywordRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("keyword_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.KeywordRules {
			if cfg.KeywordRules[i].Name == name {
				cfg.KeywordRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("keyword_rule %q not found", name)
	case KindEmbeddingRule:
		item, err := parseYAMLStrict[config.EmbeddingRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("embedding_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.EmbeddingRules {
			if cfg.EmbeddingRules[i].Name == name {
				cfg.EmbeddingRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("embedding_rule %q not found", name)
	case KindFactCheckRule:
		item, err := parseYAMLStrict[config.FactCheckRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("fact_check_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.FactCheckRules {
			if cfg.FactCheckRules[i].Name == name {
				cfg.FactCheckRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("fact_check_rule %q not found", name)
	case KindUserFeedbackRule:
		item, err := parseYAMLStrict[config.UserFeedbackRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("user_feedback_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.UserFeedbackRules {
			if cfg.UserFeedbackRules[i].Name == name {
				cfg.UserFeedbackRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("user_feedback_rule %q not found", name)
	case KindPreferenceRule:
		item, err := parseYAMLStrict[config.PreferenceRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("preference_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.PreferenceRules {
			if cfg.PreferenceRules[i].Name == name {
				cfg.PreferenceRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("preference_rule %q not found", name)
	case KindLanguageRule:
		item, err := parseYAMLStrict[config.LanguageRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("language_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.LanguageRules {
			if cfg.LanguageRules[i].Name == name {
				cfg.LanguageRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("language_rule %q not found", name)
	case KindContextRule:
		item, err := parseYAMLStrict[config.ContextRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("context_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.ContextRules {
			if cfg.ContextRules[i].Name == name {
				cfg.ContextRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("context_rule %q not found", name)
	case KindLatencyRule:
		item, err := parseYAMLStrict[config.LatencyRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("latency_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.LatencyRules {
			if cfg.LatencyRules[i].Name == name {
				cfg.LatencyRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("latency_rule %q not found", name)
	case KindComplexityRule:
		item, err := parseYAMLStrict[config.ComplexityRule](itemYAML)
		if err != nil {
			return err
		}
		if item.Name != "" && item.Name != name {
			return fmt.Errorf("complexity_rule.name must match path name (%q)", name)
		}
		item.Name = name
		for i := range cfg.ComplexityRules {
			if cfg.ComplexityRules[i].Name == name {
				cfg.ComplexityRules[i] = *item
				return nil
			}
		}
		return fmt.Errorf("complexity_rule %q not found", name)
	case KindModel:
		item, err := parseYAMLStrict[config.ModelParams](itemYAML)
		if err != nil {
			return err
		}
		if cfg.ModelConfig == nil {
			cfg.ModelConfig = map[string]config.ModelParams{}
		}
		if _, ok := cfg.ModelConfig[name]; !ok {
			return fmt.Errorf("model %q not found", name)
		}
		cfg.ModelConfig[name] = *item
		return nil
	default:
		return fmt.Errorf("unsupported kind %q", kind)
	}
}

func deleteEntity(kind, name string, cfg *config.RouterConfig) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("name is required")
	}

	switch normalizeKind(kind) {
	case KindDecision:
		out := cfg.Decisions[:0]
		found := false
		for _, d := range cfg.Decisions {
			if d.Name == name {
				found = true
				continue
			}
			out = append(out, d)
		}
		cfg.Decisions = out
		if !found {
			return fmt.Errorf("decision %q not found", name)
		}
		return nil
	case KindKeywordRule:
		out := cfg.KeywordRules[:0]
		found := false
		for _, r := range cfg.KeywordRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.KeywordRules = out
		if !found {
			return fmt.Errorf("keyword_rule %q not found", name)
		}
		return nil
	case KindEmbeddingRule:
		out := cfg.EmbeddingRules[:0]
		found := false
		for _, r := range cfg.EmbeddingRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.EmbeddingRules = out
		if !found {
			return fmt.Errorf("embedding_rule %q not found", name)
		}
		return nil
	case KindFactCheckRule:
		out := cfg.FactCheckRules[:0]
		found := false
		for _, r := range cfg.FactCheckRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.FactCheckRules = out
		if !found {
			return fmt.Errorf("fact_check_rule %q not found", name)
		}
		return nil
	case KindUserFeedbackRule:
		out := cfg.UserFeedbackRules[:0]
		found := false
		for _, r := range cfg.UserFeedbackRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.UserFeedbackRules = out
		if !found {
			return fmt.Errorf("user_feedback_rule %q not found", name)
		}
		return nil
	case KindPreferenceRule:
		out := cfg.PreferenceRules[:0]
		found := false
		for _, r := range cfg.PreferenceRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.PreferenceRules = out
		if !found {
			return fmt.Errorf("preference_rule %q not found", name)
		}
		return nil
	case KindLanguageRule:
		out := cfg.LanguageRules[:0]
		found := false
		for _, r := range cfg.LanguageRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.LanguageRules = out
		if !found {
			return fmt.Errorf("language_rule %q not found", name)
		}
		return nil
	case KindContextRule:
		out := cfg.ContextRules[:0]
		found := false
		for _, r := range cfg.ContextRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.ContextRules = out
		if !found {
			return fmt.Errorf("context_rule %q not found", name)
		}
		return nil
	case KindLatencyRule:
		out := cfg.LatencyRules[:0]
		found := false
		for _, r := range cfg.LatencyRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.LatencyRules = out
		if !found {
			return fmt.Errorf("latency_rule %q not found", name)
		}
		return nil
	case KindComplexityRule:
		out := cfg.ComplexityRules[:0]
		found := false
		for _, r := range cfg.ComplexityRules {
			if r.Name == name {
				found = true
				continue
			}
			out = append(out, r)
		}
		cfg.ComplexityRules = out
		if !found {
			return fmt.Errorf("complexity_rule %q not found", name)
		}
		return nil
	case KindModel:
		if cfg.ModelConfig == nil {
			return fmt.Errorf("model_config is nil")
		}
		if _, ok := cfg.ModelConfig[name]; !ok {
			return fmt.Errorf("model %q not found", name)
		}
		delete(cfg.ModelConfig, name)
		return nil
	default:
		return fmt.Errorf("unsupported kind %q", kind)
	}
}
