package config

import (
	"fmt"
	"path/filepath"
	"strings"
)

func validateKnowledgeBaseContracts(cfg *RouterConfig) error {
	referenced := referencedKnowledgeBaseNames(cfg)
	kbs, definitions, err := knowledgeBaseDefinitions(cfg, referenced)
	if err != nil {
		return err
	}

	signalNames := make(map[string]struct{}, len(cfg.KBRules))
	for _, rule := range cfg.KBRules {
		if err := validateKnowledgeBaseSignalRule(rule, signalNames, kbs, definitions); err != nil {
			return err
		}
	}

	return nil
}

// referencedKnowledgeBaseNames returns the set of knowledge-base names that
// are actually referenced by a routing.signals.kb[] rule. Default KBs that
// nobody references can skip the on-disk manifest load below — see #1829.
func referencedKnowledgeBaseNames(cfg *RouterConfig) map[string]struct{} {
	out := make(map[string]struct{}, len(cfg.KBRules))
	for _, rule := range cfg.KBRules {
		if rule.KB != "" {
			out[rule.KB] = struct{}{}
		}
	}
	return out
}

func validateKnowledgeBaseSignalRule(
	rule KBSignalRule,
	signalNames map[string]struct{},
	kbs map[string]KnowledgeBaseConfig,
	definitions map[string]KnowledgeBaseDefinition,
) error {
	if rule.Name == "" {
		return fmt.Errorf("routing.signals.kb: name cannot be empty")
	}
	if _, exists := signalNames[rule.Name]; exists {
		return fmt.Errorf("routing.signals.kb[%q]: duplicate signal name", rule.Name)
	}
	signalNames[rule.Name] = struct{}{}

	kb, ok := kbs[rule.KB]
	if !ok {
		return fmt.Errorf("routing.signals.kb[%q]: kb %q is not declared in global.model_catalog.kbs", rule.Name, rule.KB)
	}
	if err := validateKnowledgeBaseSignalTarget(rule, kb, definitions[kb.Name]); err != nil {
		return err
	}

	switch normalizedKBMatch(rule.Match) {
	case KBMatchBest, KBMatchThreshold:
		return nil
	default:
		return fmt.Errorf("routing.signals.kb[%q]: match %q is unsupported (supported: best, threshold)", rule.Name, rule.Match)
	}
}

func validateKnowledgeBaseSignalTarget(rule KBSignalRule, kb KnowledgeBaseConfig, definition KnowledgeBaseDefinition) error {
	switch rule.Target.Kind {
	case KBTargetKindLabel:
		if len(definition.Labels) == 0 {
			return nil
		}
		if _, ok := definition.Labels[rule.Target.Value]; !ok {
			return fmt.Errorf("routing.signals.kb[%q]: target.value %q is not a declared label for kb %q", rule.Name, rule.Target.Value, rule.KB)
		}
		return nil
	case KBTargetKindGroup:
		if _, ok := kb.Groups[rule.Target.Value]; !ok {
			return fmt.Errorf("routing.signals.kb[%q]: target.value %q is not a declared group for kb %q", rule.Name, rule.Target.Value, rule.KB)
		}
		return nil
	default:
		return fmt.Errorf("routing.signals.kb[%q]: target.kind %q is unsupported (supported: label, group)", rule.Name, rule.Target.Kind)
	}
}

func knowledgeBaseDefinitions(cfg *RouterConfig, referenced map[string]struct{}) (map[string]KnowledgeBaseConfig, map[string]KnowledgeBaseDefinition, error) {
	kbs := make(map[string]KnowledgeBaseConfig, len(cfg.KnowledgeBases))
	definitions := make(map[string]KnowledgeBaseDefinition, len(cfg.KnowledgeBases))
	for _, kb := range cfg.KnowledgeBases {
		if kb.Name == "" {
			return nil, nil, fmt.Errorf("global.model_catalog.kbs: name cannot be empty")
		}
		if _, exists := kbs[kb.Name]; exists {
			return nil, nil, fmt.Errorf("global.model_catalog.kbs[%q]: duplicate kb name", kb.Name)
		}
		if kb.Source.Path == "" {
			return nil, nil, fmt.Errorf("global.model_catalog.kbs[%q]: source.path cannot be empty", kb.Name)
		}
		if kb.Threshold < 0 || kb.Threshold > 1 {
			return nil, nil, fmt.Errorf("global.model_catalog.kbs[%q]: threshold must be between 0 and 1", kb.Name)
		}
		if cfg.ConfigBaseDir == "" && !filepath.IsAbs(kb.Source.Path) {
			kbs[kb.Name] = kb
			continue
		}

		// #1829: defaults like privacy_kb and mmlu_kb get merged into every router
		// config via canonical defaults. Stock Helm/operator deployments without
		// bundled KB assets shouldn't fatal at startup just because nobody asked
		// the router to use those KBs. Only load the on-disk labels manifest for
		// KBs that a routing.signals.kb[] rule actually references; unreferenced
		// KBs keep the basic config validation above (name, source.path,
		// threshold) but skip the manifest read.
		if _, isReferenced := referenced[kb.Name]; !isReferenced {
			kbs[kb.Name] = kb
			continue
		}

		definition, err := LoadKnowledgeBaseDefinition(cfg.ConfigBaseDir, kb.Source)
		if err != nil {
			return nil, nil, fmt.Errorf("global.model_catalog.kbs[%q]: failed to load labels manifest: %w", kb.Name, err)
		}
		if err := validateKnowledgeBaseDefinition(kb, definition); err != nil {
			return nil, nil, err
		}
		kbs[kb.Name] = kb
		definitions[kb.Name] = definition
	}
	return kbs, definitions, nil
}

func validateKnowledgeBaseDefinition(kb KnowledgeBaseConfig, definition KnowledgeBaseDefinition) error {
	labelNames, err := collectKnowledgeBaseLabelNames(kb, definition)
	if err != nil {
		return err
	}
	if err := validateKnowledgeBaseLabelThresholds(kb, labelNames); err != nil {
		return err
	}
	if err := validateKnowledgeBaseGroups(kb, labelNames); err != nil {
		return err
	}
	return validateKnowledgeBaseMetrics(kb)
}

func collectKnowledgeBaseLabelNames(kb KnowledgeBaseConfig, definition KnowledgeBaseDefinition) (map[string]struct{}, error) {
	labelNames := make(map[string]struct{}, len(definition.Labels))
	for name, label := range definition.Labels {
		if strings.TrimSpace(name) == "" {
			return nil, fmt.Errorf("global.model_catalog.kbs[%q]: labels manifest cannot contain an empty label name", kb.Name)
		}
		if len(label.Exemplars) == 0 {
			return nil, fmt.Errorf("global.model_catalog.kbs[%q]: label %q must include at least one exemplar", kb.Name, name)
		}
		labelNames[name] = struct{}{}
	}
	return labelNames, nil
}

func validateKnowledgeBaseLabelThresholds(kb KnowledgeBaseConfig, labelNames map[string]struct{}) error {
	for labelName, threshold := range kb.LabelThresholds {
		if _, ok := labelNames[labelName]; !ok {
			return fmt.Errorf("global.model_catalog.kbs[%q]: label_thresholds references unknown label %q", kb.Name, labelName)
		}
		if threshold < 0 || threshold > 1 {
			return fmt.Errorf("global.model_catalog.kbs[%q]: label_thresholds[%q] must be between 0 and 1", kb.Name, labelName)
		}
	}
	return nil
}

func validateKnowledgeBaseGroups(kb KnowledgeBaseConfig, labelNames map[string]struct{}) error {
	for groupName, labels := range kb.Groups {
		if strings.TrimSpace(groupName) == "" {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups cannot contain an empty group name", kb.Name)
		}
		if len(labels) == 0 {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups[%q] must include at least one label", kb.Name, groupName)
		}
		if err := validateKnowledgeBaseGroupLabels(kb, groupName, labels, labelNames); err != nil {
			return err
		}
	}
	return nil
}

func validateKnowledgeBaseGroupLabels(kb KnowledgeBaseConfig, groupName string, labels []string, labelNames map[string]struct{}) error {
	seen := map[string]struct{}{}
	for _, label := range labels {
		if _, ok := labelNames[label]; !ok {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups[%q] references unknown label %q", kb.Name, groupName, label)
		}
		if _, exists := seen[label]; exists {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups[%q] references label %q more than once", kb.Name, groupName, label)
		}
		seen[label] = struct{}{}
	}
	return nil
}

func validateKnowledgeBaseMetrics(kb KnowledgeBaseConfig) error {
	metricNames := map[string]struct{}{
		KBMetricBestScore:        {},
		KBMetricBestMatchedScore: {},
	}
	for _, metric := range kb.Metrics {
		if err := validateKnowledgeBaseMetric(kb, metric, metricNames); err != nil {
			return err
		}
	}
	return nil
}

func validateKnowledgeBaseMetric(kb KnowledgeBaseConfig, metric KnowledgeBaseMetricConfig, metricNames map[string]struct{}) error {
	if strings.TrimSpace(metric.Name) == "" {
		return fmt.Errorf("global.model_catalog.kbs[%q]: metrics.name cannot be empty", kb.Name)
	}
	if _, exists := metricNames[metric.Name]; exists {
		return fmt.Errorf("global.model_catalog.kbs[%q]: duplicate metric name %q", kb.Name, metric.Name)
	}
	metricNames[metric.Name] = struct{}{}
	if metric.Type != KBMetricTypeGroupMargin {
		return fmt.Errorf("global.model_catalog.kbs[%q]: metrics[%q] has unsupported type %q (supported: group_margin)", kb.Name, metric.Name, metric.Type)
	}
	if _, ok := kb.Groups[metric.PositiveGroup]; !ok {
		return fmt.Errorf("global.model_catalog.kbs[%q]: metrics[%q] references unknown positive_group %q", kb.Name, metric.Name, metric.PositiveGroup)
	}
	if _, ok := kb.Groups[metric.NegativeGroup]; !ok {
		return fmt.Errorf("global.model_catalog.kbs[%q]: metrics[%q] references unknown negative_group %q", kb.Name, metric.Name, metric.NegativeGroup)
	}
	return nil
}

func normalizedKBMatch(value string) string {
	if strings.TrimSpace(value) == "" {
		return KBMatchThreshold
	}
	return strings.ToLower(strings.TrimSpace(value))
}
