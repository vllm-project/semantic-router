package config

import (
	"fmt"
	"path/filepath"
	"strings"
)

func validateKnowledgeBaseContracts(cfg *RouterConfig) error {
	kbs, definitions, err := knowledgeBaseDefinitions(cfg)
	if err != nil {
		return err
	}

	signalNames := make(map[string]struct{}, len(cfg.KBRules))
	for _, rule := range cfg.KBRules {
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
		definition, hasDefinition := definitions[kb.Name]
		if !hasDefinition {
			continue
		}

		switch rule.Target.Kind {
		case KBTargetKindLabel:
			if _, ok := definition.Labels[rule.Target.Value]; !ok {
				return fmt.Errorf("routing.signals.kb[%q]: target.value %q is not a declared label for kb %q", rule.Name, rule.Target.Value, rule.KB)
			}
		case KBTargetKindGroup:
			if _, ok := kb.Groups[rule.Target.Value]; !ok {
				return fmt.Errorf("routing.signals.kb[%q]: target.value %q is not a declared group for kb %q", rule.Name, rule.Target.Value, rule.KB)
			}
		default:
			return fmt.Errorf("routing.signals.kb[%q]: target.kind %q is unsupported (supported: label, group)", rule.Name, rule.Target.Kind)
		}

		switch normalizedKBMatch(rule.Match) {
		case KBMatchBest, KBMatchThreshold:
		default:
			return fmt.Errorf("routing.signals.kb[%q]: match %q is unsupported (supported: best, threshold)", rule.Name, rule.Match)
		}
	}

	return nil
}

func knowledgeBaseDefinitions(cfg *RouterConfig) (map[string]KnowledgeBaseConfig, map[string]KnowledgeBaseDefinition, error) {
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
	labelNames := make(map[string]struct{}, len(definition.Labels))
	for name, label := range definition.Labels {
		if strings.TrimSpace(name) == "" {
			return fmt.Errorf("global.model_catalog.kbs[%q]: labels manifest cannot contain an empty label name", kb.Name)
		}
		if len(label.Exemplars) == 0 {
			return fmt.Errorf("global.model_catalog.kbs[%q]: label %q must include at least one exemplar", kb.Name, name)
		}
		labelNames[name] = struct{}{}
	}

	for labelName, threshold := range kb.LabelThresholds {
		if _, ok := labelNames[labelName]; !ok {
			return fmt.Errorf("global.model_catalog.kbs[%q]: label_thresholds references unknown label %q", kb.Name, labelName)
		}
		if threshold < 0 || threshold > 1 {
			return fmt.Errorf("global.model_catalog.kbs[%q]: label_thresholds[%q] must be between 0 and 1", kb.Name, labelName)
		}
	}

	for groupName, labels := range kb.Groups {
		if strings.TrimSpace(groupName) == "" {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups cannot contain an empty group name", kb.Name)
		}
		if len(labels) == 0 {
			return fmt.Errorf("global.model_catalog.kbs[%q]: groups[%q] must include at least one label", kb.Name, groupName)
		}
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
	}

	metricNames := map[string]struct{}{
		KBMetricBestScore:        {},
		KBMetricBestMatchedScore: {},
	}
	for _, metric := range kb.Metrics {
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
	}

	return nil
}

func normalizedKBMatch(value string) string {
	if strings.TrimSpace(value) == "" {
		return KBMatchThreshold
	}
	return strings.ToLower(strings.TrimSpace(value))
}
