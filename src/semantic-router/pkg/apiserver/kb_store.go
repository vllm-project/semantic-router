//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	knowledgeBaseRuntimeRoot     = "knowledge_bases"
	knowledgeBaseManifestName    = "labels.json"
	knowledgeBaseManifestVersion = "1.0.0"
	knowledgeBaseDocumentType    = "embedding_kb"
)

var knowledgeBaseNamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type knowledgeBaseLabelPayload struct {
	Name        string   `json:"name" yaml:"name"`
	Description string   `json:"description,omitempty" yaml:"description,omitempty"`
	Exemplars   []string `json:"exemplars" yaml:"exemplars"`
}

type knowledgeBaseSignalReference struct {
	Name   string                `json:"name" yaml:"name"`
	Target config.KBSignalTarget `json:"target" yaml:"target"`
	Match  string                `json:"match,omitempty" yaml:"match,omitempty"`
}

type knowledgeBaseBindOptions struct {
	Labels  []string `json:"labels" yaml:"labels"`
	Groups  []string `json:"groups" yaml:"groups"`
	Metrics []string `json:"metrics" yaml:"metrics"`
}

type knowledgeBaseDocument struct {
	Name             string                             `json:"name" yaml:"name"`
	Type             string                             `json:"type" yaml:"type"`
	Builtin          bool                               `json:"builtin" yaml:"builtin"`
	Managed          bool                               `json:"managed" yaml:"managed"`
	Editable         bool                               `json:"editable" yaml:"editable"`
	Threshold        float32                            `json:"threshold" yaml:"threshold"`
	LabelThresholds  map[string]float32                 `json:"label_thresholds,omitempty" yaml:"label_thresholds,omitempty"`
	Description      string                             `json:"description,omitempty" yaml:"description,omitempty"`
	Source           config.KnowledgeBaseSource         `json:"source" yaml:"source"`
	Labels           []knowledgeBaseLabelPayload        `json:"labels,omitempty" yaml:"labels,omitempty"`
	Groups           map[string][]string                `json:"groups,omitempty" yaml:"groups,omitempty"`
	Metrics          []config.KnowledgeBaseMetricConfig `json:"metrics,omitempty" yaml:"metrics,omitempty"`
	SignalReferences []knowledgeBaseSignalReference     `json:"signal_references,omitempty" yaml:"signal_references,omitempty"`
	BindOptions      knowledgeBaseBindOptions           `json:"bind_options" yaml:"bind_options"`
	LoadError        string                             `json:"load_error,omitempty" yaml:"load_error,omitempty"`
}

type knowledgeBaseListResponse struct {
	Items []knowledgeBaseDocument `json:"items" yaml:"items"`
}

type knowledgeBaseUpsertRequest struct {
	Name            string                             `json:"name,omitempty" yaml:"name,omitempty"`
	Threshold       float32                            `json:"threshold" yaml:"threshold"`
	LabelThresholds map[string]float32                 `json:"label_thresholds,omitempty" yaml:"label_thresholds,omitempty"`
	Description     string                             `json:"description,omitempty" yaml:"description,omitempty"`
	Labels          []knowledgeBaseLabelPayload        `json:"labels" yaml:"labels"`
	Groups          map[string][]string                `json:"groups,omitempty" yaml:"groups,omitempty"`
	Metrics         []config.KnowledgeBaseMetricConfig `json:"metrics,omitempty" yaml:"metrics,omitempty"`
}

type knowledgeBaseDeleteResponse struct {
	Status string `json:"status" yaml:"status"`
	Name   string `json:"name" yaml:"name"`
}

func normalizeKnowledgeBaseRequest(payload knowledgeBaseUpsertRequest) (knowledgeBaseUpsertRequest, error) {
	name, err := normalizeKnowledgeBaseName(payload.Name)
	if err != nil {
		return payload, err
	}
	if scoreErr := validateKnowledgeBaseScore("threshold", payload.Threshold); scoreErr != nil {
		return payload, scoreErr
	}
	labels, seenLabels, err := normalizeKnowledgeBaseLabels(payload.Labels)
	if err != nil {
		return payload, err
	}
	thresholds, err := normalizeKnowledgeBaseThresholds(payload.LabelThresholds, seenLabels)
	if err != nil {
		return payload, err
	}
	groups, err := normalizeKnowledgeBaseGroups(payload.Groups, seenLabels)
	if err != nil {
		return payload, err
	}
	metrics, err := normalizeKnowledgeBaseMetrics(payload.Metrics, groups)
	if err != nil {
		return payload, err
	}

	payload.Name = name
	payload.Labels = labels
	payload.LabelThresholds = thresholds
	payload.Groups = groups
	payload.Metrics = metrics
	payload.Description = strings.TrimSpace(payload.Description)
	return payload, nil
}

func normalizeKnowledgeBaseName(name string) (string, error) {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return "", fmt.Errorf("name is required")
	}
	if !knowledgeBaseNamePattern.MatchString(trimmed) {
		return "", fmt.Errorf("name %q contains invalid characters (allowed: letters, numbers, _, -)", trimmed)
	}
	return trimmed, nil
}

func validateKnowledgeBaseScore(fieldName string, value float32) error {
	if value < 0 || value > 1 {
		return fmt.Errorf("%s must be between 0 and 1", fieldName)
	}
	return nil
}

func normalizeKnowledgeBaseLabels(labels []knowledgeBaseLabelPayload) ([]knowledgeBaseLabelPayload, map[string]struct{}, error) {
	if len(labels) == 0 {
		return nil, nil, fmt.Errorf("at least one label is required")
	}

	seenLabels := make(map[string]struct{}, len(labels))
	normalized := make([]knowledgeBaseLabelPayload, 0, len(labels))
	for _, label := range labels {
		normalizedLabel, err := normalizeKnowledgeBaseLabel(label, seenLabels)
		if err != nil {
			return nil, nil, err
		}
		normalized = append(normalized, normalizedLabel)
	}
	sort.Slice(normalized, func(i, j int) bool { return normalized[i].Name < normalized[j].Name })
	return normalized, seenLabels, nil
}

func normalizeKnowledgeBaseLabel(label knowledgeBaseLabelPayload, seenLabels map[string]struct{}) (knowledgeBaseLabelPayload, error) {
	label.Name = strings.TrimSpace(label.Name)
	label.Description = strings.TrimSpace(label.Description)
	if label.Name == "" {
		return label, fmt.Errorf("label name cannot be empty")
	}
	if !knowledgeBaseNamePattern.MatchString(label.Name) {
		return label, fmt.Errorf("label %q contains invalid characters", label.Name)
	}
	if _, exists := seenLabels[label.Name]; exists {
		return label, fmt.Errorf("label %q is declared more than once", label.Name)
	}
	seenLabels[label.Name] = struct{}{}

	cleanExemplars, err := normalizeKnowledgeBaseExemplars(label.Name, label.Exemplars)
	if err != nil {
		return label, err
	}
	label.Exemplars = cleanExemplars
	return label, nil
}

func normalizeKnowledgeBaseExemplars(labelName string, exemplars []string) ([]string, error) {
	if len(exemplars) == 0 {
		return nil, fmt.Errorf("label %q must include at least one exemplar", labelName)
	}
	cleanExemplars := make([]string, 0, len(exemplars))
	for _, exemplar := range exemplars {
		trimmed := strings.TrimSpace(exemplar)
		if trimmed == "" {
			continue
		}
		cleanExemplars = append(cleanExemplars, trimmed)
	}
	if len(cleanExemplars) == 0 {
		return nil, fmt.Errorf("label %q must include at least one non-empty exemplar", labelName)
	}
	return cleanExemplars, nil
}

func normalizeKnowledgeBaseThresholds(labelThresholds map[string]float32, seenLabels map[string]struct{}) (map[string]float32, error) {
	cleanThresholds := make(map[string]float32, len(labelThresholds))
	for labelName, threshold := range labelThresholds {
		trimmed := strings.TrimSpace(labelName)
		if trimmed == "" {
			return nil, fmt.Errorf("label_thresholds cannot contain an empty label name")
		}
		if _, exists := seenLabels[trimmed]; !exists {
			return nil, fmt.Errorf("label_thresholds references unknown label %q", trimmed)
		}
		if err := validateKnowledgeBaseScore(fmt.Sprintf("label_thresholds[%q]", trimmed), threshold); err != nil {
			return nil, err
		}
		cleanThresholds[trimmed] = threshold
	}
	if len(cleanThresholds) == 0 {
		return nil, nil
	}
	return cleanThresholds, nil
}

func normalizeKnowledgeBaseGroups(groups map[string][]string, seenLabels map[string]struct{}) (map[string][]string, error) {
	if len(groups) == 0 {
		return nil, nil
	}

	cleanGroups := make(map[string][]string, len(groups))
	for groupName, labels := range groups {
		normalizedGroup, err := normalizeKnowledgeBaseGroup(groupName, labels, seenLabels)
		if err != nil {
			return nil, err
		}
		cleanGroups[strings.TrimSpace(groupName)] = normalizedGroup
	}
	return cleanGroups, nil
}

func normalizeKnowledgeBaseGroup(groupName string, labels []string, seenLabels map[string]struct{}) ([]string, error) {
	trimmedGroupName := strings.TrimSpace(groupName)
	if trimmedGroupName == "" {
		return nil, fmt.Errorf("groups cannot contain an empty group name")
	}
	if !knowledgeBaseNamePattern.MatchString(trimmedGroupName) {
		return nil, fmt.Errorf("group %q contains invalid characters", trimmedGroupName)
	}

	cleanLabels := make([]string, 0, len(labels))
	seenGroupLabels := make(map[string]struct{}, len(labels))
	for _, labelName := range labels {
		trimmedLabel := strings.TrimSpace(labelName)
		if trimmedLabel == "" {
			return nil, fmt.Errorf("groups[%q] cannot contain an empty label name", trimmedGroupName)
		}
		if _, exists := seenLabels[trimmedLabel]; !exists {
			return nil, fmt.Errorf("groups[%q] references unknown label %q", trimmedGroupName, trimmedLabel)
		}
		if _, exists := seenGroupLabels[trimmedLabel]; exists {
			continue
		}
		seenGroupLabels[trimmedLabel] = struct{}{}
		cleanLabels = append(cleanLabels, trimmedLabel)
	}
	if len(cleanLabels) == 0 {
		return nil, fmt.Errorf("groups[%q] must include at least one label", trimmedGroupName)
	}
	sort.Strings(cleanLabels)
	return cleanLabels, nil
}

func normalizeKnowledgeBaseMetrics(metrics []config.KnowledgeBaseMetricConfig, groups map[string][]string) ([]config.KnowledgeBaseMetricConfig, error) {
	if len(metrics) == 0 {
		return nil, nil
	}

	seenMetrics := make(map[string]struct{}, len(metrics))
	normalized := make([]config.KnowledgeBaseMetricConfig, 0, len(metrics))
	for _, metric := range metrics {
		normalizedMetric, err := normalizeKnowledgeBaseMetric(metric, groups, seenMetrics)
		if err != nil {
			return nil, err
		}
		normalized = append(normalized, normalizedMetric)
	}
	sort.Slice(normalized, func(i, j int) bool { return normalized[i].Name < normalized[j].Name })
	return normalized, nil
}

func normalizeKnowledgeBaseMetric(metric config.KnowledgeBaseMetricConfig, groups map[string][]string, seenMetrics map[string]struct{}) (config.KnowledgeBaseMetricConfig, error) {
	metric.Name = strings.TrimSpace(metric.Name)
	metric.Type = strings.TrimSpace(metric.Type)
	metric.PositiveGroup = strings.TrimSpace(metric.PositiveGroup)
	metric.NegativeGroup = strings.TrimSpace(metric.NegativeGroup)
	if metric.Name == "" {
		return metric, fmt.Errorf("metric name cannot be empty")
	}
	if !knowledgeBaseNamePattern.MatchString(metric.Name) {
		return metric, fmt.Errorf("metric %q contains invalid characters", metric.Name)
	}
	if _, exists := seenMetrics[metric.Name]; exists {
		return metric, fmt.Errorf("metric %q is declared more than once", metric.Name)
	}
	seenMetrics[metric.Name] = struct{}{}
	if metric.Type != config.KBMetricTypeGroupMargin {
		return metric, fmt.Errorf("metric %q has unsupported type %q", metric.Name, metric.Type)
	}
	if metric.PositiveGroup == "" || metric.NegativeGroup == "" {
		return metric, fmt.Errorf("metric %q must declare positive_group and negative_group", metric.Name)
	}
	if _, exists := groups[metric.PositiveGroup]; !exists {
		return metric, fmt.Errorf("metric %q references unknown positive_group %q", metric.Name, metric.PositiveGroup)
	}
	if _, exists := groups[metric.NegativeGroup]; !exists {
		return metric, fmt.Errorf("metric %q references unknown negative_group %q", metric.Name, metric.NegativeGroup)
	}
	return metric, nil
}
