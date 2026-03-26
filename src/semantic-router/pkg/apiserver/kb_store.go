//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	knowledgeBaseCustomRoot      = "kbs/custom"
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

type managedKnowledgeBaseAssetsTxn struct {
	finalDir   string
	backupDir  string
	removeOnly bool
}

func normalizeKnowledgeBaseRequest(payload knowledgeBaseUpsertRequest) (knowledgeBaseUpsertRequest, error) {
	payload.Name = strings.TrimSpace(payload.Name)
	if payload.Name == "" {
		return payload, fmt.Errorf("name is required")
	}
	if !knowledgeBaseNamePattern.MatchString(payload.Name) {
		return payload, fmt.Errorf("name %q contains invalid characters (allowed: letters, numbers, _, -)", payload.Name)
	}
	if len(payload.Labels) == 0 {
		return payload, fmt.Errorf("at least one label is required")
	}

	seenLabels := make(map[string]struct{}, len(payload.Labels))
	normalizedLabels := make([]knowledgeBaseLabelPayload, 0, len(payload.Labels))
	for _, label := range payload.Labels {
		label.Name = strings.TrimSpace(label.Name)
		label.Description = strings.TrimSpace(label.Description)
		if label.Name == "" {
			return payload, fmt.Errorf("label name cannot be empty")
		}
		if !knowledgeBaseNamePattern.MatchString(label.Name) {
			return payload, fmt.Errorf("label %q contains invalid characters", label.Name)
		}
		if _, exists := seenLabels[label.Name]; exists {
			return payload, fmt.Errorf("label %q is declared more than once", label.Name)
		}
		seenLabels[label.Name] = struct{}{}
		if len(label.Exemplars) == 0 {
			return payload, fmt.Errorf("label %q must include at least one exemplar", label.Name)
		}
		cleanExemplars := make([]string, 0, len(label.Exemplars))
		for _, exemplar := range label.Exemplars {
			trimmed := strings.TrimSpace(exemplar)
			if trimmed == "" {
				continue
			}
			cleanExemplars = append(cleanExemplars, trimmed)
		}
		if len(cleanExemplars) == 0 {
			return payload, fmt.Errorf("label %q must include at least one non-empty exemplar", label.Name)
		}
		label.Exemplars = cleanExemplars
		normalizedLabels = append(normalizedLabels, label)
	}
	sort.Slice(normalizedLabels, func(i, j int) bool { return normalizedLabels[i].Name < normalizedLabels[j].Name })
	payload.Labels = normalizedLabels

	cleanThresholds := make(map[string]float32, len(payload.LabelThresholds))
	for labelName, threshold := range payload.LabelThresholds {
		trimmed := strings.TrimSpace(labelName)
		if trimmed == "" {
			return payload, fmt.Errorf("label_thresholds cannot contain an empty label name")
		}
		if _, exists := seenLabels[trimmed]; !exists {
			return payload, fmt.Errorf("label_thresholds references unknown label %q", trimmed)
		}
		cleanThresholds[trimmed] = threshold
	}
	if len(cleanThresholds) == 0 {
		payload.LabelThresholds = nil
	} else {
		payload.LabelThresholds = cleanThresholds
	}

	if len(payload.Groups) > 0 {
		cleanGroups := make(map[string][]string, len(payload.Groups))
		for groupName, labels := range payload.Groups {
			groupName = strings.TrimSpace(groupName)
			if groupName == "" {
				return payload, fmt.Errorf("groups cannot contain an empty group name")
			}
			if !knowledgeBaseNamePattern.MatchString(groupName) {
				return payload, fmt.Errorf("group %q contains invalid characters", groupName)
			}
			cleanLabels := make([]string, 0, len(labels))
			seenGroupLabels := make(map[string]struct{}, len(labels))
			for _, labelName := range labels {
				labelName = strings.TrimSpace(labelName)
				if _, exists := seenLabels[labelName]; !exists {
					return payload, fmt.Errorf("groups[%q] references unknown label %q", groupName, labelName)
				}
				if _, exists := seenGroupLabels[labelName]; exists {
					continue
				}
				seenGroupLabels[labelName] = struct{}{}
				cleanLabels = append(cleanLabels, labelName)
			}
			sort.Strings(cleanLabels)
			cleanGroups[groupName] = cleanLabels
		}
		payload.Groups = cleanGroups
	}

	if len(payload.Metrics) > 0 {
		seenMetrics := make(map[string]struct{}, len(payload.Metrics))
		normalizedMetrics := make([]config.KnowledgeBaseMetricConfig, 0, len(payload.Metrics))
		for _, metric := range payload.Metrics {
			metric.Name = strings.TrimSpace(metric.Name)
			metric.Type = strings.TrimSpace(metric.Type)
			metric.PositiveGroup = strings.TrimSpace(metric.PositiveGroup)
			metric.NegativeGroup = strings.TrimSpace(metric.NegativeGroup)
			if metric.Name == "" {
				return payload, fmt.Errorf("metric name cannot be empty")
			}
			if !knowledgeBaseNamePattern.MatchString(metric.Name) {
				return payload, fmt.Errorf("metric %q contains invalid characters", metric.Name)
			}
			if _, exists := seenMetrics[metric.Name]; exists {
				return payload, fmt.Errorf("metric %q is declared more than once", metric.Name)
			}
			seenMetrics[metric.Name] = struct{}{}
			if metric.Type != config.KBMetricTypeGroupMargin {
				return payload, fmt.Errorf("metric %q has unsupported type %q", metric.Name, metric.Type)
			}
			if metric.PositiveGroup == "" || metric.NegativeGroup == "" {
				return payload, fmt.Errorf("metric %q must declare positive_group and negative_group", metric.Name)
			}
			if _, exists := payload.Groups[metric.PositiveGroup]; !exists {
				return payload, fmt.Errorf("metric %q references unknown positive_group %q", metric.Name, metric.PositiveGroup)
			}
			if _, exists := payload.Groups[metric.NegativeGroup]; !exists {
				return payload, fmt.Errorf("metric %q references unknown negative_group %q", metric.Name, metric.NegativeGroup)
			}
			normalizedMetrics = append(normalizedMetrics, metric)
		}
		sort.Slice(normalizedMetrics, func(i, j int) bool { return normalizedMetrics[i].Name < normalizedMetrics[j].Name })
		payload.Metrics = normalizedMetrics
	}

	payload.Description = strings.TrimSpace(payload.Description)
	return payload, nil
}

func defaultKnowledgeBaseMap() map[string]config.KnowledgeBaseConfig {
	defaults := config.DefaultCanonicalGlobal().ModelCatalog.KBs
	result := make(map[string]config.KnowledgeBaseConfig, len(defaults))
	for _, kb := range defaults {
		result[kb.Name] = kb
	}
	return result
}

func isBuiltinKnowledgeBase(kb config.KnowledgeBaseConfig) bool {
	defaults := defaultKnowledgeBaseMap()
	defaultKB, ok := defaults[kb.Name]
	if !ok {
		return false
	}
	return cleanKnowledgeBaseSourcePath(defaultKB.Source.Path) == cleanKnowledgeBaseSourcePath(kb.Source.Path)
}

func isManagedKnowledgeBase(kb config.KnowledgeBaseConfig) bool {
	cleaned := cleanKnowledgeBaseSourcePath(kb.Source.Path)
	customRoot := cleanKnowledgeBaseSourcePath(knowledgeBaseCustomRoot)
	return cleaned == customRoot || strings.HasPrefix(cleaned, customRoot+"/")
}

func cleanKnowledgeBaseSourcePath(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return ""
	}
	cleaned := path.Clean(filepath.ToSlash(trimmed))
	if cleaned == "." {
		return ""
	}
	return strings.TrimSuffix(cleaned, "/")
}

func managedKnowledgeBaseSourcePath(name string) string {
	return filepath.ToSlash(filepath.Join(knowledgeBaseCustomRoot, name)) + "/"
}

func managedKnowledgeBaseDir(baseDir string, name string) string {
	return filepath.Join(baseDir, filepath.FromSlash(cleanKnowledgeBaseSourcePath(managedKnowledgeBaseSourcePath(name))))
}

func knowledgeBaseConfigBaseDir(cfg *config.RouterConfig, configPath string) string {
	if cfg != nil && strings.TrimSpace(cfg.ConfigBaseDir) != "" {
		return cfg.ConfigBaseDir
	}
	if configPath == "" {
		return ""
	}
	return filepath.Dir(resolveConfigPersistencePaths(configPath).sourcePath)
}

func listKnowledgeBaseDocuments(cfg *config.RouterConfig, baseDir string) ([]knowledgeBaseDocument, error) {
	documents := make([]knowledgeBaseDocument, 0, len(cfg.KnowledgeBases))
	for _, kb := range cfg.KnowledgeBases {
		document, err := buildKnowledgeBaseDocument(cfg, baseDir, kb)
		if err != nil {
			return nil, err
		}
		documents = append(documents, document)
	}
	sort.Slice(documents, func(i, j int) bool { return documents[i].Name < documents[j].Name })
	return documents, nil
}

func buildKnowledgeBaseDocument(cfg *config.RouterConfig, baseDir string, kb config.KnowledgeBaseConfig) (knowledgeBaseDocument, error) {
	document := knowledgeBaseDocument{
		Name:             kb.Name,
		Type:             knowledgeBaseDocumentType,
		Builtin:          isBuiltinKnowledgeBase(kb),
		Managed:          isManagedKnowledgeBase(kb),
		Editable:         existingKnowledgeBaseEditable(kb),
		Threshold:        kb.Threshold,
		LabelThresholds:  cloneLabelThresholds(kb.LabelThresholds),
		Source:           kb.Source,
		Groups:           cloneKnowledgeBaseGroups(kb.Groups),
		Metrics:          cloneKnowledgeBaseMetrics(kb.Metrics),
		SignalReferences: knowledgeBaseSignalReferences(cfg, kb.Name),
	}

	definition, err := config.LoadKnowledgeBaseDefinition(baseDir, kb.Source)
	if err != nil {
		document.LoadError = err.Error()
		return document, nil
	}

	document.Description = strings.TrimSpace(definition.Description)
	document.Labels = knowledgeBaseLabelPayloads(definition)
	document.BindOptions = knowledgeBaseBindOptions{
		Labels:  extractKnowledgeBaseLabelNames(document.Labels),
		Groups:  extractKnowledgeBaseGroupNames(document.Groups),
		Metrics: extractKnowledgeBaseMetricNames(document.Metrics),
	}
	return document, nil
}

func knowledgeBaseSignalReferences(cfg *config.RouterConfig, kbName string) []knowledgeBaseSignalReference {
	references := make([]knowledgeBaseSignalReference, 0)
	if cfg == nil {
		return references
	}
	for _, rule := range cfg.KBRules {
		if rule.KB != kbName {
			continue
		}
		references = append(references, knowledgeBaseSignalReference{
			Name:   rule.Name,
			Target: rule.Target,
			Match:  normalizeKnowledgeBaseMatch(rule.Match),
		})
	}
	sort.Slice(references, func(i, j int) bool { return references[i].Name < references[j].Name })
	return references
}

func knowledgeBaseLabelPayloads(definition config.KnowledgeBaseDefinition) []knowledgeBaseLabelPayload {
	labels := make([]knowledgeBaseLabelPayload, 0, len(definition.Labels))
	for name, label := range definition.Labels {
		labels = append(labels, knowledgeBaseLabelPayload{
			Name:        name,
			Description: strings.TrimSpace(label.Description),
			Exemplars:   append([]string(nil), label.Exemplars...),
		})
	}
	sort.Slice(labels, func(i, j int) bool { return labels[i].Name < labels[j].Name })
	return labels
}

func extractKnowledgeBaseLabelNames(labels []knowledgeBaseLabelPayload) []string {
	names := make([]string, 0, len(labels))
	for _, label := range labels {
		names = append(names, label.Name)
	}
	sort.Strings(names)
	return names
}

func extractKnowledgeBaseGroupNames(groups map[string][]string) []string {
	names := make([]string, 0, len(groups))
	for name := range groups {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

func extractKnowledgeBaseMetricNames(metrics []config.KnowledgeBaseMetricConfig) []string {
	names := []string{config.KBMetricBestScore, config.KBMetricBestMatchedScore}
	for _, metric := range metrics {
		names = append(names, metric.Name)
	}
	sort.Strings(names)
	return names
}

func cloneLabelThresholds(thresholds map[string]float32) map[string]float32 {
	if len(thresholds) == 0 {
		return nil
	}
	cloned := make(map[string]float32, len(thresholds))
	for key, value := range thresholds {
		cloned[key] = value
	}
	return cloned
}

func cloneKnowledgeBaseGroups(groups map[string][]string) map[string][]string {
	if len(groups) == 0 {
		return nil
	}
	cloned := make(map[string][]string, len(groups))
	for key, values := range groups {
		copyValues := append([]string(nil), values...)
		sort.Strings(copyValues)
		cloned[key] = copyValues
	}
	return cloned
}

func cloneKnowledgeBaseMetrics(metrics []config.KnowledgeBaseMetricConfig) []config.KnowledgeBaseMetricConfig {
	if len(metrics) == 0 {
		return nil
	}
	cloned := append([]config.KnowledgeBaseMetricConfig(nil), metrics...)
	sort.Slice(cloned, func(i, j int) bool { return cloned[i].Name < cloned[j].Name })
	return cloned
}

func desiredKnowledgeBases(current []config.KnowledgeBaseConfig, next config.KnowledgeBaseConfig) []config.KnowledgeBaseConfig {
	updated := make([]config.KnowledgeBaseConfig, 0, len(current)+1)
	replaced := false
	for _, kb := range current {
		if kb.Name == next.Name {
			updated = append(updated, next)
			replaced = true
			continue
		}
		updated = append(updated, kb)
	}
	if !replaced {
		updated = append(updated, next)
	}
	sort.Slice(updated, func(i, j int) bool { return updated[i].Name < updated[j].Name })
	return updated
}

func removeKnowledgeBase(current []config.KnowledgeBaseConfig, name string) []config.KnowledgeBaseConfig {
	filtered := make([]config.KnowledgeBaseConfig, 0, len(current))
	for _, kb := range current {
		if kb.Name == name {
			continue
		}
		filtered = append(filtered, kb)
	}
	sort.Slice(filtered, func(i, j int) bool { return filtered[i].Name < filtered[j].Name })
	return filtered
}

func knowledgeBaseOverrideYAML(existingData []byte, kbs []config.KnowledgeBaseConfig) ([]byte, error) {
	doc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil, err
	}

	if !shouldPersistKnowledgeBaseOverride(kbs) {
		deleteNestedMappingValue(root, "global", "model_catalog", "kbs")
		return marshalYAMLDocument(doc)
	}

	kbNode, err := yamlNodeFromValue(kbs)
	if err != nil {
		return nil, fmt.Errorf("encode kb override: %w", err)
	}
	setNestedMappingValue(root, kbNode, "global", "model_catalog", "kbs")
	return marshalYAMLDocument(doc)
}

func shouldPersistKnowledgeBaseOverride(kbs []config.KnowledgeBaseConfig) bool {
	defaults := normalizeKnowledgeBaseSlice(config.DefaultCanonicalGlobal().ModelCatalog.KBs)
	current := normalizeKnowledgeBaseSlice(kbs)
	return !reflect.DeepEqual(defaults, current)
}

func normalizeKnowledgeBaseSlice(kbs []config.KnowledgeBaseConfig) []config.KnowledgeBaseConfig {
	normalized := make([]config.KnowledgeBaseConfig, 0, len(kbs))
	for _, kb := range kbs {
		copyKB := kb
		copyKB.Source.Path = strings.TrimSpace(copyKB.Source.Path)
		copyKB.Source.Manifest = strings.TrimSpace(copyKB.Source.Manifest)
		copyKB.LabelThresholds = cloneLabelThresholds(copyKB.LabelThresholds)
		copyKB.Groups = cloneKnowledgeBaseGroups(copyKB.Groups)
		copyKB.Metrics = cloneKnowledgeBaseMetrics(copyKB.Metrics)
		normalized = append(normalized, copyKB)
	}
	sort.Slice(normalized, func(i, j int) bool { return normalized[i].Name < normalized[j].Name })
	return normalized
}

func validateConfigWithBaseDir(baseDir string, yamlBytes []byte) (*config.RouterConfig, error) {
	tempFile, err := os.CreateTemp(baseDir, ".kb-*.yaml")
	if err != nil {
		return nil, err
	}
	tempPath := tempFile.Name()
	defer func() {
		_ = tempFile.Close()
		_ = os.Remove(tempPath)
	}()
	if _, err := tempFile.Write(yamlBytes); err != nil {
		return nil, err
	}
	if err := tempFile.Close(); err != nil {
		return nil, err
	}
	return config.Parse(tempPath)
}

func persistConfigAndSync(
	s *ClassificationAPIServer,
	paths configPersistencePaths,
	previousData []byte,
	yamlBytes []byte,
	newCfg *config.RouterConfig,
) error {
	if err := writeConfigAtomically(paths.sourcePath, yamlBytes); err != nil {
		return err
	}
	if paths.usesRuntimeOverride() {
		if _, err := runtimeConfigSyncRunner(paths.sourcePath); err != nil {
			_ = writeConfigAtomically(paths.sourcePath, previousData)
			_, _ = runtimeConfigSyncRunner(paths.sourcePath)
			return err
		}
	}
	s.updateRuntimeConfig(newCfg)
	return nil
}

func stageManagedKnowledgeBaseAssets(baseDir string, payload knowledgeBaseUpsertRequest) (*managedKnowledgeBaseAssetsTxn, error) {
	finalDir := managedKnowledgeBaseDir(baseDir, payload.Name)
	parentDir := filepath.Dir(finalDir)
	if err := os.MkdirAll(parentDir, 0o755); err != nil {
		return nil, err
	}

	stageDir := finalDir + ".tmp-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, err
	}

	if err := writeKnowledgeBaseAssets(stageDir, payload); err != nil {
		_ = os.RemoveAll(stageDir)
		return nil, err
	}

	txn := &managedKnowledgeBaseAssetsTxn{finalDir: finalDir}
	if _, err := os.Stat(finalDir); err == nil {
		txn.backupDir = finalDir + ".bak-" + fmt.Sprintf("%d", time.Now().UnixNano())
		if err := os.Rename(finalDir, txn.backupDir); err != nil {
			_ = os.RemoveAll(stageDir)
			return nil, err
		}
	}

	if err := os.Rename(stageDir, finalDir); err != nil {
		if txn.backupDir != "" {
			_ = os.Rename(txn.backupDir, finalDir)
		}
		_ = os.RemoveAll(stageDir)
		return nil, err
	}
	return txn, nil
}

func stageManagedKnowledgeBaseRemoval(baseDir string, name string) (*managedKnowledgeBaseAssetsTxn, error) {
	finalDir := managedKnowledgeBaseDir(baseDir, name)
	if _, err := os.Stat(finalDir); os.IsNotExist(err) {
		return nil, nil
	}
	backupDir := finalDir + ".bak-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.Rename(finalDir, backupDir); err != nil {
		return nil, err
	}
	return &managedKnowledgeBaseAssetsTxn{
		finalDir:   finalDir,
		backupDir:  backupDir,
		removeOnly: true,
	}, nil
}

func (txn *managedKnowledgeBaseAssetsTxn) Commit() {
	if txn == nil || txn.backupDir == "" {
		return
	}
	_ = os.RemoveAll(txn.backupDir)
}

func (txn *managedKnowledgeBaseAssetsTxn) Rollback() {
	if txn == nil {
		return
	}
	if txn.removeOnly {
		if txn.backupDir != "" {
			_ = os.RemoveAll(txn.finalDir)
			_ = os.Rename(txn.backupDir, txn.finalDir)
		}
		return
	}
	_ = os.RemoveAll(txn.finalDir)
	if txn.backupDir != "" {
		_ = os.Rename(txn.backupDir, txn.finalDir)
	}
}

func writeKnowledgeBaseAssets(root string, payload knowledgeBaseUpsertRequest) error {
	definition := config.KnowledgeBaseDefinition{
		Version:     knowledgeBaseManifestVersion,
		Description: payload.Description,
		Labels:      make(map[string]config.KnowledgeBaseLabelDef, len(payload.Labels)),
	}
	for _, label := range payload.Labels {
		definition.Labels[label.Name] = config.KnowledgeBaseLabelDef{
			Description: label.Description,
			Exemplars:   append([]string(nil), label.Exemplars...),
		}
	}

	definitionBytes, err := json.MarshalIndent(definition, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(root, knowledgeBaseManifestName), definitionBytes, 0o644)
}

func normalizeKnowledgeBaseMatch(value string) string {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "", config.KBMatchBest:
		return config.KBMatchBest
	case config.KBMatchThreshold:
		return config.KBMatchThreshold
	default:
		return strings.ToLower(strings.TrimSpace(value))
	}
}

func yamlNodeFromValue(value any) (*yaml.Node, error) {
	data, err := yaml.Marshal(value)
	if err != nil {
		return nil, err
	}
	doc, err := parseYAMLDocument(data)
	if err != nil {
		return nil, err
	}
	root, err := documentMappingNode(doc)
	if err == nil {
		return root, nil
	}
	if len(doc.Content) > 0 {
		return doc.Content[0], nil
	}
	return nil, fmt.Errorf("failed to decode YAML node")
}

func parseYAMLDocument(data []byte) (*yaml.Node, error) {
	var doc yaml.Node
	if err := yaml.Unmarshal(data, &doc); err != nil {
		return nil, err
	}
	if len(doc.Content) == 0 {
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	return &doc, nil
}

func documentMappingNode(doc *yaml.Node) (*yaml.Node, error) {
	if doc == nil {
		return nil, fmt.Errorf("yaml document is nil")
	}
	if doc.Kind == 0 && len(doc.Content) == 0 {
		doc.Kind = yaml.DocumentNode
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	if doc.Kind != yaml.DocumentNode {
		return nil, fmt.Errorf("expected YAML document node, got kind %d", doc.Kind)
	}
	if len(doc.Content) == 0 {
		doc.Content = []*yaml.Node{{Kind: yaml.MappingNode, Tag: "!!map"}}
	}
	root := doc.Content[0]
	if root.Kind == 0 {
		root.Kind = yaml.MappingNode
		root.Tag = "!!map"
	}
	if root.Kind != yaml.MappingNode {
		return nil, fmt.Errorf("expected YAML root mapping, got kind %d", root.Kind)
	}
	return root, nil
}

func mappingValueNode(mapping *yaml.Node, key string) *yaml.Node {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			return mapping.Content[i+1]
		}
	}
	return nil
}

func setMappingValueNode(mapping *yaml.Node, key string, value *yaml.Node) {
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			mapping.Content[i+1] = cloneYAMLNode(value)
			return
		}
	}
	mapping.Content = append(mapping.Content,
		&yaml.Node{Kind: yaml.ScalarNode, Tag: "!!str", Value: key},
		cloneYAMLNode(value),
	)
}

func deleteMappingValueNode(mapping *yaml.Node, key string) {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			mapping.Content = append(mapping.Content[:i], mapping.Content[i+2:]...)
			return
		}
	}
}

func cloneYAMLNode(node *yaml.Node) *yaml.Node {
	if node == nil {
		return nil
	}
	clone := *node
	if len(node.Content) > 0 {
		clone.Content = make([]*yaml.Node, len(node.Content))
		for i, child := range node.Content {
			clone.Content[i] = cloneYAMLNode(child)
		}
	}
	return &clone
}

func ensureNestedMappingNode(root *yaml.Node, keys ...string) *yaml.Node {
	current := root
	for _, key := range keys {
		next := mappingValueNode(current, key)
		if next == nil || next.Kind != yaml.MappingNode {
			next = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
			setMappingValueNode(current, key, next)
			next = mappingValueNode(current, key)
		}
		current = next
	}
	return current
}

func setNestedMappingValue(root *yaml.Node, value *yaml.Node, keys ...string) {
	if len(keys) == 0 {
		return
	}
	parent := ensureNestedMappingNode(root, keys[:len(keys)-1]...)
	setMappingValueNode(parent, keys[len(keys)-1], value)
}

func deleteNestedMappingValue(root *yaml.Node, keys ...string) {
	if len(keys) == 0 {
		return
	}
	current := root
	for _, key := range keys[:len(keys)-1] {
		current = mappingValueNode(current, key)
		if current == nil || current.Kind != yaml.MappingNode {
			return
		}
	}
	deleteMappingValueNode(current, keys[len(keys)-1])
}

func marshalYAMLDocument(doc *yaml.Node) ([]byte, error) {
	return yaml.Marshal(doc)
}
