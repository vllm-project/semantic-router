//go:build !windows && cgo

package apiserver

import (
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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
	runtimeRoot := cleanKnowledgeBaseSourcePath(knowledgeBaseRuntimeRoot)
	return cleaned == runtimeRoot || strings.HasPrefix(cleaned, runtimeRoot+"/")
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
	return filepath.ToSlash(filepath.Join(knowledgeBaseRuntimeRoot, name)) + "/"
}

func knowledgeBaseRuntimeStateBaseDir(baseDir string) string {
	return filepath.Join(baseDir, ".vllm-sr")
}

func managedKnowledgeBaseDirForSource(baseDir string, sourcePath string, fallbackName string) string {
	cleanedSourcePath := cleanKnowledgeBaseSourcePath(sourcePath)
	runtimeRoot := cleanKnowledgeBaseSourcePath(knowledgeBaseRuntimeRoot)
	if cleanedSourcePath == "" || cleanedSourcePath == runtimeRoot {
		cleanedSourcePath = cleanKnowledgeBaseSourcePath(managedKnowledgeBaseSourcePath(fallbackName))
	}
	if cleanedSourcePath != runtimeRoot && !strings.HasPrefix(cleanedSourcePath, runtimeRoot+"/") {
		cleanedSourcePath = cleanKnowledgeBaseSourcePath(managedKnowledgeBaseSourcePath(fallbackName))
	}
	return filepath.Join(
		knowledgeBaseRuntimeStateBaseDir(baseDir),
		filepath.FromSlash(cleanedSourcePath),
	)
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
