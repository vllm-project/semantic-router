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
	taxonomyClassifierCustomRoot = "classifiers/custom"
	taxonomyManifestVersion      = "1.0.0"
)

var taxonomyNamePattern = regexp.MustCompile(`^[A-Za-z0-9_-]+$`)

type taxonomyClassifierTierPayload struct {
	Name        string `json:"name" yaml:"name"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

type taxonomyClassifierCategoryPayload struct {
	Name        string   `json:"name" yaml:"name"`
	Tier        string   `json:"tier" yaml:"tier"`
	Description string   `json:"description,omitempty" yaml:"description,omitempty"`
	Exemplars   []string `json:"exemplars" yaml:"exemplars"`
}

type taxonomySignalReference struct {
	Name string                    `json:"name" yaml:"name"`
	Bind config.TaxonomySignalBind `json:"bind" yaml:"bind"`
}

type taxonomyClassifierBindOptions struct {
	Tiers      []string `json:"tiers" yaml:"tiers"`
	Categories []string `json:"categories" yaml:"categories"`
}

type taxonomyClassifierDocument struct {
	Name              string                              `json:"name" yaml:"name"`
	Type              string                              `json:"type" yaml:"type"`
	Builtin           bool                                `json:"builtin" yaml:"builtin"`
	Managed           bool                                `json:"managed" yaml:"managed"`
	Editable          bool                                `json:"editable" yaml:"editable"`
	Threshold         float32                             `json:"threshold" yaml:"threshold"`
	SecurityThreshold float32                             `json:"security_threshold,omitempty" yaml:"security_threshold,omitempty"`
	Description       string                              `json:"description,omitempty" yaml:"description,omitempty"`
	Source            config.TaxonomyClassifierSource     `json:"source" yaml:"source"`
	Tiers             []taxonomyClassifierTierPayload     `json:"tiers,omitempty" yaml:"tiers,omitempty"`
	Categories        []taxonomyClassifierCategoryPayload `json:"categories,omitempty" yaml:"categories,omitempty"`
	TierGroups        map[string][]string                 `json:"tier_groups,omitempty" yaml:"tier_groups,omitempty"`
	SignalReferences  []taxonomySignalReference           `json:"signal_references,omitempty" yaml:"signal_references,omitempty"`
	BindOptions       taxonomyClassifierBindOptions       `json:"bind_options" yaml:"bind_options"`
	LoadError         string                              `json:"load_error,omitempty" yaml:"load_error,omitempty"`
}

type taxonomyClassifierListResponse struct {
	Items []taxonomyClassifierDocument `json:"items" yaml:"items"`
}

type taxonomyClassifierUpsertRequest struct {
	Name              string                              `json:"name,omitempty"`
	Threshold         float32                             `json:"threshold" yaml:"threshold"`
	SecurityThreshold float32                             `json:"security_threshold,omitempty" yaml:"security_threshold,omitempty"`
	Description       string                              `json:"description,omitempty" yaml:"description,omitempty"`
	Tiers             []taxonomyClassifierTierPayload     `json:"tiers,omitempty" yaml:"tiers,omitempty"`
	Categories        []taxonomyClassifierCategoryPayload `json:"categories" yaml:"categories"`
	TierGroups        map[string][]string                 `json:"tier_groups,omitempty" yaml:"tier_groups,omitempty"`
}

type taxonomyClassifierDeleteResponse struct {
	Status string `json:"status" yaml:"status"`
	Name   string `json:"name" yaml:"name"`
}

type taxonomyExemplarFile struct {
	Category    string   `json:"category,omitempty"`
	Tier        string   `json:"tier,omitempty"`
	Description string   `json:"description,omitempty"`
	Exemplars   []string `json:"exemplars"`
}

type managedTaxonomyClassifierAssetsTxn struct {
	finalDir   string
	backupDir  string
	removeOnly bool
}

func normalizeTaxonomyClassifierRequest(payload taxonomyClassifierUpsertRequest) (taxonomyClassifierUpsertRequest, error) {
	payload.Name = strings.TrimSpace(payload.Name)
	if payload.Name == "" {
		return payload, fmt.Errorf("name is required")
	}
	if !taxonomyNamePattern.MatchString(payload.Name) {
		return payload, fmt.Errorf("name %q contains invalid characters (allowed: letters, numbers, _, -)", payload.Name)
	}
	if len(payload.Categories) == 0 {
		return payload, fmt.Errorf("at least one category is required")
	}

	tierDescriptions := make(map[string]string, len(payload.Tiers))
	seenTiers := make(map[string]struct{}, len(payload.Tiers))
	normalizedTiers := make([]taxonomyClassifierTierPayload, 0, len(payload.Tiers))
	for _, tier := range payload.Tiers {
		tier.Name = strings.TrimSpace(tier.Name)
		tier.Description = strings.TrimSpace(tier.Description)
		if tier.Name == "" {
			return payload, fmt.Errorf("tier name cannot be empty")
		}
		if !taxonomyNamePattern.MatchString(tier.Name) {
			return payload, fmt.Errorf("tier %q contains invalid characters", tier.Name)
		}
		if _, exists := seenTiers[tier.Name]; exists {
			return payload, fmt.Errorf("tier %q is declared more than once", tier.Name)
		}
		seenTiers[tier.Name] = struct{}{}
		tierDescriptions[tier.Name] = tier.Description
		normalizedTiers = append(normalizedTiers, tier)
	}

	seenCategories := make(map[string]struct{}, len(payload.Categories))
	normalizedCategories := make([]taxonomyClassifierCategoryPayload, 0, len(payload.Categories))
	for _, category := range payload.Categories {
		category.Name = strings.TrimSpace(category.Name)
		category.Tier = strings.TrimSpace(category.Tier)
		category.Description = strings.TrimSpace(category.Description)
		if category.Name == "" {
			return payload, fmt.Errorf("category name cannot be empty")
		}
		if !taxonomyNamePattern.MatchString(category.Name) {
			return payload, fmt.Errorf("category %q contains invalid characters", category.Name)
		}
		if category.Tier == "" {
			return payload, fmt.Errorf("category %q must declare a tier", category.Name)
		}
		if !taxonomyNamePattern.MatchString(category.Tier) {
			return payload, fmt.Errorf("category %q tier %q contains invalid characters", category.Name, category.Tier)
		}
		if _, exists := seenCategories[category.Name]; exists {
			return payload, fmt.Errorf("category %q is declared more than once", category.Name)
		}
		seenCategories[category.Name] = struct{}{}
		if len(category.Exemplars) == 0 {
			return payload, fmt.Errorf("category %q must include at least one exemplar", category.Name)
		}
		cleanExemplars := make([]string, 0, len(category.Exemplars))
		for _, exemplar := range category.Exemplars {
			trimmed := strings.TrimSpace(exemplar)
			if trimmed == "" {
				continue
			}
			cleanExemplars = append(cleanExemplars, trimmed)
		}
		if len(cleanExemplars) == 0 {
			return payload, fmt.Errorf("category %q must include at least one non-empty exemplar", category.Name)
		}
		category.Exemplars = cleanExemplars
		if _, exists := seenTiers[category.Tier]; !exists {
			seenTiers[category.Tier] = struct{}{}
			normalizedTiers = append(normalizedTiers, taxonomyClassifierTierPayload{
				Name:        category.Tier,
				Description: tierDescriptions[category.Tier],
			})
		}
		normalizedCategories = append(normalizedCategories, category)
	}

	for groupName, categories := range payload.TierGroups {
		if strings.TrimSpace(groupName) == "" {
			return payload, fmt.Errorf("tier_groups cannot contain an empty group name")
		}
		cleanCategories := make([]string, 0, len(categories))
		seenGroupCategories := make(map[string]struct{}, len(categories))
		for _, categoryName := range categories {
			categoryName = strings.TrimSpace(categoryName)
			if _, exists := seenCategories[categoryName]; !exists {
				return payload, fmt.Errorf("tier_groups[%q] references unknown category %q", groupName, categoryName)
			}
			if _, exists := seenGroupCategories[categoryName]; exists {
				continue
			}
			seenGroupCategories[categoryName] = struct{}{}
			cleanCategories = append(cleanCategories, categoryName)
		}
		sort.Strings(cleanCategories)
		payload.TierGroups[groupName] = cleanCategories
	}

	sort.Slice(normalizedTiers, func(i, j int) bool { return normalizedTiers[i].Name < normalizedTiers[j].Name })
	sort.Slice(normalizedCategories, func(i, j int) bool { return normalizedCategories[i].Name < normalizedCategories[j].Name })

	payload.Tiers = normalizedTiers
	payload.Categories = normalizedCategories
	payload.Description = strings.TrimSpace(payload.Description)
	return payload, nil
}

func defaultTaxonomyClassifierMap() map[string]config.TaxonomyClassifierConfig {
	defaults := config.DefaultCanonicalGlobal().ModelCatalog.Classifiers
	result := make(map[string]config.TaxonomyClassifierConfig, len(defaults))
	for _, classifier := range defaults {
		result[classifier.Name] = classifier
	}
	return result
}

func isBuiltinTaxonomyClassifier(classifier config.TaxonomyClassifierConfig) bool {
	defaults := defaultTaxonomyClassifierMap()
	defaultClassifier, ok := defaults[classifier.Name]
	if !ok {
		return false
	}
	return cleanClassifierSourcePath(defaultClassifier.Source.Path) == cleanClassifierSourcePath(classifier.Source.Path)
}

func isManagedTaxonomyClassifier(classifier config.TaxonomyClassifierConfig) bool {
	cleaned := cleanClassifierSourcePath(classifier.Source.Path)
	customRoot := cleanClassifierSourcePath(taxonomyClassifierCustomRoot)
	return cleaned == customRoot || strings.HasPrefix(cleaned, customRoot+"/")
}

func cleanClassifierSourcePath(value string) string {
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

func managedTaxonomyClassifierSourcePath(name string) string {
	return filepath.ToSlash(filepath.Join(taxonomyClassifierCustomRoot, name)) + "/"
}

func managedTaxonomyClassifierDir(baseDir string, name string) string {
	return filepath.Join(baseDir, filepath.FromSlash(cleanClassifierSourcePath(managedTaxonomyClassifierSourcePath(name))))
}

func taxonomyClassifierConfigBaseDir(cfg *config.RouterConfig, configPath string) string {
	if cfg != nil && strings.TrimSpace(cfg.ConfigBaseDir) != "" {
		return cfg.ConfigBaseDir
	}
	if configPath == "" {
		return ""
	}
	return filepath.Dir(resolveConfigPersistencePaths(configPath).sourcePath)
}

func listTaxonomyClassifierDocuments(cfg *config.RouterConfig, baseDir string) ([]taxonomyClassifierDocument, error) {
	documents := make([]taxonomyClassifierDocument, 0, len(cfg.TaxonomyClassifiers))
	for _, classifier := range cfg.TaxonomyClassifiers {
		document, err := buildTaxonomyClassifierDocument(cfg, baseDir, classifier)
		if err != nil {
			return nil, err
		}
		documents = append(documents, document)
	}
	sort.Slice(documents, func(i, j int) bool { return documents[i].Name < documents[j].Name })
	return documents, nil
}

func buildTaxonomyClassifierDocument(cfg *config.RouterConfig, baseDir string, classifier config.TaxonomyClassifierConfig) (taxonomyClassifierDocument, error) {
	document := taxonomyClassifierDocument{
		Name:              classifier.Name,
		Type:              classifier.NormalizedType(),
		Builtin:           isBuiltinTaxonomyClassifier(classifier),
		Managed:           isManagedTaxonomyClassifier(classifier),
		Editable:          isManagedTaxonomyClassifier(classifier),
		Threshold:         classifier.Threshold,
		SecurityThreshold: classifier.SecurityThreshold,
		Source:            classifier.Source,
		SignalReferences:  taxonomySignalReferences(cfg, classifier.Name),
	}

	taxonomy, err := config.LoadTaxonomyDefinition(baseDir, classifier.Source)
	if err != nil {
		document.LoadError = err.Error()
		return document, nil
	}

	document.Description = strings.TrimSpace(taxonomy.Description)
	document.TierGroups = cloneTierGroups(taxonomy.TierGroups)
	document.Categories, err = loadTaxonomyClassifierCategories(baseDir, classifier.Source, taxonomy)
	if err != nil {
		document.LoadError = err.Error()
	}
	document.Tiers = taxonomyTierPayloads(taxonomy, document.Categories)
	document.BindOptions = taxonomyClassifierBindOptions{
		Tiers:      extractTaxonomyTierNames(document.Tiers),
		Categories: extractTaxonomyCategoryNames(document.Categories),
	}
	return document, nil
}

func taxonomySignalReferences(cfg *config.RouterConfig, classifierName string) []taxonomySignalReference {
	references := make([]taxonomySignalReference, 0)
	if cfg == nil {
		return references
	}
	for _, rule := range cfg.TaxonomyRules {
		if rule.Classifier != classifierName {
			continue
		}
		references = append(references, taxonomySignalReference{
			Name: rule.Name,
			Bind: rule.Bind,
		})
	}
	sort.Slice(references, func(i, j int) bool { return references[i].Name < references[j].Name })
	return references
}

func loadTaxonomyClassifierCategories(
	baseDir string,
	source config.TaxonomyClassifierSource,
	taxonomy config.TaxonomyDefinition,
) ([]taxonomyClassifierCategoryPayload, error) {
	root := source.ResolvePath(baseDir)
	categories := make([]taxonomyClassifierCategoryPayload, 0, len(taxonomy.Categories))
	for categoryName, entry := range taxonomy.Categories {
		payload := taxonomyClassifierCategoryPayload{
			Name:        categoryName,
			Tier:        entry.Tier,
			Description: strings.TrimSpace(entry.Description),
		}
		filePath := filepath.Join(root, categoryName+".json")
		data, err := os.ReadFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("read category file %s: %w", filePath, err)
		}
		var exemplarFile taxonomyExemplarFile
		if err := config.UnmarshalTaxonomyExemplars(data, &exemplarFile); err != nil {
			return nil, fmt.Errorf("parse category file %s: %w", filePath, err)
		}
		payload.Exemplars = append([]string(nil), exemplarFile.Exemplars...)
		if payload.Description == "" {
			payload.Description = strings.TrimSpace(exemplarFile.Description)
		}
		if payload.Tier == "" {
			payload.Tier = strings.TrimSpace(exemplarFile.Tier)
		}
		categories = append(categories, payload)
	}
	sort.Slice(categories, func(i, j int) bool { return categories[i].Name < categories[j].Name })
	return categories, nil
}

func taxonomyTierPayloads(
	taxonomy config.TaxonomyDefinition,
	categories []taxonomyClassifierCategoryPayload,
) []taxonomyClassifierTierPayload {
	tiers := make([]taxonomyClassifierTierPayload, 0, len(taxonomy.Tiers))
	seen := make(map[string]struct{}, len(taxonomy.Tiers))
	for name, tier := range taxonomy.Tiers {
		tiers = append(tiers, taxonomyClassifierTierPayload{
			Name:        name,
			Description: strings.TrimSpace(tier.Description),
		})
		seen[name] = struct{}{}
	}
	for _, category := range categories {
		if _, exists := seen[category.Tier]; exists {
			continue
		}
		seen[category.Tier] = struct{}{}
		tiers = append(tiers, taxonomyClassifierTierPayload{Name: category.Tier})
	}
	sort.Slice(tiers, func(i, j int) bool { return tiers[i].Name < tiers[j].Name })
	return tiers
}

func extractTaxonomyTierNames(tiers []taxonomyClassifierTierPayload) []string {
	names := make([]string, 0, len(tiers))
	for _, tier := range tiers {
		names = append(names, tier.Name)
	}
	sort.Strings(names)
	return names
}

func extractTaxonomyCategoryNames(categories []taxonomyClassifierCategoryPayload) []string {
	names := make([]string, 0, len(categories))
	for _, category := range categories {
		names = append(names, category.Name)
	}
	sort.Strings(names)
	return names
}

func cloneTierGroups(groups map[string][]string) map[string][]string {
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

func desiredTaxonomyClassifiers(
	current []config.TaxonomyClassifierConfig,
	next config.TaxonomyClassifierConfig,
) []config.TaxonomyClassifierConfig {
	updated := make([]config.TaxonomyClassifierConfig, 0, len(current)+1)
	replaced := false
	for _, classifier := range current {
		if classifier.Name == next.Name {
			updated = append(updated, next)
			replaced = true
			continue
		}
		updated = append(updated, classifier)
	}
	if !replaced {
		updated = append(updated, next)
	}
	sort.Slice(updated, func(i, j int) bool { return updated[i].Name < updated[j].Name })
	return updated
}

func removeTaxonomyClassifier(
	current []config.TaxonomyClassifierConfig,
	name string,
) []config.TaxonomyClassifierConfig {
	filtered := make([]config.TaxonomyClassifierConfig, 0, len(current))
	for _, classifier := range current {
		if classifier.Name == name {
			continue
		}
		filtered = append(filtered, classifier)
	}
	sort.Slice(filtered, func(i, j int) bool { return filtered[i].Name < filtered[j].Name })
	return filtered
}

func taxonomyClassifierOverrideYAML(
	existingData []byte,
	classifiers []config.TaxonomyClassifierConfig,
) ([]byte, error) {
	doc, err := parseYAMLDocument(existingData)
	if err != nil {
		return nil, fmt.Errorf("parse config.yaml: %w", err)
	}
	root, err := documentMappingNode(doc)
	if err != nil {
		return nil, err
	}

	if !shouldPersistTaxonomyClassifierOverride(classifiers) {
		deleteNestedMappingValue(root, "global", "model_catalog", "classifiers")
		return marshalYAMLDocument(doc)
	}

	classifierNode, err := yamlNodeFromValue(classifiers)
	if err != nil {
		return nil, fmt.Errorf("encode classifier override: %w", err)
	}
	setNestedMappingValue(root, classifierNode, "global", "model_catalog", "classifiers")
	return marshalYAMLDocument(doc)
}

func shouldPersistTaxonomyClassifierOverride(classifiers []config.TaxonomyClassifierConfig) bool {
	defaults := normalizeTaxonomyClassifierSlice(config.DefaultCanonicalGlobal().ModelCatalog.Classifiers)
	current := normalizeTaxonomyClassifierSlice(classifiers)
	return !reflect.DeepEqual(defaults, current)
}

func normalizeTaxonomyClassifierSlice(classifiers []config.TaxonomyClassifierConfig) []config.TaxonomyClassifierConfig {
	normalized := make([]config.TaxonomyClassifierConfig, 0, len(classifiers))
	for _, classifier := range classifiers {
		copyClassifier := classifier
		copyClassifier.Type = copyClassifier.NormalizedType()
		copyClassifier.Source.Path = strings.TrimSpace(copyClassifier.Source.Path)
		copyClassifier.Source.TaxonomyFile = strings.TrimSpace(copyClassifier.Source.TaxonomyFile)
		normalized = append(normalized, copyClassifier)
	}
	sort.Slice(normalized, func(i, j int) bool { return normalized[i].Name < normalized[j].Name })
	return normalized
}

func validateConfigWithBaseDir(baseDir string, yamlBytes []byte) (*config.RouterConfig, error) {
	tempFile, err := os.CreateTemp(baseDir, ".taxonomy-classifier-*.yaml")
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

func stageManagedTaxonomyClassifierAssets(
	baseDir string,
	payload taxonomyClassifierUpsertRequest,
) (*managedTaxonomyClassifierAssetsTxn, error) {
	finalDir := managedTaxonomyClassifierDir(baseDir, payload.Name)
	parentDir := filepath.Dir(finalDir)
	if err := os.MkdirAll(parentDir, 0o755); err != nil {
		return nil, err
	}

	stageDir := finalDir + ".tmp-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.MkdirAll(stageDir, 0o755); err != nil {
		return nil, err
	}

	if err := writeTaxonomyClassifierAssets(stageDir, payload); err != nil {
		_ = os.RemoveAll(stageDir)
		return nil, err
	}

	txn := &managedTaxonomyClassifierAssetsTxn{finalDir: finalDir}
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

func stageManagedTaxonomyClassifierRemoval(
	baseDir string,
	name string,
) (*managedTaxonomyClassifierAssetsTxn, error) {
	finalDir := managedTaxonomyClassifierDir(baseDir, name)
	if _, err := os.Stat(finalDir); os.IsNotExist(err) {
		return nil, nil
	}
	backupDir := finalDir + ".bak-" + fmt.Sprintf("%d", time.Now().UnixNano())
	if err := os.Rename(finalDir, backupDir); err != nil {
		return nil, err
	}
	return &managedTaxonomyClassifierAssetsTxn{
		finalDir:   finalDir,
		backupDir:  backupDir,
		removeOnly: true,
	}, nil
}

func (txn *managedTaxonomyClassifierAssetsTxn) Commit() {
	if txn == nil || txn.backupDir == "" {
		return
	}
	_ = os.RemoveAll(txn.backupDir)
}

func (txn *managedTaxonomyClassifierAssetsTxn) Rollback() {
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

func writeTaxonomyClassifierAssets(root string, payload taxonomyClassifierUpsertRequest) error {
	taxonomy := config.TaxonomyDefinition{
		Version:        taxonomyManifestVersion,
		Description:    payload.Description,
		Tiers:          make(map[string]config.TaxonomyTierDefinition, len(payload.Tiers)),
		Categories:     make(map[string]config.TaxonomyCategoryEntry, len(payload.Categories)),
		CategoryToTier: make(map[string]string, len(payload.Categories)),
		TierGroups:     cloneTierGroups(payload.TierGroups),
	}

	for _, tier := range payload.Tiers {
		taxonomy.Tiers[tier.Name] = config.TaxonomyTierDefinition{
			Description: tier.Description,
		}
	}
	for _, category := range payload.Categories {
		taxonomy.Categories[category.Name] = config.TaxonomyCategoryEntry{
			Tier:        category.Tier,
			Description: category.Description,
		}
		taxonomy.CategoryToTier[category.Name] = category.Tier
	}

	taxonomyBytes, err := json.MarshalIndent(taxonomy, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(root, "taxonomy.json"), taxonomyBytes, 0o644); err != nil {
		return err
	}

	for _, category := range payload.Categories {
		payload := taxonomyExemplarFile{
			Category:    category.Name,
			Tier:        category.Tier,
			Description: category.Description,
			Exemplars:   category.Exemplars,
		}
		categoryBytes, err := json.MarshalIndent(payload, "", "  ")
		if err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(root, category.Name+".json"), categoryBytes, 0o644); err != nil {
			return err
		}
	}
	return nil
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
