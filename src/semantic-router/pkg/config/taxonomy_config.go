package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	ClassifierTypeTaxonomy        = "taxonomy"
	TaxonomyBindKindTier          = "tier"
	TaxonomyBindKindCategory      = "category"
	TaxonomyMetricContrastive     = "contrastive"
	ProjectionInputTaxonomyMetric = "taxonomy_metric"
)

// TaxonomyClassifierConfig declares a reusable taxonomy-backed classifier
// instance that is loaded at router startup.
type TaxonomyClassifierConfig struct {
	Name              string                   `json:"name" yaml:"name"`
	Type              string                   `json:"type" yaml:"type"`
	Source            TaxonomyClassifierSource `json:"source" yaml:"source"`
	Threshold         float32                  `json:"threshold" yaml:"threshold"`
	SecurityThreshold float32                  `json:"security_threshold,omitempty" yaml:"security_threshold,omitempty"`
}

func (c TaxonomyClassifierConfig) NormalizedType() string {
	if c.Type == "" {
		return ClassifierTypeTaxonomy
	}
	return c.Type
}

// TaxonomyClassifierSource points at the classifier asset directory.
type TaxonomyClassifierSource struct {
	Path         string `json:"path" yaml:"path"`
	TaxonomyFile string `json:"taxonomy_file,omitempty" yaml:"taxonomy_file,omitempty"`
}

func (s TaxonomyClassifierSource) taxonomyFileName() string {
	if s.TaxonomyFile == "" {
		return "taxonomy.json"
	}
	return s.TaxonomyFile
}

func (s TaxonomyClassifierSource) ResolveTaxonomyBaseName() string {
	return filepath.Base(s.taxonomyFileName())
}

func (s TaxonomyClassifierSource) ResolvePath(baseDir string) string {
	if s.Path == "" {
		return ""
	}
	if filepath.IsAbs(s.Path) {
		return filepath.Clean(s.Path)
	}

	candidates := make([]string, 0, 4)
	if baseDir != "" {
		candidates = append(candidates, filepath.Join(baseDir, s.Path))
	}
	for _, root := range builtinConfigAssetRoots() {
		candidates = append(candidates, filepath.Join(root, s.Path))
	}

	for _, candidate := range candidates {
		cleaned := filepath.Clean(candidate)
		if pathExists(cleaned) {
			return cleaned
		}
	}

	if baseDir == "" {
		return filepath.Clean(s.Path)
	}
	return filepath.Clean(filepath.Join(baseDir, s.Path))
}

func (s TaxonomyClassifierSource) ResolveTaxonomyPath(baseDir string) string {
	root := s.ResolvePath(baseDir)
	if root == "" {
		return ""
	}
	return filepath.Join(root, s.taxonomyFileName())
}

// TaxonomySignalRule binds one classifier output to a normal routing signal.
type TaxonomySignalRule struct {
	Name       string             `json:"name" yaml:"name"`
	Classifier string             `json:"classifier" yaml:"classifier"`
	Bind       TaxonomySignalBind `json:"bind" yaml:"bind"`
}

// TaxonomySignalBind identifies which classifier namespace a signal should
// match against.
type TaxonomySignalBind struct {
	Kind  string `json:"kind" yaml:"kind"`
	Value string `json:"value" yaml:"value"`
}

// TaxonomyDefinition is the taxonomy manifest loaded from taxonomy.json.
type TaxonomyDefinition struct {
	Version        string                            `json:"version,omitempty" yaml:"version,omitempty"`
	Description    string                            `json:"description,omitempty" yaml:"description,omitempty"`
	Tiers          map[string]TaxonomyTierDefinition `json:"tiers,omitempty" yaml:"tiers,omitempty"`
	Categories     map[string]TaxonomyCategoryEntry  `json:"categories" yaml:"categories"`
	CategoryToTier map[string]string                 `json:"category_to_tier,omitempty" yaml:"category_to_tier,omitempty"`
	TierGroups     map[string][]string               `json:"tier_groups,omitempty" yaml:"tier_groups,omitempty"`
}

type TaxonomyTierDefinition struct {
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

type TaxonomyCategoryEntry struct {
	Tier        string `json:"tier" yaml:"tier"`
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

func LoadTaxonomyDefinition(baseDir string, source TaxonomyClassifierSource) (TaxonomyDefinition, error) {
	path := source.ResolveTaxonomyPath(baseDir)
	if path == "" {
		return TaxonomyDefinition{}, fmt.Errorf("taxonomy classifier source.path cannot be empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return TaxonomyDefinition{}, err
	}
	var taxonomy TaxonomyDefinition
	if err := UnmarshalTaxonomyDefinition(data, &taxonomy); err != nil {
		return TaxonomyDefinition{}, err
	}
	return taxonomy, nil
}

func UnmarshalTaxonomyDefinition(data []byte, target *TaxonomyDefinition) error {
	return json.Unmarshal(data, target)
}

func UnmarshalTaxonomyExemplars(data []byte, target interface{}) error {
	return json.Unmarshal(data, target)
}

func builtinConfigAssetRoots() []string {
	roots := make([]string, 0, 3)
	if envRoot := strings.TrimSpace(os.Getenv("VLLM_SR_CONFIG_ASSET_ROOT")); envRoot != "" {
		roots = append(roots, envRoot)
	}
	roots = append(roots, "/app/config")
	if _, file, _, ok := runtime.Caller(0); ok {
		roots = append(roots, filepath.Join(filepath.Dir(file), "..", "..", "..", "..", "config"))
	}
	return roots
}

func pathExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
