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
	KBTargetKindLabel        = "label"
	KBTargetKindGroup        = "group"
	KBMatchBest              = "best"
	KBMatchThreshold         = "threshold"
	KBMetricTypeGroupMargin  = "group_margin"
	KBMetricBestScore        = "best_score"
	KBMetricBestMatchedScore = "best_matched_score"
	ProjectionInputKBMetric  = "kb_metric"
)

// KnowledgeBaseConfig declares a reusable embedding-backed KB instance that is
// loaded at router startup.
type KnowledgeBaseConfig struct {
	Name            string                      `json:"name" yaml:"name"`
	Source          KnowledgeBaseSource         `json:"source" yaml:"source"`
	Threshold       float32                     `json:"threshold" yaml:"threshold"`
	LabelThresholds map[string]float32          `json:"label_thresholds,omitempty" yaml:"label_thresholds,omitempty"`
	Groups          map[string][]string         `json:"groups,omitempty" yaml:"groups,omitempty"`
	Metrics         []KnowledgeBaseMetricConfig `json:"metrics,omitempty" yaml:"metrics,omitempty"`
}

// KnowledgeBaseSource points at the KB asset directory and manifest.
type KnowledgeBaseSource struct {
	Path     string `json:"path" yaml:"path"`
	Manifest string `json:"manifest,omitempty" yaml:"manifest,omitempty"`
}

func (s KnowledgeBaseSource) manifestFileName() string {
	if strings.TrimSpace(s.Manifest) == "" {
		return "labels.json"
	}
	return s.Manifest
}

func (s KnowledgeBaseSource) ResolvePath(baseDir string) string {
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

func (s KnowledgeBaseSource) ResolveManifestPath(baseDir string) string {
	root := s.ResolvePath(baseDir)
	if root == "" {
		return ""
	}
	return filepath.Join(root, s.manifestFileName())
}

func (s KnowledgeBaseSource) ResolveManifestBaseName() string {
	return filepath.Base(s.manifestFileName())
}

// KBSignalRule binds one KB output to a normal routing signal.
type KBSignalRule struct {
	Name   string         `json:"name" yaml:"name"`
	KB     string         `json:"kb" yaml:"kb"`
	Target KBSignalTarget `json:"target" yaml:"target"`
	Match  string         `json:"match,omitempty" yaml:"match,omitempty"`
}

// KBSignalTarget identifies which KB namespace a signal should match against.
type KBSignalTarget struct {
	Kind  string `json:"kind" yaml:"kind"`
	Value string `json:"value" yaml:"value"`
}

// KnowledgeBaseMetricConfig declares one named numeric metric derived from KB
// label scores.
type KnowledgeBaseMetricConfig struct {
	Name          string `json:"name" yaml:"name"`
	Type          string `json:"type" yaml:"type"`
	PositiveGroup string `json:"positive_group,omitempty" yaml:"positive_group,omitempty"`
	NegativeGroup string `json:"negative_group,omitempty" yaml:"negative_group,omitempty"`
}

// KnowledgeBaseDefinition is the neutral labels manifest loaded from labels.json.
type KnowledgeBaseDefinition struct {
	Version     string                           `json:"version,omitempty" yaml:"version,omitempty"`
	Description string                           `json:"description,omitempty" yaml:"description,omitempty"`
	Labels      map[string]KnowledgeBaseLabelDef `json:"labels" yaml:"labels"`
}

type KnowledgeBaseLabelDef struct {
	Description string   `json:"description,omitempty" yaml:"description,omitempty"`
	Exemplars   []string `json:"exemplars" yaml:"exemplars"`
}

func LoadKnowledgeBaseDefinition(baseDir string, source KnowledgeBaseSource) (KnowledgeBaseDefinition, error) {
	path := source.ResolveManifestPath(baseDir)
	if path == "" {
		return KnowledgeBaseDefinition{}, fmt.Errorf("knowledge base source.path cannot be empty")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return KnowledgeBaseDefinition{}, err
	}
	var kb KnowledgeBaseDefinition
	if err := json.Unmarshal(data, &kb); err != nil {
		return KnowledgeBaseDefinition{}, err
	}
	return kb, nil
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
