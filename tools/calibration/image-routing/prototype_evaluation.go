package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

type prototypeAsset struct {
	Source, SHA256, Asset, Role string
	SourceGroup                 string `json:"source_group"`
}
type prototypeEvaluation struct {
	root                     string
	assets                   map[string][]prototypeAsset
	assignment, groups       map[string]string
	hashes                   map[string]string
	Excluded                 []string
	classifiers              map[string]*classification.EmbeddingClassifier
	provider                 embedding.Provider
	rules                    []config.EmbeddingRule
	options                  config.HNSWConfig
	ManifestSHA, ProtocolSHA string
}

// preparePrototypeEvaluation resolves deployed content-addressed references back
// to verified repository files. Validation images can never enter either bank.
func preparePrototypeEvaluation(root, manifestPath, protocolPath string, rules []config.EmbeddingRule, options config.HNSWConfig, provider embedding.Provider) (*prototypeEvaluation, error) {
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return nil, err
	}
	var manifest struct {
		RuntimeDirectory string `json:"runtime_directory"`
		Assets           []prototypeAsset
	}
	if err = json.Unmarshal(data, &manifest); err != nil {
		return nil, err
	}
	protocolData, err := os.ReadFile(protocolPath)
	if err != nil {
		return nil, err
	}
	var protocol struct {
		Split         struct{ Assignment, Group map[string]string }
		ContentSHA256 map[string]string `json:"content_sha256"`
	}
	if err = json.Unmarshal(protocolData, &protocol); err != nil {
		return nil, err
	}
	result := &prototypeEvaluation{root: root, assets: map[string][]prototypeAsset{}, assignment: protocol.Split.Assignment, groups: protocol.Split.Group, hashes: protocol.ContentSHA256, classifiers: map[string]*classification.EmbeddingClassifier{}, rules: rules, options: options, provider: &calibrationEmbeddingCache{Provider: provider, values: map[string][]float32{}}, ManifestSHA: fileSHA(manifestPath), ProtocolSHA: fileSHA(protocolPath)}
	for source, assignment := range result.assignment {
		real, _, err := resolveInput(root, source)
		if err != nil {
			return nil, err
		}
		if fileSHA(real) != "sha256:"+result.hashes[source] {
			return nil, fmt.Errorf("frozen protocol content differs: %s", source)
		}
		if assignment == "excluded_duplicate_of_validation" {
			result.Excluded = append(result.Excluded, source)
		}
	}
	sort.Strings(result.Excluded)
	for _, asset := range manifest.Assets {
		if result.assignment[asset.Source] != "development" || result.groups[asset.Source] != asset.SourceGroup {
			return nil, fmt.Errorf("prototype %q is not in its declared development group", asset.Source)
		}
		real, _, err := resolveInput(root, asset.Source)
		if err != nil {
			return nil, err
		}
		if fileSHA(real) != "sha256:"+asset.SHA256 || asset.Asset != asset.SHA256+filepath.Ext(asset.Source) {
			return nil, fmt.Errorf("prototype checksum differs: %s", asset.Source)
		}
		ref := filepath.Join(manifest.RuntimeDirectory, asset.Asset)
		result.assets[ref] = append(result.assets[ref], asset)
	}
	for _, rule := range rules {
		for _, bank := range []struct {
			values []string
			role   string
		}{{rule.ImageCandidates, "positive"}, {rule.NegativeImageCandidates, "negative"}} {
			for _, ref := range bank.values {
				assets, ok := result.assets[ref]
				validRole := ok
				for _, asset := range assets {
					validRole = validRole && asset.Role == bank.role
				}
				if !validRole {
					return nil, fmt.Errorf("unverified %s image candidate %q", bank.role, ref)
				}
			}
		}
	}
	return result, nil
}

func (p *prototypeEvaluation) classifier(path string) (*classification.EmbeddingClassifier, error) {
	group := ""
	querySHA := ""
	if p.assignment[path] == "development" {
		group = p.groups[path]
		querySHA = p.hashes[path]
	} else if p.assignment[path] != "validation" && p.assignment[path] != "excluded_duplicate_of_validation" {
		return nil, fmt.Errorf("fixture %q has no frozen split", path)
	}
	key := group + "\x00" + querySHA
	if c := p.classifiers[key]; c != nil {
		return c, nil
	}
	rules := append([]config.EmbeddingRule(nil), p.rules...)
	resolve := func(values []string) []string {
		result := make([]string, 0, len(values))
		for _, ref := range values {
			// Identical bytes may appear in several source groups. Retain the
			// prototype when at least one authored source is outside the held-out group.
			for _, asset := range p.assets[ref] {
				if group != "" && (asset.SourceGroup == group || asset.SHA256 == querySHA) {
					continue
				}
				result = append(result, filepath.Join(p.root, asset.Source))
				break
			}
		}
		return result
	}
	for i := range rules {
		rules[i].ImageCandidates = resolve(rules[i].ImageCandidates)
		rules[i].NegativeImageCandidates = resolve(rules[i].NegativeImageCandidates)
		if len(rules[i].Candidates)+len(rules[i].ImageCandidates) == 0 {
			return nil, fmt.Errorf("group %q removes the entire positive bank", group)
		}
		if p.rules[i].HasNegativeCandidates() && !rules[i].HasNegativeCandidates() {
			return nil, fmt.Errorf("group %q removes the entire negative bank", group)
		}
	}
	c, err := classification.NewEmbeddingClassifierWithProvider(rules, p.options, p.provider)
	if err != nil {
		return nil, err
	}
	p.classifiers[key] = c
	return c, nil
}

// This evaluation-only cache shares one owned provider across group-excluded
// classifiers. Exact payload and dimension/layer form the cache key.
type calibrationEmbeddingCache struct {
	embedding.Provider
	mu     sync.Mutex
	values map[string][]float32
}

func (c *calibrationEmbeddingCache) cached(key string, infer func() ([]float32, error)) ([]float32, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if v, ok := c.values[key]; ok {
		return v, nil
	}
	v, err := infer()
	if err == nil {
		c.values[key] = v
	}
	return v, err
}

func (c *calibrationEmbeddingCache) Embed(ctx context.Context, text string) ([]float32, error) {
	return c.EmbedWithOptions(ctx, text, embedding.Options{})
}

func (c *calibrationEmbeddingCache) EmbedWithOptions(ctx context.Context, text string, o embedding.Options) ([]float32, error) {
	return c.cached(fmt.Sprintf("text:%d:%d:%s", o.Dimension, o.Layer, text), func() ([]float32, error) { return embedding.Embed(ctx, c.Provider, text, o) })
}

func (c *calibrationEmbeddingCache) EmbedImage(ctx context.Context, data []byte, dim int) ([]float32, error) {
	digest := sha256.Sum256(data)
	return c.cached(fmt.Sprintf("image:%d:%s", dim, hex.EncodeToString(digest[:])), func() ([]float32, error) {
		p, ok := c.Provider.(embedding.ImageProvider)
		if !ok {
			return nil, fmt.Errorf("prepared calibration provider has no image encoder")
		}
		return p.EmbedImage(ctx, data, dim)
	})
}

func (p *prototypeEvaluation) report(rule config.EmbeddingRule, fixtures []fixtureReport) ruleReport {
	if !rule.HasImageCandidates() {
		return calibrateRule(rule, fixtures)
	}
	var development, validation []fixtureReport
	positive := map[string]bool{}
	for _, f := range fixtures {
		if contains(f.PositiveFor, rule.Name) {
			positive[f.Path] = true
		}
		switch p.assignment[f.Path] {
		case "development":
			development = append(development, f)
		case "validation":
			validation = append(validation, f)
		}
	}
	report := calibrateRule(rule, development)
	// Match the prospective frozen protocol: maximum F1, then precision, then
	// higher threshold. Midpoints are computed only from development scores.
	values := make([]float64, 0, len(development))
	for _, f := range development {
		values = append(values, f.Scores[rule.Name])
	}
	sort.Float64s(values)
	thresholds := []float64{values[0] - 1e-6, values[len(values)-1] + 1e-6}
	for i := 1; i < len(values); i++ {
		if values[i] != values[i-1] {
			thresholds = append(thresholds, (values[i]+values[i-1])/2)
		}
	}
	best := evaluateThreshold(thresholds[0], rule.Name, positive, development)
	for _, t := range thresholds[1:] {
		candidate := evaluateThreshold(t, rule.Name, positive, development)
		better, tie := compareF1(candidate, best)
		if better || tie && (candidate.Precision > best.Precision || candidate.Precision == best.Precision && candidate.Threshold > best.Threshold) {
			best = candidate
		}
	}
	report.Selected = best
	heldout := evaluateThreshold(float64(rule.SimilarityThreshold), rule.Name, positive, validation)
	report.Validation = &heldout
	selectedValidation := evaluateThreshold(best.Threshold, rule.Name, positive, validation)
	report.ValidationAtSelected = &selectedValidation
	report.Evaluation = "development source-group and content leave-out; validation content excluded from candidates and fitting"
	return report
}
