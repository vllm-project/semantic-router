package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type prototypeTestProvider struct{ vectors map[string][]float32 }

func (p *prototypeTestProvider) Embed(context.Context, string) ([]float32, error) {
	panic("image-only bank used text encoder")
}

func (p *prototypeTestProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	panic("unexpected batch")
}
func (p *prototypeTestProvider) Dimension() int  { return 2 }
func (p *prototypeTestProvider) Backend() string { return "owned-test" }
func (p *prototypeTestProvider) EmbedImage(_ context.Context, data []byte, _ int) ([]float32, error) {
	return p.vectors[string(data)], nil
}

func TestPreparePrototypeEvaluationFromRouterModule(t *testing.T) {
	root := t.TempDir()
	module := filepath.Join(root, "src", "semantic-router")
	if err := os.MkdirAll(module, 0o755); err != nil {
		t.Fatal(err)
	}
	// Make runs the calibration command from the Router module, whereas both
	// frozen documents name their sources relative to the checkout root.
	t.Chdir(module)
	assignment, groups, hashes := map[string]string{}, map[string]string{}, map[string]string{}
	var assets []prototypeAsset
	rule := config.EmbeddingRule{Name: "code", QueryModality: config.QueryModalityImage, AggregationMethodConfiged: config.AggregationMethodMax}
	for _, name := range []string{"positive", "negative", "query"} {
		source := "website/static/img/" + name + ".png"
		path := filepath.Join(root, filepath.FromSlash(source))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(name), 0o600); err != nil {
			t.Fatal(err)
		}
		hashes[source] = strings.TrimPrefix(fileSHA(path), "sha256:")
		groups[source] = name
		assignment[source] = "development"
		if name == "query" {
			assignment[source] = "validation"
			continue
		}
		asset := prototypeAsset{Source: source, SHA256: hashes[source], Asset: hashes[source] + ".png", Role: name, SourceGroup: name}
		assets = append(assets, asset)
		ref := filepath.Join("/app/share/image-routing", asset.Asset)
		if name == "positive" {
			rule.ImageCandidates = []string{ref}
		} else {
			rule.NegativeImageCandidates = []string{ref}
		}
	}
	writeJSON := func(name string, value any) string {
		t.Helper()
		data, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}
	manifest := writeJSON("manifest.json", map[string]any{"runtime_directory": "/app/share/image-routing", "assets": assets})
	protocol := writeJSON("protocol.json", map[string]any{"split": map[string]any{"assignment": assignment, "group": groups}, "content_sha256": hashes})
	provider := &prototypeTestProvider{vectors: map[string][]float32{"query": {1, 0}, "positive": {0.8, 0.6}, "negative": {0.1, float32(math.Sqrt(.99))}}}
	p, err := preparePrototypeEvaluation(root, manifest, protocol, []config.EmbeddingRule{rule}, config.HNSWConfig{}, provider)
	if err != nil {
		t.Fatal(err)
	}
	c, err := p.classifier("website/static/img/query.png")
	if err != nil {
		t.Fatal(err)
	}
	result, err := c.ClassifyDetailedMultimodal(config.QueryModalityImage, base64.StdEncoding.EncodeToString([]byte("query")))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 1 || math.Abs(result.Scores[0].Score-0.7) > 1e-6 {
		t.Fatalf("repository-relative prototype banks were not scored: %+v", result.Scores)
	}
}

func TestResolvePrototypeSourceRejectsNonCanonicalAndSymlinkPaths(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "source.png")
	if err := os.WriteFile(path, []byte("source"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(path, filepath.Join(root, "alias.png")); err != nil {
		t.Fatal(err)
	}
	for _, source := range []string{path, "../source.png", "./source.png", "alias.png"} {
		t.Run(source, func(t *testing.T) {
			if _, err := resolvePrototypeSource(root, source); err == nil {
				t.Fatalf("accepted noncanonical or aliased source %q", source)
			}
		})
	}
}

func TestPrototypeEvaluationExcludesGroupAndDuplicateContent(t *testing.T) {
	root := t.TempDir()
	provider := &prototypeTestProvider{vectors: map[string][]float32{"query": {1, 0}, "positive": {0.8, 0.6}, "negative": {0.1, float32(math.Sqrt(.99))}}}
	for _, name := range []string{"query", "positive", "negative"} {
		if err := os.WriteFile(filepath.Join(root, name), []byte(name), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	rule := config.EmbeddingRule{Name: "code", QueryModality: config.QueryModalityImage, ImageCandidates: []string{"positive-same-group", "positive-other"}, NegativeImageCandidates: []string{"negative-same-content", "negative-other"}, AggregationMethodConfiged: config.AggregationMethodMax}
	p := &prototypeEvaluation{root: root, assignment: map[string]string{"query": "development"}, groups: map[string]string{"query": "held"}, hashes: map[string]string{"query": "same-bytes"}, assets: map[string][]prototypeAsset{
		"positive-same-group":   {{Source: "query", SourceGroup: "held", SHA256: "same-bytes"}},
		"positive-other":        {{Source: "positive", SourceGroup: "other", SHA256: "positive-bytes"}},
		"negative-same-content": {{Source: "query", SourceGroup: "alias", SHA256: "same-bytes"}},
		"negative-other":        {{Source: "negative", SourceGroup: "other", SHA256: "negative-bytes"}},
	}, rules: []config.EmbeddingRule{rule}, options: config.HNSWConfig{}, provider: provider, classifiers: map[string]*classification.EmbeddingClassifier{}}
	c, err := p.classifier("query")
	if err != nil {
		t.Fatal(err)
	}
	result, err := c.ClassifyDetailedMultimodal(config.QueryModalityImage, base64.StdEncoding.EncodeToString([]byte("query")))
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Scores) != 1 || math.Abs(result.Scores[0].Score-0.7) > 1e-6 {
		t.Fatalf("held-out source or duplicate bytes leaked: %+v", result.Scores)
	}
}

func TestPrototypeReportKeepsValidationAndDuplicateExclusionsSeparate(t *testing.T) {
	p := &prototypeEvaluation{assignment: map[string]string{"dev-positive": "development", "dev-negative": "development", "val-positive": "validation", "val-negative": "validation", "duplicate": "excluded_duplicate_of_validation"}}
	rule := config.EmbeddingRule{Name: "code", ImageCandidates: []string{"positive"}, SimilarityThreshold: 0.5}
	var fixtures []fixtureReport
	for _, name := range []string{"dev-positive", "dev-negative", "val-positive", "val-negative", "duplicate"} {
		score := 0.1
		var positive []string
		if name == "dev-positive" || name == "val-positive" {
			score = 0.9
			positive = []string{"code"}
		}
		fixtures = append(fixtures, fixtureReport{Path: name, PositiveFor: positive, Scores: map[string]float64{"code": score}})
	}
	report := p.report(rule, fixtures)
	if report.Positives != 1 || report.Negatives != 1 || report.Validation == nil || report.Validation.TP != 1 || report.Validation.TN != 1 {
		t.Fatalf("split accounting differs: %+v", report)
	}
}
