//go:build !windows && cgo && (amd64 || arm64)

package modelselection

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type kmeansFixtures struct {
	Cases []struct {
		Name     string          `json:"name"`
		Artifact json.RawMessage `json:"artifact"`
		Queries  []struct {
			Vector     []float64          `json:"vector"`
			Candidates []string           `json:"candidates"`
			ClusterID  *int               `json:"cluster_id"`
			Scores     map[string]float64 `json:"scores"`
			Model      string             `json:"model"`
			Error      string             `json:"error"`
		} `json:"queries"`
	} `json:"cases"`
	ArtifactRejects []struct {
		Reason   string          `json:"reason"`
		Artifact json.RawMessage `json:"artifact"`
	} `json:"artifact_rejects"`
}

func loadKMeansFixtures(t *testing.T) kmeansFixtures {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "ml-binding", "tests", "fixtures", "python_kmeans.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixtures kmeansFixtures
	if err := json.Unmarshal(data, &fixtures); err != nil {
		t.Fatal(err)
	}
	return fixtures
}

func loadKMeans(t *testing.T, artifact []byte) *KMeansSelector {
	t.Helper()
	s := NewKMeansSelector(8)
	t.Cleanup(func() { _ = s.Close() })
	if err := s.LoadFromJSON(artifact); err != nil {
		t.Fatal(err)
	}
	return s
}

func refsFor(names []string) []config.ModelRef {
	refs := make([]config.ModelRef, len(names))
	for i, name := range names {
		refs[i] = config.ModelRef{Model: name}
	}
	return refs
}

func TestKMeansSelectionMatchesPythonFixture(t *testing.T) {
	for _, tc := range loadKMeansFixtures(t).Cases {
		t.Run(tc.Name, func(t *testing.T) {
			s := loadKMeans(t, tc.Artifact)
			for i, q := range tc.Queries {
				label := fmt.Sprintf("query %d %v", i, q.Candidates)
				candidates := q.Candidates
				if candidates == nil {
					candidates = s.candidates
				}
				got, err := s.selectFeatures(q.Vector, refsFor(candidates))
				switch q.Error {
				case "no_eligible_candidate":
					if !errors.Is(err, ErrNoEligibleCandidate) || got != nil {
						t.Fatalf("%s: got %v, %v; want ErrNoEligibleCandidate", label, got, err)
					}
					continue
				case "invalid_query":
					if err == nil || errors.Is(err, ErrNoEligibleCandidate) {
						t.Fatalf("%s: got %v, %v; want a scoring error", label, got, err)
					}
					continue
				}
				if err != nil || got.Model != q.Model {
					t.Fatalf("%s: selected %v, %v; want %s", label, got, err, q.Model)
				}
				scores := make([]float64, len(s.candidates))
				cluster, err := s.mlKMeans.Score(q.Vector, scores)
				if err != nil || q.ClusterID == nil || cluster != *q.ClusterID {
					t.Fatalf("%s: cluster %d, %v; want %v", label, cluster, err, q.ClusterID)
				}
				for j, name := range s.candidates {
					if want, ok := q.Scores[name]; ok && scores[j] != want {
						t.Fatalf("%s: %s scored %v, want %v", label, name, scores[j], want)
					}
				}
			}
		})
	}
}

func TestKMeansSelectsBestEligibleCandidate(t *testing.T) {
	s := loadArtifactSelector(t, "kmeans", "testdata")
	ctx := &SelectionContext{QueryEmbedding: []float64{0.25, 0.25}, CategoryName: "math"}

	// The v1 artifact scores its cluster's model 1 and the rest 0, so model-b is the best present.
	got, err := s.Select(ctx, []config.ModelRef{{Model: "foreign"}, {Model: "model-b"}})
	if err != nil || got.Model != "model-b" {
		t.Fatalf("missing top candidate: selected %v, %v; want model-b", got, err)
	}
	got, err = s.Select(ctx, []config.ModelRef{{Model: "foreign-a"}, {Model: "foreign-b"}})
	if !errors.Is(err, ErrNoEligibleCandidate) || got != nil {
		t.Fatalf("no eligible candidate: got %v, %v; want ErrNoEligibleCandidate", got, err)
	}
}

func TestKMeansRejectsInvalidV2ArtifactAtLoad(t *testing.T) {
	for _, reject := range loadKMeansFixtures(t).ArtifactRejects {
		t.Run(reject.Reason, func(t *testing.T) {
			s := NewKMeansSelector(8)
			defer func() { _ = s.Close() }()
			if err := s.LoadFromJSON(reject.Artifact); err == nil {
				t.Fatalf("%s artifact loaded", reject.Reason)
			}
		})
	}
}
