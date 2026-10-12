package modelselection

import (
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// Selections recorded from ml-binding (KNN, KMeans, SVM) and the candle MLP
// before they were removed. An empty selection means the binding failed.
type selectionFixtures struct {
	Models []struct {
		Name       string      `json:"name"`
		Algorithm  string      `json:"algorithm"`
		Model      string      `json:"model"`
		ModelFile  string      `json:"model_file"`
		SHA256     string      `json:"sha256"`
		Seed       int64       `json:"seed"`
		Queries    [][]float64 `json:"queries"`
		QueryCount int         `json:"query_count"`
		Selections []string    `json:"selections"`
		LatencyNs  int64       `json:"latency_ns"`
	} `json:"models"`
}

func loadSelectionFixtures(t testing.TB, name string) selectionFixtures {
	t.Helper()
	file, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	reader, err := gzip.NewReader(file)
	if err != nil {
		t.Fatal(err)
	}
	var fixtures selectionFixtures
	if err := json.NewDecoder(reader).Decode(&fixtures); err != nil {
		t.Fatal(err)
	}
	return fixtures
}

func classifierFor(t testing.TB, algorithm string, artifact []byte) classifier {
	t.Helper()
	parse := map[string]func([]byte) (classifier, error){"knn": parseKNN, "kmeans": parseKMeans, "svm": parseSVM, "mlp": parseMLP}[algorithm]
	model, err := parse(artifact)
	if err != nil {
		t.Fatalf("%s: %v", algorithm, err)
	}
	return model
}

func checkSelections(t *testing.T, name string, model classifier, queries [][]float64, want []string) {
	t.Helper()
	if len(queries) != len(want) {
		t.Fatalf("%s: %d queries for %d selections", name, len(queries), len(want))
	}
	for i, query := range queries {
		got, err := model.classify(query)
		if err != nil {
			got = ""
		}
		if got != want[i] {
			t.Errorf("%s query %d: selected %q, binding selected %q", name, i, got, want[i])
		}
	}
}

func TestSelectionsMatchBindingFixtures(t *testing.T) {
	for _, fixture := range loadSelectionFixtures(t, "selection_binding_fixtures.json.gz").Models {
		artifact := []byte(fixture.Model)
		if fixture.ModelFile != "" {
			var err error
			if artifact, err = os.ReadFile(filepath.Join("testdata", fixture.ModelFile)); err != nil {
				t.Fatal(err)
			}
		}
		checkSHA256(t, fixture.Name, artifact, fixture.SHA256)
		checkSelections(t, fixture.Name, classifierFor(t, fixture.Algorithm, artifact), fixture.Queries, fixture.Selections)
	}
}

// The published artifacts are large; their queries are regenerated from the
// recorded seed and the artifact itself.
func TestPublishedSelectionsMatchBindingFixtures(t *testing.T) {
	dir := publishedModelsDir(t)
	for _, fixture := range loadSelectionFixtures(t, "selection_published_fixtures.json.gz").Models {
		artifact, err := os.ReadFile(filepath.Join(dir, fixture.ModelFile))
		if err != nil {
			t.Fatal(err)
		}
		checkSHA256(t, fixture.Name, artifact, fixture.SHA256)
		queries := queriesFor(t, artifact, fixture.Seed, fixture.QueryCount)
		checkSelections(t, fixture.Name, classifierFor(t, fixture.Algorithm, artifact), queries, fixture.Selections)
	}
}

// BenchmarkPublishedSelection measures selection on the published artifacts
// over the fixture queries; the binding's latency on the same queries is in
// the fixtures (latency_ns).
func BenchmarkPublishedSelection(b *testing.B) {
	dir := publishedModelsDir(b)
	for _, fixture := range loadSelectionFixtures(b, "selection_published_fixtures.json.gz").Models {
		artifact, err := os.ReadFile(filepath.Join(dir, fixture.ModelFile))
		if err != nil {
			b.Fatal(err)
		}
		model := classifierFor(b, fixture.Algorithm, artifact)
		queries := queriesFor(b, artifact, fixture.Seed, fixture.QueryCount)
		b.Run(fixture.Algorithm, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _ = model.classify(queries[i%len(queries)])
			}
			b.ReportMetric(float64(fixture.LatencyNs), "binding-ns/op")
			b.ReportMetric(float64(time.Duration(fixture.LatencyNs))/(float64(b.Elapsed().Nanoseconds())/float64(b.N)), "speedup")
		})
	}
}

func publishedModelsDir(t testing.TB) string {
	t.Helper()
	dir, err := pretrainedTestModelsDir()
	if err != nil {
		t.Fatal(err)
	}
	if dir == "" {
		t.Skip("set " + pretrainedTestModelsEnv + " to compare the published artifacts")
	}
	return dir
}

func checkSHA256(t testing.TB, name string, artifact []byte, want string) {
	t.Helper()
	digest := sha256.Sum256(artifact)
	if got := hex.EncodeToString(digest[:]); got != want {
		t.Fatalf("%s: artifact sha256 %s differs from the recorded %s", name, got, want)
	}
}

// queriesFor reproduces the recorder's queries: half random unit directions,
// half model points with Gaussian noise (σ = 0.05), and every tenth an exact
// model point.
func queriesFor(t testing.TB, artifact []byte, seed int64, count int) [][]float64 {
	t.Helper()
	points, dim := modelPoints(t, artifact)
	rng := rand.New(rand.NewSource(seed))
	queries := make([][]float64, 0, count)
	for i := 0; i < count; i++ {
		query := make([]float64, dim)
		if i%2 == 0 || len(points) == 0 {
			var norm float64
			for j := range query {
				query[j] = rng.NormFloat64()
				norm += float64(query[j] * query[j])
			}
			norm = math.Sqrt(norm)
			for j := range query {
				query[j] /= norm
			}
		} else {
			point := points[rng.Intn(len(points))]
			for j := range query {
				query[j] = point[j] + float64(0.05*rng.NormFloat64())
			}
		}
		if i%10 == 9 && len(points) > 0 {
			copy(query, points[rng.Intn(len(points))])
		}
		queries = append(queries, query)
	}
	return queries
}

func modelPoints(t testing.TB, artifact []byte) ([][]float64, int) {
	t.Helper()
	var parsed struct {
		Embeddings [][]float64 `json:"embeddings"`
		Centroids  [][]float64 `json:"centroids"`
		FeatureDim int         `json:"feature_dim"`
		Support    [][]float64 `json:"support_vectors"`
		Svc        *struct {
			Support [][]float64 `json:"support_vectors"`
		} `json:"svc"`
		Linear []struct {
			Weights []float64 `json:"weights"`
		} `json:"linear_classifiers"`
		RBF []struct {
			Support [][]float64 `json:"support_vectors"`
		} `json:"rbf_classifiers"`
	}
	if err := json.Unmarshal(artifact, &parsed); err != nil {
		t.Fatal(err)
	}
	var points [][]float64
	switch {
	case len(parsed.Embeddings) > 0:
		points = parsed.Embeddings
	case len(parsed.Centroids) > 0:
		points = parsed.Centroids
	case parsed.Svc != nil:
		points = parsed.Svc.Support
	case len(parsed.Support) > 0:
		points = parsed.Support
	}
	for _, c := range parsed.RBF {
		points = append(points, c.Support...)
	}
	for _, c := range parsed.Linear {
		points = append(points, c.Weights)
	}
	dim := parsed.FeatureDim
	if dim == 0 && len(points) > 0 {
		dim = len(points[0])
	}
	if len(points) > 512 {
		step := len(points) / 512
		sampled := make([][]float64, 0, 512)
		for i := 0; i < len(points) && len(sampled) < 512; i += step {
			sampled = append(sampled, points[i])
		}
		points = sampled
	}
	return points, dim
}
