//go:build record_binding_fixtures

// Package bindingfixtures records model-selection fixtures from ml-binding
// (KNN, KMeans, SVM) and the candle MLP before the bindings are removed. It is
// temporary: the fixtures it writes are the parity reference for the pure-Go
// selectors.
//
//	go test -tags record_binding_fixtures ./pkg/modelselection/internal/bindingfixtures \
//	  -run TestRecordSelectionFixtures -args -out <dir> -testdata ../../testdata -pretrained <dir>
package bindingfixtures

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	ml_binding "github.com/vllm-project/semantic-router/ml-binding"
)

var (
	outDir     = flag.String("out", "", "fixture output directory")
	testdata   = flag.String("testdata", "", "hand-authored model directory")
	pretrained = flag.String("pretrained", "", "published model directory")
	queryCount = flag.Int("queries", 400, "queries per pretrained model")
)

type modelFixture struct {
	Name       string      `json:"name"`
	Algorithm  string      `json:"algorithm"`
	Model      string      `json:"model,omitempty"`
	ModelFile  string      `json:"model_file,omitempty"`
	SHA256     string      `json:"sha256"`
	Seed       int64       `json:"seed"`
	Queries    [][]float64 `json:"queries,omitempty"`
	QueryCount int         `json:"query_count"`
	Selections []string    `json:"selections"`
	// LatencyNs is the binding's mean select latency (one warm process, CPU).
	LatencyNs int64 `json:"latency_ns"`
}

type fixtureFile struct {
	Binding string         `json:"binding"`
	Note    string         `json:"note"`
	Models  []modelFixture `json:"models"`
}

type selector interface {
	Select([]float64) (string, error)
}

func TestRecordSelectionFixtures(t *testing.T) {
	if *outDir == "" {
		t.Skip("-out is required")
	}
	inline := fixtureFile{Binding: "ml-binding (linfa-nn 0.7.2) and candle-binding MLP (CPU, f32)", Note: "Queries are inline; an empty selection means the binding returned an error."}
	for _, model := range syntheticModels() {
		inline.Models = append(inline.Models, record(t, model.name, model.algorithm, []byte(model.json), "", int64(len(inline.Models)+1), 64, true))
	}
	if *testdata != "" {
		for _, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
			path := filepath.Join(*testdata, algorithm+"_model.json")
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			inline.Models = append(inline.Models, record(t, "testdata-"+algorithm, algorithm, data, algorithm+"_model.json", 100+int64(len(inline.Models)), 64, true))
		}
	}
	write(t, filepath.Join(*outDir, "selection_binding_fixtures.json"), inline)

	if *pretrained != "" {
		published := fixtureFile{Binding: inline.Binding, Note: "Queries are regenerated from seed and the model file (see queriesFor)."}
		for i, algorithm := range []string{"knn", "kmeans", "svm", "mlp"} {
			data, err := os.ReadFile(filepath.Join(*pretrained, algorithm+"_model.json"))
			if err != nil {
				t.Fatal(err)
			}
			published.Models = append(published.Models, record(t, "published-"+algorithm, algorithm, data, algorithm+"_model.json", 1000+int64(i), *queryCount, false))
		}
		write(t, filepath.Join(*outDir, "selection_published_fixtures.json"), published)
	}
}

func record(t *testing.T, name, algorithm string, model []byte, file string, seed int64, count int, inlineQueries bool) modelFixture {
	digest := sha256.Sum256(model)
	s, closeFn := load(t, algorithm, model)
	defer closeFn()
	queries := queriesFor(algorithm, model, seed, count)
	fixture := modelFixture{Name: name, Algorithm: algorithm, ModelFile: file, SHA256: hex.EncodeToString(digest[:]), Seed: seed, QueryCount: len(queries)}
	if file == "" {
		fixture.Model = string(model)
	}
	if inlineQueries {
		fixture.Queries = queries
	}
	for _, query := range queries {
		selected, err := s.Select(query)
		if err != nil {
			selected = ""
		}
		fixture.Selections = append(fixture.Selections, selected)
	}
	// Mean latency over the same queries after the selections above warmed the process.
	start := time.Now()
	rounds := 0
	for time.Since(start) < 2*time.Second || rounds < 3 {
		for _, query := range queries {
			_, _ = s.Select(query)
		}
		rounds++
	}
	fixture.LatencyNs = time.Since(start).Nanoseconds() / int64(rounds*len(queries))
	t.Logf("%s: %d queries, mean %v per select", name, len(queries), time.Duration(fixture.LatencyNs))
	return fixture
}

func load(t *testing.T, algorithm string, model []byte) (selector, func()) {
	switch algorithm {
	case "knn":
		s, err := ml_binding.KNNFromJSON(string(model))
		if err != nil {
			t.Fatal(err)
		}
		return s, s.Close
	case "kmeans":
		s, err := ml_binding.KMeansFromJSON(string(model))
		if err != nil {
			t.Fatal(err)
		}
		return s, s.Close
	case "svm":
		s, err := ml_binding.SVMFromJSON(string(model))
		if err != nil {
			t.Fatal(err)
		}
		return s, s.Close
	case "mlp":
		s, err := candle_binding.MLPFromJSON(string(model))
		if err != nil {
			t.Fatal(err)
		}
		return s, s.Close
	}
	t.Fatalf("unknown algorithm %s", algorithm)
	return nil, nil
}

func write(t *testing.T, path string, file fixtureFile) {
	data, err := json.MarshalIndent(file, "", " ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

// queriesFor derives queries from a seed and the model's own points: half are
// random directions, half are model points with small noise, so both regions
// far from and near the decision boundaries are covered. The pure-Go parity
// test regenerates them with the same function.
func queriesFor(algorithm string, model []byte, seed int64, count int) [][]float64 {
	points, dim := modelPoints(algorithm, model)
	rng := rand.New(rand.NewSource(seed))
	queries := make([][]float64, 0, count)
	for i := 0; i < count; i++ {
		query := make([]float64, dim)
		if i%2 == 0 || len(points) == 0 {
			var norm float64
			for j := range query {
				query[j] = rng.NormFloat64()
				norm += query[j] * query[j]
			}
			norm = math.Sqrt(norm)
			for j := range query {
				query[j] /= norm
			}
		} else {
			point := points[rng.Intn(len(points))]
			for j := range query {
				query[j] = point[j] + 0.05*rng.NormFloat64()
			}
		}
		if i%10 == 9 && len(points) > 0 {
			copy(query, points[rng.Intn(len(points))]) // exact model points: distance ties
		}
		queries = append(queries, query)
	}
	return queries
}

func modelPoints(algorithm string, model []byte) ([][]float64, int) {
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
	if err := json.Unmarshal(model, &parsed); err != nil {
		panic(err)
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
		// Bound the sample without depending on map order.
		step := len(points) / 512
		sampled := make([][]float64, 0, 512)
		for i := 0; i < len(points) && len(sampled) < 512; i += step {
			sampled = append(sampled, points[i])
		}
		points = sampled
	}
	return points, dim
}

type synthetic struct {
	name      string
	algorithm string
	json      string
}

func syntheticModels() []synthetic {
	rng := rand.New(rand.NewSource(42))
	gauss := func(rows, cols int, scale float64) [][]float64 {
		m := make([][]float64, rows)
		for i := range m {
			m[i] = make([]float64, cols)
			for j := range m[i] {
				m[i][j] = scale * rng.NormFloat64()
			}
		}
		return m
	}
	vec := func(n int, scale float64) []float64 { return gauss(1, n, scale)[0] }
	labels := func(n int, names ...string) []string {
		out := make([]string, n)
		for i := range out {
			out[i] = names[rng.Intn(len(names))]
		}
		return out
	}
	encode := func(v interface{}) string {
		data, err := json.Marshal(v)
		if err != nil {
			panic(err)
		}
		return string(data)
	}
	var models []synthetic

	// KNN: 3 classes, duplicates (exact distance ties), k beyond n, lexicographic score ties.
	embeddings := gauss(40, 12, 1)
	embeddings = append(embeddings, embeddings[0], embeddings[1], embeddings[2])
	qualities := make([]float64, len(embeddings))
	latencies := make([]int64, len(embeddings))
	for i := range qualities {
		qualities[i] = math.Round(rng.Float64()*100) / 100
		latencies[i] = int64(rng.Intn(5_000_000_000))
	}
	knnLabels := labels(len(embeddings), "model-c", "model-a", "model-b")
	for _, k := range []int{1, 3, 7, 100} {
		models = append(models, synthetic{fmt.Sprintf("knn-k%d", k), "knn", encode(map[string]interface{}{
			"algorithm": "knn", "format_version": 2, "trained": true, "k": k, "embeddings": embeddings,
			"labels": knnLabels, "qualities": qualities, "latencies": latencies,
		})})
	}
	tied := gauss(6, 4, 1)
	models = append(models, synthetic{"knn-equal-votes", "knn", encode(map[string]interface{}{
		"algorithm": "knn", "trained": true, "k": 2, "embeddings": [][]float64{tied[0], tied[0], tied[1], tied[1]},
		"labels": []string{"zeta", "alpha", "beta", "beta"},
	})})

	// KMeans.
	models = append(models, synthetic{"kmeans-10", "kmeans", encode(map[string]interface{}{
		"algorithm": "kmeans", "trained": true, "num_clusters": 10, "centroids": gauss(10, 16, 1),
		"cluster_models": labels(10, "model-a", "model-b", "model-c", "model-d"),
	})})

	// Exact SVC parameters: multiclass RBF and linear, binary RBF.
	svc := func(classes, dim, perClass int, kernel string, gamma float64) string {
		n := classes * perClass
		nSupport := make([]int, classes)
		for i := range nSupport {
			nSupport[i] = perClass
		}
		names := make([]string, classes)
		for i := range names {
			names[i] = fmt.Sprintf("model-%c", 'a'+i)
		}
		return encode(map[string]interface{}{
			"algorithm": "svm", "format_version": 2, "trained": true, "feature_dim": dim,
			"input_normalization": "none", "model_names": names, "kernel_type": kernel, "gamma": gamma,
			"svc": map[string]interface{}{
				"support_vectors": gauss(n, dim, 1), "dual_coef": gauss(classes-1, n, 1),
				"intercept": vec(classes*(classes-1)/2, 0.5), "n_support": nSupport,
			},
		})
	}
	models = append(models,
		synthetic{"svc-rbf-4class", "svm", svc(4, 10, 6, "Rbf", 0.2)},
		synthetic{"svc-linear-3class", "svm", svc(3, 10, 5, "Linear", 1)},
		synthetic{"svc-rbf-binary", "svm", svc(2, 8, 7, "Rbf", 0.5)},
		synthetic{"svc-linear-binary", "svm", svc(2, 8, 4, "Linear", 1)},
	)

	// Legacy one-vs-rest artifacts (no exact SVC parameters).
	var linear, rbf []map[string]interface{}
	for _, name := range []string{"model-a", "model-b", "model-c"} {
		linear = append(linear, map[string]interface{}{"model_name": name, "weights": vec(9, 1), "rho": rng.NormFloat64() * 0.1})
		rbf = append(rbf, map[string]interface{}{"model_name": name, "alpha": vec(5, 1), "support_vectors": gauss(5, 9, 0.5), "rho": rng.NormFloat64() * 0.1, "gamma": 0.7})
	}
	models = append(models,
		synthetic{"ovr-linear", "svm", encode(map[string]interface{}{"algorithm": "svm", "trained": true, "model_names": []string{"model-a", "model-b", "model-c"}, "kernel_type": "Linear", "gamma": 1, "linear_classifiers": linear})},
		synthetic{"ovr-rbf", "svm", encode(map[string]interface{}{"algorithm": "svm", "trained": true, "model_names": []string{"model-a", "model-b", "model-c"}, "kernel_type": "Rbf", "gamma": 0.7, "rbf_classifiers": rbf})},
	)

	// MLP: linear, batch norm, relu and dropout layers; a zero head for argmax ties.
	linearLayer := func(in, out int, bias bool) map[string]interface{} {
		layer := map[string]interface{}{"type": "linear", "in_features": in, "out_features": out, "weight": gauss(out, in, 1/math.Sqrt(float64(in)))}
		if bias {
			layer["bias"] = vec(out, 0.1)
		}
		return layer
	}
	batchNorm := func(n int) map[string]interface{} {
		variance := vec(n, 0.3)
		for i := range variance {
			variance[i] = math.Abs(variance[i]) + 0.5
		}
		return map[string]interface{}{"type": "batch_norm", "num_features": n, "weight": vec(n, 1), "bias": vec(n, 0.1), "running_mean": vec(n, 0.2), "running_var": variance, "eps": 1e-5}
	}
	models = append(models, synthetic{"mlp-deep", "mlp", encode(map[string]interface{}{
		"algorithm": "mlp", "trained": true, "model_names": []string{"model-a", "model-b", "model-c", "model-d"},
		"feature_dim": 20, "n_classes": 4, "hidden_sizes": []int{32, 16}, "dropout": 0.1,
		"layers": []interface{}{
			linearLayer(20, 32, true), batchNorm(32), map[string]interface{}{"type": "relu"}, map[string]interface{}{"type": "dropout", "p": 0.1},
			linearLayer(32, 16, false), map[string]interface{}{"type": "batch_norm", "num_features": 16}, map[string]interface{}{"type": "relu"},
			linearLayer(16, 4, true),
		},
	})})
	zero := linearLayer(6, 3, false)
	zero["weight"] = [][]float64{make([]float64, 6), make([]float64, 6), make([]float64, 6)}
	models = append(models, synthetic{"mlp-argmax-ties", "mlp", encode(map[string]interface{}{
		"algorithm": "mlp", "trained": true, "model_names": []string{"model-a", "model-b", "model-c"},
		"feature_dim": 6, "n_classes": 3, "hidden_sizes": []int{}, "dropout": 0, "layers": []interface{}{zero},
	})})
	sort.SliceStable(models, func(i, j int) bool { return models[i].algorithm < models[j].algorithm })
	return models
}
