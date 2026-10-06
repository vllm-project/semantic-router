package modelselection

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"
)

// svmModel is an SVM artifact: the exact libsvm one-vs-one parameters
// (sklearn SVC), or, for artifacts that predate them, one-vs-rest
// classifiers scored on the normalized query.
type svmModel struct {
	names     []string
	dim       int
	kernelRBF bool
	gamma     float64
	svc       *svcModel
	ovr       []ovrClassifier
}

// svcModel uses sklearn's public layout: for the class pair (i, j), row j-1
// of the dual coefficients weighs class i's support vectors and row i weighs
// class j's.
type svcModel struct {
	supportVectors []float64 // vectors × dim, row-major
	dualCoef       [][]float64
	intercept      []float64
	starts         []int // first support vector of each class, plus the total
}

type ovrClassifier struct {
	name           int
	weights        []float64 // linear
	alpha          []float64 // RBF
	supportVectors []float64 // RBF, vectors × dim
	gamma          float64
	rho            float64
}

type svcArtifact struct {
	SupportVectors [][]float64 `json:"support_vectors"`
	DualCoef       [][]float64 `json:"dual_coef"`
	Intercept      []float64   `json:"intercept"`
	NSupport       []int       `json:"n_support"`
}

func parseSVM(data []byte) (classifier, error) {
	var artifact struct {
		Algorithm          string       `json:"algorithm"`
		FormatVersion      *int         `json:"format_version"`
		FeatureDim         int          `json:"feature_dim"`
		InputNormalization *string      `json:"input_normalization"`
		Svc                *svcArtifact `json:"svc"`
		Trained            bool         `json:"trained"`
		ModelNames         []string     `json:"model_names"`
		KernelType         string       `json:"kernel_type"`
		Gamma              float64      `json:"gamma"`
		LinearClassifiers  []struct {
			ModelName string    `json:"model_name"`
			Weights   []float64 `json:"weights"`
			Rho       float64   `json:"rho"`
		} `json:"linear_classifiers"`
		RBFClassifiers []struct {
			ModelName      string      `json:"model_name"`
			Alpha          []float64   `json:"alpha"`
			SupportVectors [][]float64 `json:"support_vectors"`
			Rho            float64     `json:"rho"`
			Gamma          float64     `json:"gamma"`
		} `json:"rbf_classifiers"`
		// Older exports keep the exact parameters at the top level.
		LegacySupportVectors json.RawMessage `json:"support_vectors"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, err
	}
	if artifact.Algorithm != "svm" || (artifact.FormatVersion != nil && *artifact.FormatVersion != 1 && *artifact.FormatVersion != 2) {
		return nil, errors.New("unsupported SVM artifact format")
	}
	if artifact.KernelType != "Linear" && artifact.KernelType != "Rbf" {
		return nil, fmt.Errorf("unsupported SVM kernel %q", artifact.KernelType)
	}
	names := make(map[string]int, len(artifact.ModelNames))
	for i, name := range artifact.ModelNames {
		names[name] = i
	}
	if len(artifact.ModelNames) == 0 || len(names) != len(artifact.ModelNames) {
		return nil, errors.New("SVM model names must be nonempty and unique")
	}
	if math.IsNaN(artifact.Gamma) || math.IsInf(artifact.Gamma, 0) || artifact.Gamma <= 0 {
		return nil, errors.New("SVM gamma must be finite and positive")
	}
	if !artifact.Trained {
		return nil, errors.New("artifact is not trained")
	}
	model := &svmModel{names: artifact.ModelNames, dim: artifact.FeatureDim, kernelRBF: artifact.KernelType == "Rbf", gamma: artifact.Gamma}

	svc := artifact.Svc
	if svc == nil && len(artifact.LegacySupportVectors) > 0 && string(artifact.LegacySupportVectors) != "null" {
		svc = &svcArtifact{}
		if err := json.Unmarshal(data, svc); err != nil {
			return nil, fmt.Errorf("invalid legacy SVC parameters: %w", err)
		}
	}
	if svc != nil {
		if artifact.InputNormalization != nil && *artifact.InputNormalization != "none" {
			return nil, errors.New("unsupported SVM input normalization")
		}
		exact, err := newSVC(svc, len(artifact.ModelNames), model.dim)
		if err != nil {
			return nil, err
		}
		model.svc = exact
		return model, nil
	}
	if artifact.FormatVersion != nil && *artifact.FormatVersion == 2 {
		return nil, errors.New("SVM version 2 requires exact SVC parameters")
	}
	for _, c := range artifact.LinearClassifiers {
		if model.dim == 0 {
			model.dim = len(c.Weights)
		}
		name, known := names[c.ModelName]
		if model.dim == 0 || len(c.Weights) != model.dim || !finite(c.Weights) || !finite([]float64{c.Rho}) || !known {
			return nil, errors.New("invalid linear SVM classifier")
		}
		model.ovr = append(model.ovr, ovrClassifier{name: name, weights: c.Weights, rho: c.Rho})
	}
	for _, c := range artifact.RBFClassifiers {
		dim := 0
		if len(c.SupportVectors) > 0 {
			dim = len(c.SupportVectors[0])
		}
		if model.dim == 0 {
			model.dim = dim
		}
		name, known := names[c.ModelName]
		if len(c.SupportVectors) == 0 || dim == 0 || dim != model.dim || len(c.Alpha) != len(c.SupportVectors) ||
			!finite(c.Alpha) || !finite([]float64{c.Rho, c.Gamma}) || c.Gamma <= 0 || !known {
			return nil, errors.New("invalid RBF SVM classifier")
		}
		classifier := ovrClassifier{name: name, alpha: c.Alpha, gamma: c.Gamma, rho: c.Rho}
		for _, vector := range c.SupportVectors {
			if len(vector) != dim || !finite(vector) {
				return nil, errors.New("invalid RBF SVM classifier")
			}
			classifier.supportVectors = append(classifier.supportVectors, vector...)
		}
		model.ovr = append(model.ovr, classifier)
	}
	return model, nil
}

func newSVC(artifact *svcArtifact, classes, dim int) (*svcModel, error) {
	invalid := errors.New("invalid SVM parameter shapes or values")
	n := len(artifact.SupportVectors)
	if classes < 2 || dim == 0 || n == 0 || len(artifact.DualCoef) != classes-1 ||
		len(artifact.Intercept) != classes*(classes-1)/2 || !finite(artifact.Intercept) || len(artifact.NSupport) != classes {
		return nil, invalid
	}
	model := &svcModel{dualCoef: artifact.DualCoef, intercept: artifact.Intercept, starts: make([]int, classes+1)}
	for i, count := range artifact.NSupport {
		if count < 0 {
			return nil, invalid
		}
		model.starts[i+1] = model.starts[i] + count
	}
	if model.starts[classes] != n {
		return nil, invalid
	}
	for _, row := range artifact.DualCoef {
		if len(row) != n || !finite(row) {
			return nil, invalid
		}
	}
	model.supportVectors = make([]float64, 0, n*dim)
	for _, vector := range artifact.SupportVectors {
		if len(vector) != dim || !finite(vector) {
			return nil, invalid
		}
		model.supportVectors = append(model.supportVectors, vector...)
	}
	return model, nil
}

func (m *svmModel) classify(query []float64) (string, error) {
	if len(query) != m.dim || !finite(query) {
		return "", fmt.Errorf("expected %d finite SVM features, got %d", m.dim, len(query))
	}
	if m.svc != nil {
		return m.names[m.svc.vote(query, m.dim, m.kernelRBF, m.gamma)], nil
	}
	if len(m.ovr) == 0 {
		return m.names[0], nil
	}
	if n := norm(query); n > 1e-10 {
		scaled := make([]float64, len(query))
		for i, x := range query {
			scaled[i] = x / n
		}
		query = scaled
	}
	best, bestScore := -1, math.Inf(-1)
	for i := range m.ovr {
		if score := m.ovr[i].decision(query, m.dim); score > bestScore {
			best, bestScore = i, score
		}
	}
	if best < 0 {
		return m.names[0], nil
	}
	return m.names[m.ovr[best].name], nil
}

func (c *ovrClassifier) decision(query []float64, dim int) float64 {
	if c.weights != nil {
		return vecmath.Dot64(c.weights, query) - c.rho
	}
	var sum float64
	for i, alpha := range c.alpha {
		sum += alpha * math.Exp(-c.gamma*vecmath.SquaredDistance64(c.supportVectors[i*dim:(i+1)*dim], query))
	}
	return sum - c.rho
}

// vote runs libsvm's one-vs-one vote; equal vote counts go to the first class.
func (s *svcModel) vote(query []float64, dim int, rbf bool, gamma float64) int {
	vectors := len(s.supportVectors) / dim
	kernels := make([]float64, vectors)
	for i := range kernels {
		vector := s.supportVectors[i*dim : (i+1)*dim]
		if rbf {
			kernels[i] = math.Exp(-gamma * vecmath.SquaredDistance64(vector, query))
		} else {
			kernels[i] = vecmath.Dot64(vector, query)
		}
	}
	classes := len(s.starts) - 1
	votes := make([]int, classes)
	pair := 0
	for i := 0; i < classes; i++ {
		for j := i + 1; j < classes; j++ {
			var score float64
			for k := s.starts[i]; k < s.starts[i+1]; k++ {
				score += s.dualCoef[j-1][k] * kernels[k]
			}
			for k := s.starts[j]; k < s.starts[j+1]; k++ {
				score += s.dualCoef[i][k] * kernels[k]
			}
			score += s.intercept[pair]
			// sklearn reverses the public coefficients and intercept of a
			// binary SVC; multiclass ones keep libsvm's signs.
			winner := j
			if (classes == 2 && score < 0) || (classes > 2 && score > 0) {
				winner = i
			}
			votes[winner]++
			pair++
		}
	}
	winner := 0
	for i := 1; i < classes; i++ {
		if votes[i] > votes[winner] {
			winner = i
		}
	}
	return winner
}
