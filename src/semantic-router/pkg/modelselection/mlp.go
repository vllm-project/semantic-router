package modelselection

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding/vecmath"
)

// mlpModel is an MLP artifact evaluated in float32 like the candle model it
// replaces: linear layers, ReLU, batch normalization with running
// statistics, and dropout as the identity. Each output of a linear layer is
// one SIMD inner product over a contiguous weight row.
type mlpModel struct {
	names  []string
	dim    int
	layers []mlpLayer
}

type mlpLayer struct {
	kind string
	// linear: out × in weights, row-major
	in, out int
	weight  []float32
	bias    []float32
	// batch_norm
	scale, shift, mean, std []float32
}

type mlpLayerArtifact struct {
	Type        string          `json:"type"`
	InFeatures  int             `json:"in_features"`
	OutFeatures int             `json:"out_features"`
	Weight      json.RawMessage `json:"weight"`
	Bias        []float64       `json:"bias"`
	NumFeatures int             `json:"num_features"`
	RunningMean []float64       `json:"running_mean"`
	RunningVar  []float64       `json:"running_var"`
	Eps         *float64        `json:"eps"`
}

func parseMLP(data []byte) (classifier, error) {
	var artifact struct {
		Algorithm  string             `json:"algorithm"`
		Trained    bool               `json:"trained"`
		ModelNames []string           `json:"model_names"`
		FeatureDim int                `json:"feature_dim"`
		Layers     []mlpLayerArtifact `json:"layers"`
	}
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, err
	}
	if artifact.Algorithm != "mlp" {
		return nil, fmt.Errorf("invalid algorithm: expected 'mlp', got '%s'", artifact.Algorithm)
	}
	if !artifact.Trained {
		return nil, errors.New("artifact is not trained")
	}
	if len(artifact.ModelNames) == 0 || artifact.FeatureDim <= 0 {
		return nil, errors.New("MLP needs model names and a feature dimension")
	}
	model := &mlpModel{names: artifact.ModelNames, dim: artifact.FeatureDim}
	width := artifact.FeatureDim
	for i, layer := range artifact.Layers {
		compiled, err := compileLayer(layer, width)
		if err != nil {
			return nil, fmt.Errorf("layer %d (%s): %w", i, layer.Type, err)
		}
		if compiled.kind == "linear" {
			width = compiled.out
		}
		model.layers = append(model.layers, compiled)
	}
	return model, nil
}

func compileLayer(layer mlpLayerArtifact, width int) (mlpLayer, error) {
	switch layer.Type {
	case "relu", "dropout":
		return mlpLayer{kind: layer.Type}, nil
	case "linear":
		if layer.InFeatures != width || layer.OutFeatures <= 0 {
			return mlpLayer{}, fmt.Errorf("expects %d inputs, got in_features %d", width, layer.InFeatures)
		}
		var rows [][]float64
		if err := json.Unmarshal(layer.Weight, &rows); err != nil || len(rows) != layer.OutFeatures {
			return mlpLayer{}, errors.New("weight must be out_features rows")
		}
		compiled := mlpLayer{kind: "linear", in: layer.InFeatures, out: layer.OutFeatures, weight: make([]float32, 0, layer.InFeatures*layer.OutFeatures)}
		for _, row := range rows {
			if len(row) != layer.InFeatures {
				return mlpLayer{}, errors.New("weight rows must have in_features values")
			}
			compiled.weight = append(compiled.weight, toFloat32(row)...)
		}
		compiled.bias = make([]float32, layer.OutFeatures)
		if layer.Bias != nil {
			if len(layer.Bias) != layer.OutFeatures {
				return mlpLayer{}, errors.New("bias must have out_features values")
			}
			compiled.bias = toFloat32(layer.Bias)
		}
		return compiled, nil
	case "batch_norm":
		n := layer.NumFeatures
		if n != width {
			return mlpLayer{}, fmt.Errorf("expects %d features, got num_features %d", width, n)
		}
		var weight []float64
		if len(layer.Weight) > 0 && string(layer.Weight) != "null" {
			if err := json.Unmarshal(layer.Weight, &weight); err != nil {
				return mlpLayer{}, err
			}
		}
		scale, err := vectorOr(weight, n, 1)
		if err != nil {
			return mlpLayer{}, err
		}
		shift, err := vectorOr(layer.Bias, n, 0)
		if err != nil {
			return mlpLayer{}, err
		}
		mean, err := vectorOr(layer.RunningMean, n, 0)
		if err != nil {
			return mlpLayer{}, err
		}
		variance, err := vectorOr(layer.RunningVar, n, 1)
		if err != nil {
			return mlpLayer{}, err
		}
		eps := float32(1e-5)
		if layer.Eps != nil {
			eps = float32(*layer.Eps)
		}
		std := make([]float32, n)
		for i, v := range variance {
			std[i] = float32(math.Sqrt(float64(v + eps)))
		}
		return mlpLayer{kind: "batch_norm", scale: scale, shift: shift, mean: mean, std: std}, nil
	}
	return mlpLayer{}, fmt.Errorf("unknown layer type %q", layer.Type)
}

// vectorOr converts an optional per-feature parameter, defaulting to value.
func vectorOr(values []float64, n int, value float32) ([]float32, error) {
	if values == nil {
		out := make([]float32, n)
		for i := range out {
			out[i] = value
		}
		return out, nil
	}
	if len(values) != n {
		return nil, fmt.Errorf("batch-norm parameter has %d values, want %d", len(values), n)
	}
	return toFloat32(values), nil
}

func toFloat32(values []float64) []float32 {
	out := make([]float32, len(values))
	for i, v := range values {
		out[i] = float32(v)
	}
	return out
}

// classify runs the network; equal maximal outputs go to the last class.
func (m *mlpModel) classify(query []float64) (string, error) {
	if len(query) != m.dim {
		return "", fmt.Errorf("feature dimension mismatch: expected %d, got %d", m.dim, len(query))
	}
	x := toFloat32(query)
	for _, layer := range m.layers {
		switch layer.kind {
		case "linear":
			out := make([]float32, layer.out)
			for j := range out {
				out[j] = vecmath.Dot(layer.weight[j*layer.in:(j+1)*layer.in], x) + layer.bias[j]
			}
			x = out
		case "relu":
			for i, v := range x {
				x[i] = max(v, 0)
			}
		case "batch_norm":
			for i, v := range x {
				x[i] = float32(float32(float32(v-layer.mean[i])/layer.std[i])*layer.scale[i]) + layer.shift[i]
			}
		}
	}
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] >= x[best] {
			best = i
		}
	}
	if best >= len(m.names) {
		return "", fmt.Errorf("invalid class index: %d (have %d classes)", best, len(m.names))
	}
	return m.names[best], nil
}
