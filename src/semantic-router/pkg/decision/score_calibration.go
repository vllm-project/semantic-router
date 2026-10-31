package decision

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	scoreCalibrationSchema = "signal-calibration-artifact/v1"
	labelCorrectnessScale  = "label_correctness/v1"
	scoreCalibrationLimit  = 1 << 20
)

// ScoreCalibration maps one signal family's probability onto the declared
// label_correctness/v1 scale. It is built by tools/calibration/tuning from a
// held-out dataset and only ever loaded from a content-pinned reference.
type ScoreCalibration struct {
	Family     string
	ArtifactID string
	knots      [][2]float64
}

type scoreCalibrationArtifact struct {
	SchemaVersion string `json:"artifact_schema_version"`
	ArtifactID    string `json:"artifact_id"`
	Status        string `json:"status"`
	Family        string `json:"family"`
	Scale         string `json:"scale"`
	Model         struct {
		Labels      []string `json:"labels"`
		ModelSHA256 string   `json:"model_sha256"`
	} `json:"model"`
	Mapping struct {
		Knots [][2]float64 `json:"knots"`
	} `json:"mapping"`
}

// WithScoreCalibration declares which families report calibrated scores and
// supplies the loaded mapping for them.
func (e *DecisionEngine) WithScoreCalibration(families []string, calibration *ScoreCalibration) *DecisionEngine {
	if e != nil {
		e.calibrated = families
		e.calibration = calibration
	}
	return e
}

// LoadScoreCalibration verifies the artifact against its pinned digest, the
// family's ordered labels and modelSHA256, the identity the model runtime
// reports for the model it serves. Any mismatch is an error, so a missing or
// stale artifact stops startup instead of ranking.
func LoadScoreCalibration(ref config.ScoreCalibrationReference, family, modelSHA256 string, labels []string) (*ScoreCalibration, error) {
	data, err := readLimited(ref.Path)
	if err != nil {
		return nil, fmt.Errorf("calibration artifact: %w", err)
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != ref.SHA256 {
		return nil, fmt.Errorf("calibration artifact %s differs from its pinned sha256", ref.Path)
	}
	var artifact scoreCalibrationArtifact
	if err := json.Unmarshal(data, &artifact); err != nil {
		return nil, fmt.Errorf("calibration artifact %s: %w", ref.Path, err)
	}
	switch {
	case artifact.SchemaVersion != scoreCalibrationSchema:
		return nil, fmt.Errorf("calibration artifact schema %q is not %q", artifact.SchemaVersion, scoreCalibrationSchema)
	case artifact.Status != "calibrated":
		return nil, fmt.Errorf("calibration artifact status is %q, not calibrated", artifact.Status)
	case artifact.Family != family || artifact.Scale != labelCorrectnessScale:
		return nil, fmt.Errorf("calibration artifact maps %s onto %q, want %s onto %q", artifact.Family, artifact.Scale, family, labelCorrectnessScale)
	case !slices.Equal(artifact.Model.Labels, labels):
		return nil, fmt.Errorf("calibration artifact labels %v differ from the classifier's %v", artifact.Model.Labels, labels)
	case artifact.Model.ModelSHA256 == "":
		return nil, fmt.Errorf("calibration artifact does not bind a model_sha256")
	case artifact.Model.ModelSHA256 != modelSHA256:
		return nil, fmt.Errorf("calibration artifact was fitted on a different model: model_sha256 %s, served %q", artifact.Model.ModelSHA256, modelSHA256)
	}
	if err := validKnots(artifact.Mapping.Knots); err != nil {
		return nil, err
	}
	return &ScoreCalibration{Family: family, ArtifactID: artifact.ArtifactID, knots: artifact.Mapping.Knots}, nil
}

// Apply interpolates linearly between knots and clamps outside them. Knots rise
// strictly in score and never fall in value, so the order of scores holds.
func (c *ScoreCalibration) Apply(score float64) float64 {
	knots := c.knots
	if score <= knots[0][0] {
		return knots[0][1]
	}
	last := knots[len(knots)-1]
	if score >= last[0] {
		return last[1]
	}
	i := sort.Search(len(knots), func(i int) bool { return knots[i][0] >= score })
	lo, hi := knots[i-1], knots[i]
	return lo[1] + (score-lo[0])*(hi[1]-lo[1])/(hi[0]-lo[0])
}

func validKnots(knots [][2]float64) error {
	if len(knots) < 2 {
		return fmt.Errorf("calibration artifact needs at least two knots")
	}
	for i, knot := range knots {
		if knot[0] < 0 || knot[0] > 1 || knot[1] < 0 || knot[1] > 1 {
			return fmt.Errorf("calibration knot %d lies outside [0, 1]", i)
		}
		if i > 0 && (knot[0] <= knots[i-1][0] || knot[1] < knots[i-1][1]) {
			return fmt.Errorf("calibration knot %d does not rise from the one before it", i)
		}
	}
	return nil
}

func readLimited(path string) ([]byte, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, scoreCalibrationLimit+1))
	if err != nil {
		return nil, err
	}
	if len(data) > scoreCalibrationLimit {
		return nil, fmt.Errorf("%s exceeds %d bytes", path, scoreCalibrationLimit)
	}
	return data, nil
}
