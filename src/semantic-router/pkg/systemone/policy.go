package systemone

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type policyAction struct {
	Weights []float64 `json:"weights"`
	CostMS  float64   `json:"training_mean_cost_ms"`
}

// PolicyActionBinding ties fitted action coefficients to the provider alias
// and observed native inference identity. Repointing an alias does not transfer
// evidence to a different model, revision or numerical profile.
type PolicyActionBinding struct {
	Model    string            `json:"model"`
	Identity InferenceIdentity `json:"identity"`
}

// LearnedPolicy contains data-only fitted coefficients. The operator's YAML
// remains the authority for stages, models, generation settings and budgets.
type LearnedPolicy struct {
	SchemaVersion string                             `json:"schema_version"`
	FeatureNames  []string                           `json:"feature_names"`
	Heads         map[string]map[string]policyAction `json:"heads"`
	Actions       map[string]PolicyActionBinding     `json:"actions"`
	StopValue     float64                            `json:"stop_value"`
	Training      map[string]json.RawMessage         `json:"training"`
}

func readArtifact(path, digest string, result any) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open inference artifact: %w", err)
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, (4<<20)+1))
	if err != nil || len(data) > 4<<20 {
		return errors.New("inference artifact exceeds limit or cannot be read")
	}
	sum := sha256.Sum256(data)
	if !strings.EqualFold(hex.EncodeToString(sum[:]), digest) {
		return errors.New("inference artifact digest mismatch")
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(result); err != nil {
		return fmt.Errorf("invalid inference artifact: %w", err)
	}
	if decoder.Decode(new(any)) != io.EOF {
		return errors.New("inference artifact has trailing data")
	}
	return nil
}

// LoadPolicy verifies the bytes and every action against the authored stages.
// It is called when preparing a generation, never on each request.
func LoadPolicy(spec *config.PolicyAlgorithmConfig, stages []config.CascadeStage) (*LearnedPolicy, error) {
	if spec == nil {
		return nil, errors.New("missing learned policy")
	}
	var p LearnedPolicy
	if err := readArtifact(spec.Source, spec.SHA256, &p); err != nil {
		return nil, err
	}
	if p.SchemaVersion != "systemone-policy/v1" || !slices.Equal(p.FeatureNames, FeatureNames) || p.StopValue != 0 || len(p.Heads) == 0 {
		return nil, errors.New("unsupported inference policy schema or features")
	}
	declared := make(map[string]config.CascadeStage, len(stages))
	for _, stage := range stages {
		declared[stage.Name] = stage
		if stage.Kind == "native" {
			binding, ok := p.Actions[stage.Name]
			if !ok || binding.Model != stage.Model || !binding.Identity.valid() {
				return nil, fmt.Errorf("policy stage %q has no matching immutable native identity", stage.Name)
			}
		}
	}
	for name := range p.Actions {
		if stage, ok := declared[name]; !ok || stage.Kind != "native" {
			return nil, fmt.Errorf("policy identity %q is not a declared native stage", name)
		}
	}
	for from, actions := range p.Heads {
		stage, ok := declared[from]
		if !ok || stage.Kind != "native" {
			return nil, fmt.Errorf("policy source %q is not a native stage", from)
		}
		for to, action := range actions {
			stage, ok := declared[to]
			if !ok || stage.Kind != "native" || to == from || len(action.Weights) != len(FeatureNames) || !finite(action.CostMS) || action.CostMS <= 0 {
				return nil, fmt.Errorf("invalid inference policy action %q to %q", from, to)
			}
			for _, weight := range action.Weights {
				if !finite(weight) {
					return nil, errors.New("non-finite policy weight")
				}
			}
		}
	}
	return &p, nil
}

func (p *LearnedPolicy) matches(stage config.CascadeStage, candidate Candidate) bool {
	identity, ok := responseInferenceIdentity(candidate.Body)
	binding, declared := p.Actions[stage.Name]
	return ok && declared && binding.Model == stage.Model && identity == binding.Identity
}

// Next chooses a positive estimated gain among remaining declared actions.
// This predicts incremental loss reduction; it is not quality certification.
func (p *LearnedPolicy) Next(from string, features []float64, stages []config.CascadeStage, attempted map[string]bool, costWeight float64) int {
	if p == nil || len(features) != len(FeatureNames) {
		return -1
	}
	best, value := -1, 0.0
	for i, stage := range stages {
		if attempted[stage.Name] || !stage.IsEnabled() {
			continue
		}
		action, ok := p.Heads[from][stage.Name]
		if !ok {
			continue
		}
		gain := -costWeight * action.CostMS
		for j, feature := range features {
			gain += feature * action.Weights[j]
		}
		if finite(gain) && gain > value {
			best, value = i, gain
		}
	}
	return best
}
