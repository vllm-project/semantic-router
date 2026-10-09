package systemone

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"reflect"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// NativeCalibration is data-only evidence for a fixed policy and evaluated
// population. It is not an individual-request or out-of-distribution guarantee.
// Bounds are computed from independent source-bundle counts, never supplied risks.
type NativeCalibration struct {
	SchemaVersion string                   `json:"schema_version"`
	Loss          string                   `json:"loss"`
	Feature       string                   `json:"feature"`
	PolicySHA256  string                   `json:"policy_sha256"`
	Alpha         float64                  `json:"alpha"`
	Certification CalibrationCertification `json:"certification"`
	Cells         []CalibrationCell        `json:"cells"`
}

// CalibrationCertification records the producer's auditable sampling claim.
// The runtime can validate these declarations and artifact integrity; the
// independent dataset and label provenance still require an external audit.
type CalibrationCertification struct {
	Population                   string `json:"population"`
	SourceManifestSHA256         string `json:"source_manifest_sha256"`
	FrozenGatesSHA256            string `json:"frozen_gates_sha256"`
	Unit                         string `json:"unit"`
	Sampling                     string `json:"sampling"`
	PolicyFrozenBeforeCollection bool   `json:"policy_frozen_before_collection"`
	IndependentFromTraining      bool   `json:"independent_from_training"`
}

// CalibrationCell conditions on a fixed task and complete successful native
// arrival path. Error and judge histories require a later evidence protocol.
type CalibrationCell struct {
	TaskSHA256 string               `json:"task_sha256"`
	History    []CalibrationHistory `json:"history"`
	Bins       []CalibrationBin     `json:"bins"`
}

type CalibrationHistory struct {
	Stage    string            `json:"stage"`
	Model    string            `json:"model"`
	Identity InferenceIdentity `json:"identity"`
}

// CalibrationBin is [lower, upper), except upper=1 includes 1. Counts are
// independent source bundles; multiple questions or variants do not increase n.
type CalibrationBin struct {
	Lower  float64 `json:"lower"`
	Upper  float64 `json:"upper"`
	Accept bool    `json:"accept"`
	Count  int     `json:"count"`
	Errors int     `json:"errors"`
	risk   float64
}

// LoadCalibration prepares immutable applicability checks once per generation.
// No request performs file I/O. Missing runtime identity or an unmatched task,
// arrival history or score bin is unresolved, never an uncalibrated success.
func LoadCalibration(cfg *config.RouterConfig, recipe *config.RoutingRecipe, decision *config.Decision) (QualityEvaluator, error) {
	if decision == nil || !decision.Algorithm.IsNative() || decision.Algorithm.Quality == nil {
		return nil, errors.New("missing native quality configuration")
	}
	quality := decision.Algorithm.Quality
	if quality.Type != "calibrated" {
		return nil, nil
	}
	if quality.Loss != "bundle_error" || quality.MaxRisk == nil || !probability(quality.MaxRisk) {
		return nil, errors.New("calibration requires bundle_error and max_risk in [0, 1]")
	}
	if recipe == nil || !emptyCalibrationCollections(recipe.Profile.Signals) || !emptyCalibrationCollections(recipe.Profile.Projections) {
		return nil, errors.New("calibrated execution does not yet support routing signal or projection provenance")
	}
	for _, stage := range decision.Algorithm.Stages {
		if stage.IsEnabled() && stage.Kind != "native" {
			return nil, errors.New("calibrated execution does not yet support judge identity and history evidence")
		}
	}
	resource, ok := cfg.Calibration(quality.Calibration)
	if !ok {
		return nil, errors.New("unknown calibration resource")
	}
	var artifact NativeCalibration
	source := resource.Source
	if !filepath.IsAbs(source) {
		source = filepath.Join(cfg.ConfigBaseDir, source)
	}
	if err := readArtifact(source, resource.SHA256, &artifact); err != nil {
		return nil, err
	}
	policySHA, err := CalibrationPolicySHA256(recipe, decision)
	if err != nil {
		return nil, err
	}
	if err := artifact.prepare(policySHA, decision.Algorithm.Stages, *quality.MaxRisk); err != nil {
		return nil, err
	}
	maxRisk := *quality.MaxRisk
	return func(request *NativeRequest, candidate Candidate) (bool, error) {
		return artifact.accept(request, candidate, maxRisk), nil
	}, nil
}

func emptyCalibrationCollections(value any) bool {
	v := reflect.ValueOf(value)
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		if field.Kind() == reflect.Slice || field.Kind() == reflect.Map {
			if field.Len() != 0 {
				return false
			}
		} else if !field.IsZero() {
			return false
		}
	}
	return true
}

func (a *NativeCalibration) prepare(policySHA string, stages []config.CascadeStage, maxRisk float64) error {
	cert := a.Certification
	if a.SchemaVersion != "systemone-calibration/v1" || a.Loss != "bundle_error" || a.Feature != "min_top_probability" || a.PolicySHA256 != policySHA {
		return errors.New("unsupported or mismatched native calibration contract")
	}
	gateSHA, err := CalibrationGateSHA256(a.Cells)
	if err != nil || cert.FrozenGatesSHA256 != gateSHA || cert.Population == "" || !validSHA256(cert.SourceManifestSHA256) || cert.Unit != "independent_source_bundle" || cert.Sampling != "iid_source_bundles" || !cert.PolicyFrozenBeforeCollection || !cert.IndependentFromTraining {
		return errors.New("calibration requires i.i.d. source bundles collected after freezing the policy and bins")
	}
	if !finite(a.Alpha) || a.Alpha <= 0 || a.Alpha >= 1 || len(a.Cells) == 0 || len(stages) == 0 {
		return errors.New("calibration requires cells and alpha in (0, 1)")
	}
	declared := make(map[string]config.CascadeStage, len(stages))
	for _, stage := range stages {
		declared[stage.Name] = stage
	}
	seen, cells := map[string]bool{}, 0
	for _, cell := range a.Cells {
		if err := validateCalibrationCell(cell, declared, stages[0].Name); err != nil {
			return err
		}
		key, err := calibrationSHA(struct {
			Task    string
			History []CalibrationHistory
		}{cell.TaskSHA256, cell.History})
		if err != nil || seen[key] {
			return errors.New("duplicate or invalid calibration cell")
		}
		seen[key], cells = true, cells+len(cell.Bins)
	}
	// Bonferroni covers every predeclared cell, including cells a later request
	// does not visit. Reusing a source bundle across different cells is allowed;
	// each cell still needs independent units within its own count.
	alpha := a.Alpha / float64(cells)
	for i := range a.Cells {
		for j := range a.Cells[i].Bins {
			bin := &a.Cells[i].Bins[j]
			if bin.Count == 0 {
				bin.risk = 1
				continue
			}
			upper, err := calibrationRiskUpper(bin.Errors, bin.Count, alpha)
			if err != nil {
				return err
			}
			bin.risk = upper
			if bin.Accept && upper > maxRisk {
				return errors.New("a frozen acceptance bin exceeds max_risk; collect new evidence without changing certified arrival gates")
			}
		}
	}
	return nil
}

func validateCalibrationCell(cell CalibrationCell, stages map[string]config.CascadeStage, first string) error {
	if !validSHA256(cell.TaskSHA256) || len(cell.History) == 0 || len(cell.Bins) == 0 || cell.History[0].Stage != first {
		return errors.New("calibration cell requires a task, full arrival history and bins")
	}
	visited := map[string]bool{}
	for _, step := range cell.History {
		stage, exists := stages[step.Stage]
		if !exists || !stage.IsEnabled() || stage.Kind != "native" || stage.Model != step.Model || visited[step.Stage] || !step.Identity.Valid() {
			return fmt.Errorf("calibration contains an undeclared, repeated or unverified native stage %q", step.Stage)
		}
		visited[step.Stage] = true
	}
	upper := -1.0
	for _, bin := range cell.Bins {
		if !finite(bin.Lower) || !finite(bin.Upper) || bin.Lower < 0 || bin.Upper > 1 || bin.Lower >= bin.Upper || bin.Lower < upper || bin.Count < 0 || bin.Count > 1_000_000_000 || bin.Errors < 0 || bin.Errors > bin.Count || (bin.Accept && bin.Count == 0) {
			return errors.New("calibration requires ordered disjoint bins and valid independent counts")
		}
		upper = bin.Upper
	}
	return nil
}

func (a *NativeCalibration) accept(request *NativeRequest, candidate Candidate, maxRisk float64) bool {
	if request == nil || len(candidate.History) == 0 {
		return false
	}
	observations := request.Observe(candidate.Body)
	if !complete(observations) {
		return false
	}
	value := 1.0
	for _, observation := range observations {
		top, known := observationStatistic(observation, "top_probability")
		if !known {
			return false
		}
		value = math.Min(value, top)
	}
	taskSHA, err := CalibrationTaskSHA256(request)
	if err != nil {
		return false
	}
	last := candidate.History[len(candidate.History)-1]
	if last.Stage != candidate.Stage || last.Model != candidate.Model || !bytes.Equal(last.Response, candidate.Body) {
		return false
	}
	for _, cell := range a.Cells {
		if cell.TaskSHA256 != taskSHA || !calibrationHistoryMatches(cell.History, candidate.History) {
			continue
		}
		for _, bin := range cell.Bins {
			if value >= bin.Lower && (value < bin.Upper || (bin.Upper == 1 && value == 1)) {
				return bin.Accept && bin.risk <= maxRisk
			}
		}
	}
	return false
}

func calibrationHistoryMatches(expected []CalibrationHistory, observed []StageEvidence) bool {
	if len(expected) != len(observed) {
		return false
	}
	for i, step := range expected {
		actual := observed[i]
		identity, known := responseInferenceIdentity(actual.Response)
		if actual.Stage != step.Stage || actual.Model != step.Model || actual.Outcome != "response" || !known || identity != step.Identity {
			return false
		}
	}
	return true
}
