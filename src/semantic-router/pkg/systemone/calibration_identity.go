package systemone

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func calibrationSHA(value any) (string, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// CalibrationGateSHA256 commits to every task, arrival path, boundary and
// acceptance decision before independent certification. Counts cannot select
// thresholds or enable bins after observing certification labels.
func CalibrationGateSHA256(cells []CalibrationCell) (string, error) {
	type gateBin struct {
		Lower  float64 `json:"lower"`
		Upper  float64 `json:"upper"`
		Accept bool    `json:"accept"`
	}
	type gateCell struct {
		TaskSHA256 string               `json:"task_sha256"`
		History    []CalibrationHistory `json:"history"`
		Bins       []gateBin            `json:"bins"`
	}
	gates := make([]gateCell, len(cells))
	for i, cell := range cells {
		gates[i] = gateCell{TaskSHA256: cell.TaskSHA256, History: cell.History, Bins: make([]gateBin, len(cell.Bins))}
		for j, bin := range cell.Bins {
			gates[i].Bins[j] = gateBin{bin.Lower, bin.Upper, bin.Accept}
		}
	}
	return calibrationSHA(gates)
}

// CalibrationTaskSHA256 binds exact question instructions, criteria, options,
// named states and other request fields. Only model selection and input content
// vary within this task template. Variable instructions need their own evidence.
func CalibrationTaskSHA256(request *NativeRequest) (string, error) {
	if request == nil {
		return "", errors.New("missing native calibration task")
	}
	var task map[string]json.RawMessage
	if err := json.Unmarshal(request.Body, &task); err != nil || task == nil {
		return "", errors.New("invalid native calibration task")
	}
	delete(task, "model")
	// Keep questions as raw JSON: choice criteria object order determines the
	// model's option order. Compact marshaling removes whitespace without
	// sorting these model-visible fields into a different prompt template.
	var options map[string]json.RawMessage
	if raw, ok := task["options"]; ok {
		if err := json.Unmarshal(raw, &options); err != nil {
			return "", errors.New("invalid native calibration options")
		}
		delete(options, "return_meta")
		if len(options) == 0 {
			delete(task, "options")
		} else {
			task["options"], _ = json.Marshal(options)
		}
	}
	if err := maskCalibrationState(task); err != nil {
		return "", err
	}
	if states, present := task["states"]; present {
		var named map[string]map[string]json.RawMessage
		if err := json.Unmarshal(states, &named); err != nil || named == nil {
			return "", errors.New("invalid named calibration states")
		}
		for _, state := range named {
			if err := maskCalibrationState(state); err != nil {
				return "", err
			}
		}
		task["states"], _ = json.Marshal(named)
	}
	return calibrationSHA(task)
}

func maskCalibrationState(state map[string]json.RawMessage) error {
	var value any
	if err := json.Unmarshal(state["state"], &value); err != nil {
		return errors.New("unsupported calibration input state")
	}
	kind := ""
	switch value.(type) {
	case string:
		kind = "text"
	case map[string]any:
		kind = "object"
	case []any:
		kind = "array"
	default:
		return errors.New("unsupported calibration input state")
	}
	state["state"], _ = json.Marshal(map[string]string{"calibration_input_type": kind})
	return nil
}

// CalibrationPolicySHA256 binds only recipe execution semantics. Description,
// evidence file location, provider endpoints, credentials, listeners and global
// configuration generations are deliberately absent. Model runtime identity is
// separately checked against every observed response in the arrival history.
func CalibrationPolicySHA256(recipe *config.RoutingRecipe, decision *config.Decision) (string, error) {
	if recipe == nil || decision == nil || !decision.Algorithm.IsNative() {
		return "", errors.New("missing native calibration policy")
	}
	profile := recipe.Profile
	profile.Decisions = make([]config.Decision, len(recipe.Profile.Decisions))
	for i, source := range recipe.Profile.Decisions {
		copy := source
		copy.Description, copy.Annotations = "", nil
		if source.Algorithm != nil {
			algorithm := *source.Algorithm
			copy.Algorithm = &algorithm
			if algorithm.Quality != nil {
				quality := *algorithm.Quality
				quality.Calibration = "" // The artifact cannot hash itself.
				algorithm.Quality = &quality
			}
			if algorithm.Policy != nil {
				policy := *algorithm.Policy
				policy.Source = ""
				algorithm.Policy = &policy
			}
		}
		profile.Decisions[i] = copy
	}
	return calibrationSHA(struct {
		Version  string                `json:"version"`
		Recipe   config.RecipeName     `json:"recipe"`
		Decision string                `json:"decision"`
		Profile  config.RoutingProfile `json:"profile"`
	}{"systemone-arrival/v1", recipe.Name, decision.Name, profile})
}
