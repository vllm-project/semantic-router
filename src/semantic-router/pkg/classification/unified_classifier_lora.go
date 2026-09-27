package classification

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

func (uc *UnifiedClassifier) classificationMode() (bool, error) {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	if !uc.initialized {
		return false, fmt.Errorf("unified classifier not initialized")
	}
	return uc.useLoRA, nil
}

func (uc *UnifiedClassifier) ensureLoRAInitialized() error {
	uc.mu.Lock()
	defer uc.mu.Unlock()
	if uc.loraInitialized {
		return nil
	}
	if err := uc.initializeLoRABindings(); err != nil {
		return err
	}
	uc.loraInitialized = true
	return nil
}

func (uc *UnifiedClassifier) initializeLoRABindings() error {
	if uc.testInitializeLoRA != nil {
		return uc.testInitializeLoRA()
	}
	if uc.loraModelPaths == nil {
		return fmt.Errorf("loRA model paths not configured")
	}
	models := uc.models
	if models == nil {
		models = standaloneModelRuntime()
		uc.models = models
	}
	specs := make([]config.ResolvedModelBinding, 3)
	for i, item := range []struct{ name, path, contract string }{{"domain_classifier", uc.loraModelPaths.IntentPath, config.RemoteClassifierContractLabelDistribution}, {"pii_classifier", uc.loraModelPaths.PIIPath, config.RemoteClassifierContractTokenSpans}, {"prompt_guard", uc.loraModelPaths.SecurityPath, config.RemoteClassifierContractLabelDistribution}} {
		adapter, err := mergedLoRAAdapter(item.path)
		if err != nil {
			return err
		}
		specs[i] = models.localSpec(item.name, item.path, adapter, item.contract, true)
	}
	model, err := models.runtime.LoRABatch(context.Background(), specs[0], specs[1], specs[2])
	if err != nil {
		return err
	}
	uc.intentLabels, uc.piiLabels, uc.securityLabels = model.Labels()
	uc.lora = model
	return nil
}

// These paths name merged checkpoints. Adapter-only PEFT weights cannot satisfy
// the underlying loader's full backbone tensors and fail during preparation.
func mergedLoRAAdapter(path string) (string, error) {
	data, err := os.ReadFile(filepath.Join(config.ResolveModelPath(path), "config.json"))
	if err != nil {
		return "", err
	}
	var model struct {
		Type string `json:"model_type"`
	}
	if err = json.Unmarshal(data, &model); err != nil {
		return "", err
	}
	switch strings.ToLower(model.Type) {
	case "bert":
		return "bert_lora", nil
	case "modernbert", "mmbert", "mmbert32k", "mmbert-32k":
		return strings.ToLower(model.Type), nil
	default:
		return "", fmt.Errorf("%w: merged LoRA architecture %q is unavailable", binding.ErrCapability, model.Type)
	}
}

func (uc *UnifiedClassifier) classifyBatchWithLoRAContext(ctx context.Context, texts []string) (*UnifiedBatchResults, error) {
	if uc.testClassifyBatchWithLoRA != nil {
		return uc.testClassifyBatchWithLoRA(texts)
	}
	if uc.lora == nil {
		return nil, binding.ErrClosed
	}
	output, err := uc.lora.ClassifyBatch(ctx, string(uc.models.recipe), texts)
	if err != nil {
		return nil, err
	}
	result := &UnifiedBatchResults{BatchSize: len(texts), IntentResults: make([]IntentResult, len(texts)), PIIResults: make([]PIIResult, len(texts)), SecurityResults: make([]SecurityResult, len(texts))}
	mapping := &PIIMapping{IdxToLabel: map[string]string{}}
	for i, label := range uc.piiLabels {
		mapping.IdxToLabel[fmt.Sprint(i)] = label
	}
	_, outside := knownPIILabels(mapping)
	for i := range texts {
		intent, confidence := deriveArgmax(output.Intent[i].Probabilities)
		if intent < 0 || intent >= len(uc.intentLabels) {
			return nil, fmt.Errorf("LoRA intent label outside prepared mapping")
		}
		result.IntentResults[i] = IntentResult{Category: uc.intentLabels[intent], Confidence: confidence, Probabilities: append([]float32(nil), output.Intent[i].Probabilities...)}
		types := map[string]bool{}
		positiveScores, outsideScores := []float32{}, []float32{}
		if !output.PII[i].HasScores() {
			return nil, fmt.Errorf("LoRA token probabilities are unavailable")
		}
		for _, entity := range output.PII[i].Entities {
			label := mapping.TranslatePIIType(entity.EntityType)
			if _, isOutside := outside[label]; isOutside {
				outsideScores = append(outsideScores, entity.Confidence)
			} else {
				types[label] = true
				positiveScores = append(positiveScores, entity.Confidence)
			}
		}
		scores := positiveScores
		if len(scores) == 0 {
			scores = outsideScores
		}
		score := float32(0)
		hasScore := len(scores) > 0
		for _, value := range scores {
			score += value
		}
		if hasScore {
			score /= float32(len(scores))
		}
		labels := make([]string, 0, len(types))
		for label := range types {
			labels = append(labels, label)
		}
		sort.Strings(labels)
		result.PIIResults[i] = PIIResult{HasPII: len(labels) > 0, PIITypes: labels, Confidence: score, ScoresAvailable: &hasScore}
		index, confidence := deriveArgmax(output.Security[i].Probabilities)
		if index < 0 || index >= len(uc.securityLabels) {
			return nil, fmt.Errorf("LoRA security label outside prepared mapping")
		}
		label := uc.securityLabels[index]
		available := true
		threat, err := legacyLoRAThreatLabel(label, uc.securityLabels)
		if err != nil {
			return nil, err
		}
		result.SecurityResults[i] = SecurityResult{IsJailbreak: threat, ThreatType: label, Confidence: confidence, ScoresAvailable: &available}
	}
	return result, nil
}

// A declared threat label is positive unless it is an explicit negative label.
// Substring matching would incorrectly classify "unsafe" as safe.
func legacyLoRAThreatLabel(label string, declared []string) (bool, error) {
	known := false
	for _, candidate := range declared {
		if candidate == label {
			known = true
			break
		}
	}
	if !known {
		return false, fmt.Errorf("unknown LoRA security label %q", label)
	}
	normalized := strings.NewReplacer("-", "_", " ", "_").Replace(strings.ToLower(strings.TrimSpace(label)))
	switch normalized {
	case "safe", "benign", "no_threat":
		return false, nil
	case "":
		return false, fmt.Errorf("empty LoRA security label")
	default:
		return true, nil
	}
}
