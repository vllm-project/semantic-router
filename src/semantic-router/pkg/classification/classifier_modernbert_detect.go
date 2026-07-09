package classification

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
)

// modelArchInfo is the minimal subset of a HuggingFace config.json needed to
// recognize a ModernBERT model before initialization.
type modelArchInfo struct {
	ModelType     string   `json:"model_type"`
	Architectures []string `json:"architectures"`
}

// isModernBertModel reports whether the model at modelID is a ModernBERT
// architecture, by reading model_type / architectures from its config.json.
//
// A ModernBERT config.json is incompatible with the traditional Candle BERT
// loader (model_type "modernbert", position_embedding_type "sans_pos", no
// top-level hidden_act). Probing the traditional loader first therefore emits
// alarming "Failed to initialize ... BERT" errors before the ModernBERT
// initializer succeeds as a fallback (issue #2096). Detecting ModernBERT up
// front lets callers try the ModernBERT initializer first and skip that doomed
// probe.
//
// Returns false on any read/parse error or for non-ModernBERT models, so
// callers fall back to the existing auto-detect-first ordering — detection is a
// best-effort optimization, never a hard gate.
func isModernBertModel(modelID string) bool {
	data, err := os.ReadFile(filepath.Join(modelID, "config.json"))
	if err != nil {
		return false
	}

	var cfg modelArchInfo
	if err := json.Unmarshal(data, &cfg); err != nil {
		return false
	}

	if strings.EqualFold(strings.TrimSpace(cfg.ModelType), "modernbert") {
		return true
	}
	for _, arch := range cfg.Architectures {
		if strings.HasPrefix(strings.ToLower(strings.TrimSpace(arch)), "modernbert") {
			return true
		}
	}
	return false
}
