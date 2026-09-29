package native

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
)

// Metadata is read from the actual export, never inferred from a model alias.
// The IR tokenizer must be exported without built-in truncation so preflight
// observes the original count. The owned provider applies the input policy.
type openvinoArtifact struct {
	Root, Graph, Device, Overflow      string
	MaxTokens, ModelTokens, PadTokenID int
	EndTokenIDs                        []int
	Labels                             []string
}

func readOpenVINOArtifact(spec config.ResolvedModelBinding) (openvinoArtifact, error) {
	d := spec.Deployment.WithDefaults()
	a := openvinoArtifact{Root: d.Artifact, Device: d.Device, Overflow: d.Input.Overflow}
	if d.Precision != "native" || d.CustomOpsProfile != "" || d.CompilationCacheDir != "" {
		return a, fmt.Errorf("%w: OpenVINO requires native IR precision without ORT options", binding.ErrCapability)
	}
	if a.Overflow != "reject" && a.Overflow != "truncate" {
		return a, fmt.Errorf("%w: OpenVINO supports reject or truncate", binding.ErrCapability)
	}
	if filepath.Ext(a.Root) == ".xml" {
		a.Graph = a.Root
		a.Root = filepath.Dir(a.Root)
	}
	if spec.Binding.Head != "" {
		a.Graph = spec.Binding.Head
		if !filepath.IsAbs(a.Graph) {
			a.Graph = filepath.Join(a.Root, a.Graph)
		}
	}
	if a.Graph == "" {
		for _, name := range []string{"openvino/openvino_model.xml", "openvino_model.xml"} {
			candidate := filepath.Join(a.Root, name)
			if info, err := os.Stat(candidate); err == nil && !info.IsDir() {
				a.Graph = candidate
				break
			}
		}
	}
	if filepath.Ext(a.Graph) != ".xml" {
		return a, fmt.Errorf("%w: OpenVINO requires a complete IR XML graph", binding.ErrCapability)
	}
	var err error
	if err = validateOpenVINOTokenizerIR(a.Graph); err != nil {
		return a, err
	}
	a.Root, err = filepath.Abs(a.Root)
	if err != nil {
		return a, err
	}
	a.Graph, err = filepath.Abs(a.Graph)
	if err != nil {
		return a, err
	}
	data, err := os.ReadFile(filepath.Join(a.Root, "config.json"))
	if err != nil {
		return a, err
	}
	var metadata struct {
		MaxTokens  int               `json:"max_position_embeddings"`
		Labels     map[string]string `json:"id2label"`
		EOS        json.RawMessage   `json:"eos_token_id"`
		PadTokenID *int              `json:"pad_token_id"`
		SEP        json.RawMessage   `json:"sep_token_id"`
	}
	if err = json.Unmarshal(data, &metadata); err != nil {
		return a, err
	}
	if metadata.PadTokenID == nil || *metadata.PadTokenID < 0 || *metadata.PadTokenID > 2147483647 {
		return a, fmt.Errorf("%w: OpenVINO export must declare pad_token_id", binding.ErrCapability)
	}
	a.PadTokenID = *metadata.PadTokenID
	a.ModelTokens = metadata.MaxTokens
	tokenizerData, tokenizerErr := os.ReadFile(filepath.Join(a.Root, "tokenizer_config.json"))
	if tokenizerErr == nil {
		var tokenizer struct {
			MaxTokens float64 `json:"model_max_length"`
		}
		if err = json.Unmarshal(tokenizerData, &tokenizer); err != nil {
			return a, err
		}
		if tokenizer.MaxTokens > 0 && tokenizer.MaxTokens <= 2147483647 && tokenizer.MaxTokens == float64(int(tokenizer.MaxTokens)) && (a.ModelTokens <= 0 || int(tokenizer.MaxTokens) < a.ModelTokens) {
			a.ModelTokens = int(tokenizer.MaxTokens)
		}
	} else if !os.IsNotExist(tokenizerErr) {
		return a, tokenizerErr
	}
	if a.ModelTokens <= 0 || a.ModelTokens > 2147483647 {
		return a, fmt.Errorf("%w: OpenVINO export must declare a finite tokenizer/model capacity", binding.ErrCapability)
	}
	a.MaxTokens = d.Input.MaxTokens
	if a.MaxTokens == 0 {
		a.MaxTokens = a.ModelTokens
	}
	if a.MaxTokens > a.ModelTokens || a.MaxTokens <= 0 {
		return a, fmt.Errorf("%w: deployment budget exceeds OpenVINO export capacity %d", binding.ErrCapability, a.ModelTokens)
	}
	a.Labels = make([]string, len(metadata.Labels))
	for key, label := range metadata.Labels {
		index, indexErr := strconv.Atoi(key)
		if indexErr != nil || index < 0 || index >= len(a.Labels) || strings.TrimSpace(label) == "" {
			return a, fmt.Errorf("%w: export id2label must be contiguous and non-empty", binding.ErrCapability)
		}
		a.Labels[index] = label
	}
	for _, label := range a.Labels {
		if label == "" {
			return a, fmt.Errorf("%w: export id2label is not contiguous", binding.ErrCapability)
		}
	}
	data, err = os.ReadFile(filepath.Join(a.Root, "tokenizer.json"))
	if err != nil {
		return a, err
	}
	var tokenizer struct {
		PostProcessor map[string]any `json:"post_processor"`
	}
	if err = json.Unmarshal(data, &tokenizer); err != nil {
		return a, err
	}
	a.EndTokenIDs = openVINOSuffixTokens(tokenizer.PostProcessor)
	// Config EOS/SEP are a fallback only when the post-processor declares no suffix.
	if len(a.EndTokenIDs) == 0 {
		for _, raw := range []json.RawMessage{metadata.SEP, metadata.EOS} {
			var ids []int
			if json.Unmarshal(raw, &ids) != nil {
				var id int
				if string(raw) != "null" && json.Unmarshal(raw, &id) == nil {
					ids = []int{id}
				}
			}
			for _, id := range ids {
				if id >= 0 {
					a.EndTokenIDs = append(a.EndTokenIDs, id)
				}
			}
		}
	}
	sort.Ints(a.EndTokenIDs)
	return a, nil
}

func openVINOSuffixTokens(processor map[string]any) []int {
	var ids []int
	switch processor["type"] {
	case "BertProcessing", "RobertaProcessing":
		if sep, ok := processor["sep"].([]any); ok && len(sep) == 2 {
			if id, ok := sep[1].(float64); ok && id >= 0 && id == float64(int(id)) {
				ids = append(ids, int(id))
			}
		}
	case "TemplateProcessing":
		tokens, _ := processor["special_tokens"].(map[string]any)
		single, _ := processor["single"].([]any)
		afterSequence := false
		for _, value := range single {
			element, _ := value.(map[string]any)
			if _, ok := element["Sequence"]; ok {
				afterSequence = true
				ids = nil
				continue
			}
			special, _ := element["SpecialToken"].(map[string]any)
			name, _ := special["id"].(string)
			token, _ := tokens[name].(map[string]any)
			values, _ := token["ids"].([]any)
			if afterSequence {
				for _, value := range values {
					if id, ok := value.(float64); ok && id >= 0 && id == float64(int(id)) {
						ids = append(ids, int(id))
					}
				}
			}
		}
	case "Sequence":
		processors, _ := processor["processors"].([]any)
		for _, value := range processors {
			if nested, ok := value.(map[string]any); ok {
				ids = append(ids, openVINOSuffixTokens(nested)...)
			}
		}
	}
	return ids
}
