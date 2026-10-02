package operatingpoint

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// CKFlashAttentionLibrary is an installed trusted implementation. A policy may
// bind its content identity but cannot select an executable library path.
const CKFlashAttentionLibrary = "/usr/local/lib/libort_ck_flash_attn.so.1"

// Load verifies only explicitly selected files. The native provider remains the
// model loader and owns its actual input budget, task head and resource identity.
func Load(ctx context.Context, spec config.ResolvedModelBinding, labels []string) (*Policy, error) {
	ref := spec.Binding.OperatingPoint
	if ref == nil {
		return nil, fmt.Errorf("independent scores require an explicit operating point reference")
	}
	if err := ref.Validate(); err != nil {
		return nil, err
	}
	if spec.Binding.Contract != config.RemoteClassifierContractLabelScores || (spec.Deployment.Provider != "candle" && spec.Deployment.Provider != "ort") {
		return nil, fmt.Errorf("operating point requires a complete local label_scores.v1 artifact")
	}
	file, err := os.Open(ref.ResolvePath(spec.Deployment.Artifact))
	if err != nil {
		return nil, err
	}
	defer file.Close()
	data, err := io.ReadAll(io.LimitReader(file, 1<<20))
	if err != nil {
		return nil, err
	}
	if len(data) == 1<<20 {
		return nil, fmt.Errorf("operating point exceeds the metadata size limit")
	}
	digest := sha256.Sum256(data)
	if hex.EncodeToString(digest[:]) != ref.SHA256 {
		return nil, fmt.Errorf("operating point SHA256 differs from its immutable reference")
	}
	p, err := Decode(data, ref.SHA256)
	if err != nil {
		return nil, err
	}
	if !slices.Equal(labels, p.definition.Labels) {
		return nil, fmt.Errorf("operating point labels differ from the generic rule's ordered labels")
	}
	deployment := spec.Deployment.WithDefaults()
	if deployment.Input.MaxTokens != p.MaxTokens() || deployment.Input.Overflow != "reject" {
		return nil, fmt.Errorf("deployment must explicitly match the operating point document budget and reject overflow")
	}
	precision := deployment.Precision
	if deployment.Provider == "candle" && (precision == "native" || precision == "fp32") {
		precision = "float32"
	}
	execution, err := p.selectExecution(deployment.Provider, precision, deployment.Device)
	if err != nil {
		return nil, err
	}
	p.execution = execution
	if execution.ONNX == nil && spec.Binding.Head != "" {
		return nil, fmt.Errorf("operating point forbids separate Candle heads")
	}
	if execution.ONNX != nil && spec.Binding.Head != "" && spec.Binding.Head != execution.ONNX.File {
		return nil, fmt.Errorf("selected ONNX head differs from operating point")
	}
	profile := ""
	if execution.ONNX != nil {
		profile = execution.ONNX.CustomOpsProfile
	}
	if deployment.CustomOpsProfile != profile {
		return nil, fmt.Errorf("operating point graph does not declare a custom-ops execution")
	}
	if err := p.VerifyArtifacts(ctx, deployment.Artifact); err != nil {
		return nil, err
	}
	if err := p.validateMetadata(deployment.Artifact); err != nil {
		return nil, err
	}
	return p, nil
}

// VerifyArtifacts is also called after native preparation. Startup replacement
// cannot bind a policy to different files; a published generation is immutable.
func (p *Policy) VerifyArtifacts(ctx context.Context, root string) error {
	return p.verifyArtifacts(ctx, root, CKFlashAttentionLibrary)
}

func (p *Policy) verifyArtifacts(ctx context.Context, root, trustedLibrary string) error {
	customOpsSHA256 := ""
	files := map[string]string{"model.safetensors": p.definition.ModelWeightsSHA256, "config.json": p.definition.ModelConfigSHA256, "tokenizer.json": p.definition.TokenizerSHA256}
	if graph := p.ONNX(); graph != nil {
		for _, artifact := range graph.Artifacts {
			if artifact.Role == "custom-ops" {
				customOpsSHA256 = artifact.SHA256
				continue
			}
			path := graph.File
			if artifact.Role != "graph" {
				path = filepath.Join(filepath.Dir(graph.File), strings.TrimPrefix(artifact.Role, "external:"))
			}
			if previous, exists := files[path]; exists && previous != artifact.SHA256 {
				return fmt.Errorf("conflicting artifact identities for %s", path)
			}
			files[path] = artifact.SHA256
		}
	}
	for name, want := range files {
		path, err := containedArtifact(root, name)
		if err != nil {
			return err
		}
		if err := verifyFileDigest(ctx, path, want); err != nil {
			return err
		}
	}
	if customOpsSHA256 != "" {
		return verifyFileDigest(ctx, trustedLibrary, customOpsSHA256)
	}
	return nil
}

func verifyFileDigest(ctx context.Context, path, want string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		_ = file.Close()
		return fmt.Errorf("operating point artifact must be a regular file: %s", path)
	}
	hash := sha256.New()
	_, copyErr := io.Copy(hash, &contextReader{ctx: ctx, reader: file})
	closeErr := file.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	if hex.EncodeToString(hash.Sum(nil)) != want {
		return fmt.Errorf("operating point %s SHA256 differs from selected artifact", path)
	}
	return nil
}

// Explicit artifact paths may not escape the selected model, including symlinks.
func containedArtifact(root, name string) (string, error) {
	if !localArtifactPath(name) {
		return "", fmt.Errorf("invalid artifact path %q", name)
	}
	base, err := filepath.EvalSymlinks(root)
	if err != nil {
		return "", err
	}
	base, err = filepath.Abs(base)
	if err != nil {
		return "", err
	}
	actual, err := filepath.EvalSymlinks(filepath.Join(base, name))
	if err != nil {
		return "", err
	}
	relative, err := filepath.Rel(base, actual)
	if err != nil || !filepath.IsLocal(relative) {
		return "", fmt.Errorf("artifact escapes model directory: %s", name)
	}
	return actual, nil
}

type contextReader struct {
	ctx    context.Context
	reader io.Reader
}

func (r *contextReader) Read(data []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.reader.Read(data)
}

func (p *Policy) validateMetadata(root string) error {
	data, err := os.ReadFile(filepath.Join(root, "config.json"))
	if err != nil {
		return err
	}
	var metadata struct {
		ProblemType  string            `json:"problem_type"`
		Pooling      string            `json:"classifier_pooling"`
		MaxPositions int               `json:"max_position_embeddings"`
		PadTokenID   *uint32           `json:"pad_token_id"`
		ID2Label     map[string]string `json:"id2label"`
		Label2ID     map[string]int    `json:"label2id"`
	}
	if err = json.Unmarshal(data, &metadata); err != nil {
		return err
	}
	if metadata.ProblemType != "multi_label_classification" || (metadata.Pooling != "mean" && metadata.Pooling != "cls") || metadata.MaxPositions < p.Window().Size || metadata.PadTokenID == nil || *metadata.PadTokenID != *p.definition.Input.PadTokenID {
		return fmt.Errorf("model config does not declare the operating point's independent head, pooling or input contract")
	}
	if len(metadata.ID2Label) != len(p.definition.Labels) || len(metadata.Label2ID) != len(p.definition.Labels) {
		return fmt.Errorf("model config labels are incomplete")
	}
	for i, label := range p.definition.Labels {
		id, ok := metadata.Label2ID[label]
		if !ok || id != i || metadata.ID2Label[strconv.Itoa(i)] != label {
			return fmt.Errorf("model config label order differs from operating point")
		}
	}
	data, err = os.ReadFile(filepath.Join(root, "tokenizer.json"))
	if err != nil {
		return err
	}
	prefix, suffix, err := tokenizerEnvelope(data)
	if err != nil {
		return err
	}
	if !slices.Equal(prefix, p.definition.Input.SpecialPrefixIDs) || !slices.Equal(suffix, p.definition.Input.SpecialSuffixIDs) {
		return fmt.Errorf("tokenizer special-token envelope differs from operating point")
	}
	return nil
}

// Inspect the saved token template, not text or decoded windows. Actual token
// slicing and special-token restoration remain solely in the owned native API.
func tokenizerEnvelope(data []byte) ([]uint32, []uint32, error) {
	var tokenizer struct {
		Processor json.RawMessage `json:"post_processor"`
	}
	if err := json.Unmarshal(data, &tokenizer); err != nil {
		return nil, nil, err
	}
	if len(tokenizer.Processor) == 0 || string(tokenizer.Processor) == "null" {
		return nil, nil, nil
	}
	var processor struct {
		Type     string                       `json:"type"`
		Single   []map[string]json.RawMessage `json:"single"`
		Specials map[string]struct {
			IDs []uint32 `json:"ids"`
		} `json:"special_tokens"`
		CLS []json.RawMessage `json:"cls"`
		SEP []json.RawMessage `json:"sep"`
	}
	if err := json.Unmarshal(tokenizer.Processor, &processor); err != nil {
		return nil, nil, err
	}
	if processor.Type == "BertProcessing" {
		if len(processor.CLS) != 2 || len(processor.SEP) != 2 {
			return nil, nil, fmt.Errorf("invalid tokenizer CLS/SEP template")
		}
		var prefix, suffix uint32
		if err := json.Unmarshal(processor.CLS[1], &prefix); err != nil {
			return nil, nil, err
		}
		if err := json.Unmarshal(processor.SEP[1], &suffix); err != nil {
			return nil, nil, err
		}
		return []uint32{prefix}, []uint32{suffix}, nil
	}
	if processor.Type != "TemplateProcessing" {
		return nil, nil, fmt.Errorf("unsupported tokenizer special-token template %q", processor.Type)
	}
	var prefix, suffix []uint32
	sequence := false
	for _, item := range processor.Single {
		if len(item) != 1 {
			return nil, nil, fmt.Errorf("ambiguous tokenizer template")
		}
		if raw, ok := item["Sequence"]; ok {
			var value struct {
				ID     string `json:"id"`
				TypeID int    `json:"type_id"`
			}
			if err := json.Unmarshal(raw, &value); err != nil {
				return nil, nil, err
			}
			if sequence || value.ID != "A" || value.TypeID != 0 {
				return nil, nil, fmt.Errorf("window policy requires one contiguous sequence A")
			}
			sequence = true
			continue
		}
		raw, ok := item["SpecialToken"]
		if !ok {
			return nil, nil, fmt.Errorf("unsupported tokenizer template item")
		}
		var value struct {
			ID     string `json:"id"`
			TypeID int    `json:"type_id"`
		}
		if err := json.Unmarshal(raw, &value); err != nil {
			return nil, nil, err
		}
		special, ok := processor.Specials[value.ID]
		if !ok || len(special.IDs) == 0 || value.TypeID != 0 {
			return nil, nil, fmt.Errorf("missing or unsupported tokenizer special token")
		}
		if sequence {
			suffix = append(suffix, special.IDs...)
		} else {
			prefix = append(prefix, special.IDs...)
		}
	}
	if !sequence {
		return nil, nil, fmt.Errorf("tokenizer template has no content sequence")
	}
	return prefix, suffix, nil
}
