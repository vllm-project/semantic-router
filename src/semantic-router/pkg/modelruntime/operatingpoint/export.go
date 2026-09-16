package operatingpoint

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// BindArtifact converts an explicit score policy into a version-2 runtime
// sidecar. It preserves every score/window field and checks the existing weight
// identity. Version-2 execution declarations are preserved and their files
// verified; their qualification must precede packaging. This never selects
// thresholds or certifies graph equivalence.
func BindArtifact(ctx context.Context, source []byte, root string) ([]byte, error) {
	if err := rejectDuplicateKeys(json.NewDecoder(bytes.NewReader(source))); err != nil {
		return nil, err
	}
	var original Definition
	decoder := json.NewDecoder(bytes.NewReader(source))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&original); err != nil {
		return nil, fmt.Errorf("score policy must contain runtime fields only: %w", err)
	}
	if original.Version != 1 && original.Version != 2 {
		return nil, fmt.Errorf("score policy version must be 1 or 2")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(source, &fields); err != nil {
		return nil, err
	}
	// Preserve original thresholds (including their serialized precision) and all
	// input-policy fields. Only execution identity and schema version are added.
	executions := original.Executions
	if original.Version == 1 {
		executions = []Execution{{Provider: "candle", Precision: "float32", WeightsFile: "model.safetensors"}}
	}
	for name, value := range map[string]any{"version": 2, "executions": executions} {
		raw, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		fields[name] = raw
	}
	for _, entry := range []struct{ file, field string }{{"config.json", "model_config_sha256"}, {"tokenizer.json", "tokenizer_sha256"}} {
		raw, err := os.ReadFile(filepath.Join(root, entry.file))
		if err != nil {
			return nil, err
		}
		sum := sha256.Sum256(raw)
		fields[entry.field], err = json.Marshal(hex.EncodeToString(sum[:]))
		if err != nil {
			return nil, err
		}
	}
	output, err := json.MarshalIndent(fields, "", "  ")
	if err != nil {
		return nil, err
	}
	output = append(output, '\n')
	sum := sha256.Sum256(output)
	policy, err := Decode(output, hex.EncodeToString(sum[:]))
	if err != nil {
		return nil, err
	}
	for i := range policy.definition.Executions {
		policy.execution = &policy.definition.Executions[i]
		if err := policy.VerifyArtifacts(ctx, root); err != nil {
			return nil, err
		}
	}
	policy.execution = nil
	if err := policy.validateMetadata(root); err != nil {
		return nil, err
	}
	return output, nil
}
