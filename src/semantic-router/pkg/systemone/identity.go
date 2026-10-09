package systemone

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

// InferenceIdentity is the native runtime's observed inference identity.
// Endpoint names and a mutable model alias alone do not identify an inference result.
// Profile is the runtime's published profile; it does not imply a weight dtype.
type InferenceIdentity struct {
	ModelID     string `json:"model_id"`
	Revision    string `json:"revision"`
	ModelSHA256 string `json:"model_sha256"`
	Engine      string `json:"engine"`
	Profile     string `json:"profile"`
	Numerics    string `json:"numerics"`
	Accelerator string `json:"accelerator"`
}

func (i InferenceIdentity) valid() bool {
	for _, value := range []string{i.ModelID, i.Revision, i.Engine, i.Profile, i.Numerics, i.Accelerator} {
		if value == "" || strings.TrimSpace(value) != value {
			return false
		}
	}
	return validSHA256(i.ModelSHA256)
}

func responseInferenceIdentity(body json.RawMessage) (InferenceIdentity, bool) {
	var response struct {
		Model string            `json:"model"`
		Meta  InferenceIdentity `json:"meta"`
	}
	if json.Unmarshal(body, &response) != nil {
		return InferenceIdentity{}, false
	}
	response.Meta.ModelID = response.Model
	return response.Meta, response.Meta.valid()
}

func validSHA256(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}
