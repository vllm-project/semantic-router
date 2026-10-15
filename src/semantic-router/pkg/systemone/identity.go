package systemone

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// InferenceIdentity is shared with the model runtime wire contract.
type InferenceIdentity = api.InferenceIdentity

func responseInferenceIdentity(body json.RawMessage) (InferenceIdentity, bool) {
	return api.ResponseInferenceIdentity(body)
}

func validSHA256(value string) bool {
	decoded, err := hex.DecodeString(value)
	return err == nil && len(decoded) == sha256.Size && value == strings.ToLower(value)
}
