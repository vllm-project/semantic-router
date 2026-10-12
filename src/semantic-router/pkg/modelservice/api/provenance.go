package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"strings"
)

// InferenceIdentity records the observed model and execution profile. A public
// alias is a transport selector; it never replaces the identity in metadata.
type InferenceIdentity struct {
	ModelID     string `json:"model_id"`
	Revision    string `json:"revision"`
	ModelSHA256 string `json:"model_sha256"`
	Engine      string `json:"engine"`
	Profile     string `json:"profile"`
	Numerics    string `json:"numerics"`
	Accelerator string `json:"accelerator"`
}

// Valid requires every component used to bind inference evidence. In particular,
// a mutable name or partial diagnostics cannot certify a model identity.
func (i InferenceIdentity) Valid() bool {
	for _, value := range []string{i.ModelID, i.Revision, i.Engine, i.Profile, i.Numerics, i.Accelerator} {
		if value == "" || strings.TrimSpace(value) != value {
			return false
		}
	}
	decoded, err := hex.DecodeString(i.ModelSHA256)
	return err == nil && len(decoded) == sha256.Size && i.ModelSHA256 == strings.ToLower(i.ModelSHA256)
}

func envelopeIdentity(response map[string]json.RawMessage) (InferenceIdentity, bool) {
	var identity InferenceIdentity
	var meta map[string]json.RawMessage
	if json.Unmarshal(response["meta"], &meta) != nil || meta == nil || json.Unmarshal(response["meta"], &identity) != nil {
		return identity, false
	}
	// A raw native response may identify its model in the outer envelope. Once
	// model_id is explicit, malformed provenance must never fall back to an alias.
	if _, explicit := meta["model_id"]; !explicit {
		if json.Unmarshal(response["model"], &identity.ModelID) != nil {
			return identity, false
		}
	}
	return identity, identity.Valid()
}

// ResponseInferenceIdentity reads native provenance without consulting mutable
// configuration. Additional state envelopes may omit metadata, but any identity
// they report must agree with the enclosing response.
func ResponseInferenceIdentity(body json.RawMessage) (InferenceIdentity, bool) {
	var response map[string]json.RawMessage
	if json.Unmarshal(body, &response) != nil || response == nil {
		return InferenceIdentity{}, false
	}
	identity, known := envelopeIdentity(response)
	if !known {
		return identity, false
	}
	if raw, exists := response["states"]; exists {
		var states map[string]map[string]json.RawMessage
		if json.Unmarshal(raw, &states) != nil {
			return identity, false
		}
		for _, state := range states {
			if _, present := state["meta"]; present {
				observed, valid := envelopeIdentity(state)
				if !valid || observed != identity {
					return identity, false
				}
			}
		}
	}
	return identity, true
}

// AliasResponseModel changes only transport selectors, preserving complete
// observed native provenance before the first alias hop. It does not create
// metadata when the runtime omitted it or reconstruct identity from config.
// Answers and their user-selected names remain raw JSON.
func AliasResponseModel(response map[string]json.RawMessage, alias string) error {
	if err := aliasEnvelope(response, alias); err != nil {
		return err
	}
	if raw, exists := response["states"]; exists {
		var states map[string]map[string]json.RawMessage
		if json.Unmarshal(raw, &states) != nil {
			return errors.New("invalid native state envelopes")
		}
		for _, state := range states {
			if err := aliasEnvelope(state, alias); err != nil {
				return err
			}
		}
		response["states"], _ = json.Marshal(states)
	}
	return nil
}

func aliasEnvelope(response map[string]json.RawMessage, alias string) error {
	if response == nil {
		return errors.New("invalid native response envelope")
	}
	if raw, exists := response["meta"]; exists && string(raw) != "null" {
		var meta map[string]json.RawMessage
		if json.Unmarshal(raw, &meta) != nil || meta == nil {
			return errors.New("invalid native response metadata")
		}
		if value, explicit := meta["model_id"]; explicit {
			var model string
			if json.Unmarshal(value, &model) != nil || model == "" || strings.TrimSpace(model) != model {
				return errors.New("invalid native model identity")
			}
		} else if identity, known := envelopeIdentity(response); known {
			meta["model_id"], _ = json.Marshal(identity.ModelID)
			response["meta"], _ = json.Marshal(meta)
		}
	}
	response["model"], _ = json.Marshal(alias)
	return nil
}
