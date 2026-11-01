package handlers

import (
	"bytes"
	"errors"
	"fmt"
	"io"

	"gopkg.in/yaml.v3"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Reject unknown setup fields before canonical transport can discard them.
// KnownFields uses the authoritative Go config types, without maintaining a
// second field inventory or changing ordinary config endpoint decoding.
func decodeStrictSetupConfig(data []byte) (setupConfigFile, error) {
	var cfg setupConfigFile
	if len(bytes.TrimSpace(data)) == 0 {
		return cfg, nil
	}
	// Fields the Router rejects by presence would vanish here when null.
	if err := routerconfig.ValidateRawFields(data); err != nil {
		return cfg, err
	}
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&cfg); err != nil {
		return cfg, err
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return cfg, fmt.Errorf("setup config must contain one canonical document")
	}
	if err := resolveTransportGlobal(data, &cfg); err != nil {
		return cfg, err
	}
	return cfg, nil
}
