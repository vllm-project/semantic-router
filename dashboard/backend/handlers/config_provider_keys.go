package handlers

import (
	"log"

	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
)

// A key typed into the Dashboard, as Connect Models takes one, is saved as a
// reference. Each literal providers.models[].backend_refs[].api_key, and in a
// standalone stack each literal listener api_keys entry, moves to the store
// configwriter.KeyStore picks: an owner-only file in provider-keys/ beside
// the config, which `vllm-sr serve` mounts into both containers at one path,
// or on Kubernetes the key Secret the chart loads into the Router's and the
// Dashboard's environment. An environment reference such as
// ${OPENAI_API_KEY} stays as written. An Envoy listener keeps its keys: Envoy
// holds them in its own configuration, which the CLI renders. A key the
// deployment has no store for stays as written, with a warning.

// persistConfig writes a saved document with its literal keys stored as
// references, and returns the bytes it wrote and a warning for each key it
// kept as written.
func persistConfig(configPath string, data []byte) ([]byte, []routerconfig.ConfigWarning, error) {
	store, err := configwriter.KeyStore(configPath, routerconfig.ProviderKeyDir, resolvedConfigMapWriter)
	if err != nil {
		return nil, nil, err
	}
	save, err := routerconfig.StoreLiteralKeys(data, store, true, !managedStackRunsEnvoy())
	if err != nil {
		return nil, nil, err
	}
	if err = writeConfigAtomically(configPath, save.Data); err != nil {
		save.Discard()
		return nil, nil, err
	}
	warnings := save.Warnings()
	for _, warning := range warnings {
		log.Printf("Configuration save: %s: %s", warning.Field, warning.Message)
	}
	return save.Data, warnings, nil
}
