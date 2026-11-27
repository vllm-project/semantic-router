//go:build !windows

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// storeConfigKeys moves the literal keys of a document this API persists
// into the key store beside sourcePath: files in router-keys/, or on a
// ConfigMap target the chart's key Secret. A plan passes create false, which
// only reads the store, so it compares the document a mutation would write.
// A listener behind Envoy keeps its keys, which the CLI renders into Envoy's
// own configuration. A key the store cannot take stays, with a warning.
func (s *ClassificationAPIServer) storeConfigKeys(sourcePath string, data []byte, create bool) (config.KeySave, error) {
	store, err := configwriter.KeyStore(sourcePath, config.RouterKeyDir, resolvedConfigMapWriter)
	if err != nil {
		return config.KeySave{}, err
	}
	save, err := config.StoreLiteralKeys(data, store, create, s.gatewayMode == config.GatewayStandalone)
	for _, warning := range save.Warnings() {
		logging.Warnf("[RouterConfig] %s: %s", warning.Field, warning.Message)
	}
	return save, err
}
