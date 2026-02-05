package extproc

import (
	"fmt"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/router/engine"
)

type Adapter struct {
	engine      *engine.RouterEngine
	extprocImpl *extproc.OpenAIRouter
	port        int
	secure      bool
	certPath    string
}

// NewAdapter creates a new ExtProc adapter with the router engine
func NewAdapter(eng *engine.RouterEngine, configPath string, port int, tlsConfig *config.TLSConfig) (*Adapter, error) {
	extprocRouter, err := extproc.NewOpenAIRouter(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create ExtProc router: %w", err)
	}

	secure := false
	certPath := ""
	if tlsConfig != nil && tlsConfig.Enabled {
		secure = true
		certPath = tlsConfig.CertFile
		logging.Infof("ExtProc adapter TLS enabled with cert: %s", certPath)
	}

	return &Adapter{
		engine:      eng,
		extprocImpl: extprocRouter,
		port:        port,
		secure:      secure,
		certPath:    certPath,
	}, nil
}

// Start starts the ExtProc gRPC server
func (a *Adapter) Start() error {
	if a.secure {
		logging.Infof("Starting ExtProc adapter on port %d (TLS enabled)", a.port)
	} else {
		logging.Infof("Starting ExtProc adapter on port %d", a.port)
	}

	server := extproc.NewServerWithRouter(a.extprocImpl, a.port, a.secure, a.certPath)
	if err := server.Start(); err != nil {
		return fmt.Errorf("ExtProc server error: %w", err)
	}

	return nil
}

// Stop stops the ExtProc adapter
func (a *Adapter) Stop() error {
	logging.Infof("Stopping ExtProc adapter")
	return nil
}

func (a *Adapter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	return a.extprocImpl.Process(stream)
}

func (a *Adapter) GetEngine() *engine.RouterEngine {
	return a.engine
}
