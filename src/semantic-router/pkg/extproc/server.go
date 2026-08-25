package extproc

import (
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/quotaruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingcontext"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	tlsutil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/tls"
)

// Server represents a gRPC server for the Envoy ExtProc
type Server struct {
	service  *RouterService
	server   *grpc.Server
	port     int
	secure   bool
	certPath string
	runtime  *routerruntime.Registry
	stopOnce sync.Once
}

// ConfigParser parses one immutable authoring manifest through the
// application-composed Provider Integration boundary.
type ConfigParser func(string) (*config.RouterConfig, error)

// ServerDependencies are process-owned request and dispatch resources.
type ServerDependencies struct {
	DurableRoutingRequests *DurableRoutingRequestRuntime
	FileRequests           *FileRequestRuntime
	DispatchCapabilities   DispatchCapabilityRuntime
	OutcomeFeedback        OutcomeFeedbackRuntime
	OutcomeProjection      OutcomeLearningProjectionRuntime
	ResponseTerminals      backendinvoker.ResponseTerminalReader
	ProtocolCodecs         *protocolcodec.Registry
	ParseConfig            ConfigParser
}

func (dependencies ServerDependencies) routerDependencies() RuntimeDependencies {
	if dependencies.DurableRoutingRequests == nil {
		return RuntimeDependencies{
			DispatchCapabilities: dependencies.DispatchCapabilities,
			OutcomeFeedback:      dependencies.OutcomeFeedback,
			OutcomeProjection:    dependencies.OutcomeProjection,
			ResponseTerminals:    dependencies.ResponseTerminals,
			ProtocolCodecs:       dependencies.ProtocolCodecs,
		}
	}
	return RuntimeDependencies{
		InferenceAccess:      dependencies.DurableRoutingRequests.access,
		DispatchCapabilities: dependencies.DispatchCapabilities,
		OutcomeFeedback:      dependencies.OutcomeFeedback,
		OutcomeProjection:    dependencies.OutcomeProjection,
		ResponseTerminals:    dependencies.ResponseTerminals,
		ProtocolCodecs:       dependencies.ProtocolCodecs,
	}
}

// NewServerWithDependencies creates a server borrowing process-owned request
// and dispatch resources for its immutable process lifetime.
func NewServerWithDependencies(
	configPath string,
	port int,
	secure bool,
	certPath string,
	runtimeRegistry *routerruntime.Registry,
	dependencies ServerDependencies,
) (*Server, error) {
	routerDependencies := dependencies.routerDependencies()
	parseConfig := dependencies.ParseConfig
	if parseConfig == nil {
		parseConfig = config.NewParser(nil).Parse
	}
	router, err := newOpenAIRouterForServer(configPath, runtimeRegistry, routerDependencies, parseConfig)
	if err != nil {
		return nil, err
	}
	capabilities, err := runtimecapabilities.Derive(router.Config)
	if err != nil {
		_ = router.Close()
		return nil, fmt.Errorf("derive runtime capabilities: %w", err)
	}
	durableRouting := capabilities.DurableRouting
	if durableRouting && dependencies.DurableRoutingRequests == nil {
		_ = router.Close()
		return nil, errors.New("durable routing requires a published request runtime")
	}
	if !durableRouting && dependencies.DurableRoutingRequests != nil {
		_ = router.Close()
		return nil, errors.New("published request runtime requires durable routing")
	}
	if durableRouting && dependencies.FileRequests != nil {
		_ = router.Close()
		return nil, errors.New("file-routing request runtime was injected with durable Management")
	}
	if !durableRouting && dependencies.FileRequests == nil {
		_ = router.Close()
		return nil, errors.New("file routing requires a file request runtime")
	}
	if !durableRouting && !dependencies.FileRequests.matches(router.Config.RoutingSnapshot) {
		_ = router.Close()
		return nil, errors.New("file Router does not match the process routing snapshot")
	}
	if durableRouting && capabilities.NativeAccess != dependencies.DurableRoutingRequests.accessEnabled() {
		_ = router.Close()
		return nil, errors.New("published request runtime access capability does not match Router configuration")
	}
	attachRuntimeRegistry(router, runtimeRegistry)
	publishRouterState(router.Config, router, runtimeRegistry)

	service := newRouterServiceWithRequestRuntimes(
		router, dependencies.DurableRoutingRequests, dependencies.FileRequests,
	)
	return &Server{
		service:  service,
		port:     port,
		secure:   secure,
		certPath: certPath,
		runtime:  runtimeRegistry,
	}, nil
}

// GetRouter returns the current router instance
func (s *Server) GetRouter() *OpenAIRouter {
	return s.service.GetRouter()
}

// Start starts the gRPC server
func (s *Server) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", s.port))
	if err != nil {
		s.Stop()
		return fmt.Errorf("failed to listen on port %d: %w", s.port, err)
	}

	// Configure server options based on secure mode
	var serverOpts []grpc.ServerOption

	if s.secure {
		var cert tls.Certificate
		var err error

		if s.certPath != "" {
			// Load certificate from provided path
			certFile := filepath.Join(s.certPath, "tls.crt")
			keyFile := filepath.Join(s.certPath, "tls.key")
			cert, err = tls.LoadX509KeyPair(certFile, keyFile)
			if err != nil {
				_ = lis.Close()
				s.Stop()
				return fmt.Errorf("failed to load TLS certificate from %s: %w", s.certPath, err)
			}
			logging.ComponentEvent("extproc", "tls_certificate_loaded", map[string]interface{}{
				"path": s.certPath,
			})
		} else {
			// Create self-signed certificate
			cert, err = tlsutil.CreateSelfSignedTLSCertificate()
			if err != nil {
				_ = lis.Close()
				s.Stop()
				return fmt.Errorf("failed to create self-signed certificate: %w", err)
			}
			logging.ComponentEvent("extproc", "tls_certificate_created", map[string]interface{}{
				"source": "self_signed",
			})
		}

		creds := credentials.NewTLS(&tls.Config{
			Certificates: []tls.Certificate{cert},
		})
		serverOpts = append(serverOpts, grpc.Creds(creds))
	}

	maxMsgSize := s.configuredGRPCMaxMessageSize()
	serverOpts = append(serverOpts,
		grpc.MaxRecvMsgSize(maxMsgSize),
		grpc.MaxSendMsgSize(maxMsgSize),
	)
	logging.ComponentEvent("extproc", "server_starting", map[string]interface{}{
		"port":       s.port,
		"secure":     s.secure,
		"max_msg_mb": maxMsgSize / (1024 * 1024),
	})
	s.server = grpc.NewServer(serverOpts...)
	ext_proc.RegisterExternalProcessorServer(s.server, s.service)

	// Run the server in a separate goroutine
	serverErrCh := make(chan error, 1)
	go func() {
		if err := s.server.Serve(lis); err != nil && !errors.Is(err, grpc.ErrServerStopped) {
			serverErrCh <- err
		} else {
			serverErrCh <- nil
		}
	}()

	// Wait for interrupt signal to gracefully shut down the server
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(signalChan)

	// Wait for either server error or shutdown signal
	select {
	case err := <-serverErrCh:
		if err != nil {
			logging.ComponentErrorEvent("extproc", "server_stopped_with_error", map[string]interface{}{
				"port":  s.port,
				"error": err.Error(),
			})
			s.Stop()
			return err
		}
	case <-signalChan:
		logging.ComponentEvent("extproc", "server_shutdown_requested", map[string]interface{}{
			"port": s.port,
		})
	}

	s.Stop()
	return nil
}

// Stop stops the gRPC server
func (s *Server) Stop() {
	s.stopOnce.Do(func() {
		if s.server != nil {
			s.server.GracefulStop()
			logging.ComponentEvent("extproc", "server_stopped", map[string]interface{}{
				"port": s.port,
			})
		}
		if s.service != nil {
			_ = s.service.Close()
		}
	})
}

// RouterService owns the immutable bootstrap Router used by file-backed streams
// and durable-publication error responses. Published request generations come from the
// process-owned publication registry.
type RouterService struct {
	current        atomic.Pointer[routerGeneration]
	durableRouting *DurableRoutingRequestRuntime
	fileRequests   *FileRequestRuntime
	mu             sync.Mutex
	closed         bool
	retired        sync.WaitGroup
	errMu          sync.Mutex
	errors         []error
}

type routerGeneration struct {
	router *OpenAIRouter
	refs   sync.WaitGroup
}

func newRouterServiceWithRequestRuntimes(
	r *OpenAIRouter,
	durableRouting *DurableRoutingRequestRuntime,
	fileRequests *FileRequestRuntime,
) *RouterService {
	rs := &RouterService{durableRouting: durableRouting, fileRequests: fileRequests}
	rs.current.Store(&routerGeneration{router: r})
	return rs
}

// GetRouter returns the current router implementation.
func (rs *RouterService) GetRouter() *OpenAIRouter {
	generation := rs.current.Load()
	if generation == nil {
		return nil
	}
	return generation.router
}

// Process delegates file-backed streams to the process-global generation. With
// durable routing it authenticates once, verifies the active publication lease,
// and acquires the exactly pinned namespace generation before replaying the
// first header frame.
func (rs *RouterService) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	if rs.durableRouting != nil {
		return rs.processDurableRouting(stream)
	}
	if rs.fileRequests != nil {
		return rs.processFileRequest(stream)
	}
	generation, err := rs.acquireCurrent()
	if err != nil {
		return err
	}
	defer generation.refs.Done()
	return generation.router.Process(stream)
}

func (rs *RouterService) acquireCurrent() (*routerGeneration, error) {
	rs.mu.Lock()
	generation := rs.current.Load()
	if generation == nil {
		rs.mu.Unlock()
		return nil, errors.New("router is shutting down")
	}
	generation.refs.Add(1)
	rs.mu.Unlock()
	return generation, nil
}

func (rs *RouterService) processDurableRouting(stream ext_proc.ExternalProcessor_ProcessServer) error {
	first, processDurableRoutingErr := stream.Recv()
	if processDurableRoutingErr != nil {
		return processDurableRoutingErr
	}
	headerMap, processDurableRoutingErr := durableRoutingRequestHeaders(first)
	if processDurableRoutingErr != nil {
		return processDurableRoutingErr
	}
	if durableRoutingInternalAuthenticated(headerMap) {
		generation, ok := durableRoutingInternalGeneration(headerMap)
		if !ok {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
		}
		lease, err := rs.durableRouting.resolveInternal(generation)
		if err != nil {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
		}
		defer lease.Release()
		grant, err := consumeDispatchGrant(
			headerMap, rs.durableRouting.dispatch, stream.Context(), generation,
			strings.TrimSpace(headerMapValue(headerMap, headers.RequestID)),
		)
		if err != nil {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
		}
		requestContext, err := routingcontext.WithGeneration(stream.Context(), generation)
		if err != nil {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
		}
		requestContext = withVerifiedDispatchGrant(requestContext, grant)
		return lease.Router.Process(&replayProcessStream{
			ExternalProcessor_ProcessServer: stream, ctx: requestContext, first: first,
		})
	}

	var (
		resolution     durableRoutingExternalResolution
		requestContext = stream.Context()
	)
	if rs.durableRouting.accessEnabled() {
		credential, ok := consumeBearerCredential(headerMap)
		if !ok {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnauthenticated)
		}
		var result quotaruntime.AccessCheckResult
		var resolveErr error
		resolution, result, resolveErr = rs.durableRouting.resolveExternal(stream.Context(), credential)
		if resolveErr != nil || !result.Allowed() || resolution.lease == nil {
			return rs.sendDurableRoutingInferenceError(stream, result.Disposition)
		}
		requestContext = withInferenceAuthentication(requestContext, resolution.authentication)
	} else {
		// Without native access has no inference identity. Remove any caller bearer
		// before replay so it can never become an accidental backend credential.
		_, _ = consumeBearerCredential(headerMap)
		var resolveErr error
		resolution, resolveErr = rs.durableRouting.resolvePublic()
		if resolveErr != nil || resolution.lease == nil {
			return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
		}
	}
	defer resolution.lease.Release()
	requestContext, processDurableRoutingErr = routingcontext.WithGeneration(requestContext, resolution.generation)
	if processDurableRoutingErr != nil {
		return rs.sendDurableRoutingInferenceError(stream, quotaruntime.AdmissionUnavailable)
	}
	return resolution.lease.Router.Process(&replayProcessStream{
		ExternalProcessor_ProcessServer: stream, ctx: requestContext, first: first,
	})
}

func (rs *RouterService) sendDurableRoutingInferenceError(
	stream ext_proc.ExternalProcessor_ProcessServer,
	disposition quotaruntime.AdmissionDisposition,
) error {
	generation, err := rs.acquireCurrent()
	if err != nil {
		return err
	}
	defer generation.refs.Done()
	return stream.Send(generation.router.createInferenceAccessError(disposition, nil))
}

func (rs *RouterService) Close() error {
	rs.mu.Lock()
	if !rs.closed {
		rs.closed = true
		generation := rs.current.Swap(nil)
		if generation != nil {
			rs.retired.Add(1)
			go rs.closeRetiredGeneration(generation)
		}
	}
	rs.mu.Unlock()
	rs.retired.Wait()
	rs.errMu.Lock()
	defer rs.errMu.Unlock()
	return errors.Join(rs.errors...)
}

func (rs *RouterService) closeRetiredGeneration(generation *routerGeneration) {
	defer rs.retired.Done()
	if err := closeRouterGeneration(generation); err != nil {
		rs.errMu.Lock()
		rs.errors = append(rs.errors, err)
		rs.errMu.Unlock()
		logging.ComponentErrorEvent("extproc", "retired_router_close_failed", map[string]interface{}{
			"error": err.Error(),
		})
	}
}

func closeRouterGeneration(generation *routerGeneration) error {
	if generation == nil || generation.router == nil {
		return nil
	}
	generation.refs.Wait()
	return generation.router.Close()
}

func (s *Server) configuredGRPCMaxMessageSize() int {
	cfg := resolveServerConfig(s)
	if cfg == nil {
		return (&config.LooperConfig{}).GetGRPCMaxMsgSize()
	}
	return cfg.Looper.GetGRPCMaxMsgSize()
}

func resolveServerConfig(s *Server) *config.RouterConfig {
	if s != nil && s.service != nil {
		if router := s.service.GetRouter(); router != nil && router.Config != nil {
			return router.Config
		}
	}
	if s != nil && s.runtime != nil {
		return s.runtime.CurrentConfig()
	}
	return config.Get()
}

func attachRuntimeRegistry(router *OpenAIRouter, runtimeRegistry *routerruntime.Registry) {
	if router == nil {
		return
	}
	router.RuntimeRegistry = runtimeRegistry
}

func publishRouterState(
	cfg *config.RouterConfig,
	router *OpenAIRouter,
	runtimeRegistry *routerruntime.Registry,
) {
	if router == nil {
		return
	}
	publishRouterLearningStateStore(router)
	if runtimeRegistry != nil {
		runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
			Config:                cfg,
			ClassificationService: router.ClassificationService,
			MemoryStore:           router.MemoryStore,
			ModelSelector:         router.ModelSelector,
			LearningRuntime:       router.routerLearningRuntimeState(),
			ReplayRuntime:         router,
			ResponseCache:         router.responseCacheService(),
			ContextCompression:    router.contextCompressionService(),
			CompressionRecovery:   router.CompressionRecovery,
		})
		return
	}
	memory.SetGlobalMemoryStore(router.MemoryStore)
}
