package extproc

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modeldownload"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var errServerShuttingDown = errors.New("router server is shutting down")

// ServerOption configures a Server.
type ServerOption func(*serverOptions)

type serverOptions struct {
	historyLimit int
	parts        []configsnapshot.PartBuilder
	gateway      config.GatewayMode
}

// WithGatewayMode names the gateway mode the server serves in
// (config.GatewayExtProc unless set). A reload whose configuration needs a
// capability that mode lacks is rejected as unsupported.
func WithGatewayMode(mode config.GatewayMode) ServerOption {
	return func(o *serverOptions) { o.gateway = mode }
}

// WithConfigHistoryLimit sets how many configuration versions the server's
// history keeps (configsnapshot.DefaultHistoryLimit unless set).
func WithConfigHistoryLimit(limit int) ServerOption {
	return func(o *serverOptions) { o.historyLimit = limit }
}

// WithConfigParts makes every configuration snapshot own the parts the
// builders make, such as the upstream layer the native gateway sends through.
func WithConfigParts(builders ...configsnapshot.PartBuilder) ServerOption {
	return func(o *serverOptions) { o.parts = append(o.parts, builders...) }
}

// configManager returns the server's configuration lifecycle. Servers built
// without NewServer get one with an in-memory history on first use.
func (s *Server) configManager() *configsnapshot.Manager {
	s.configsOnce.Do(func() {
		if s.configs == nil {
			s.configs = s.newConfigManager(nil)
		}
		s.runtime.SetConfigLifecycle(s.configs)
	})
	return s.configs
}

func (s *Server) newConfigManager(history *configsnapshot.History, parts ...configsnapshot.PartBuilder) *configsnapshot.Manager {
	opts := configsnapshot.Options{Runtime: routerRuntime{s}, History: history, Parts: parts}
	if s.runtime != nil {
		opts.Reporter = s.runtime
	}
	return configsnapshot.NewManager(opts)
}

// openConfigHistory opens the history persisted beside the configuration the
// server loads. The history is best effort: when it cannot be read, the
// server keeps one in memory and reports why.
func openConfigHistory(configPath string, limit int) *configsnapshot.History {
	dir := configsnapshot.ResolvePersistence(configPath).HistoryDir
	history, err := configsnapshot.NewHistory(limit, configsnapshot.NewDirStore(dir))
	if err != nil {
		logging.ComponentWarnEvent("extproc", "config_history_unavailable", map[string]interface{}{
			"dir":   dir,
			"error": err.Error(),
		})
	}
	if history == nil {
		history, _ = configsnapshot.NewHistory(configsnapshot.DefaultHistoryLimit, nil)
	}
	return history
}

// ConfigSnapshots returns the lifecycle that owns this server's configuration.
func (s *Server) ConfigSnapshots() *configsnapshot.Manager { return s.configManager() }

// RejectConfigUpdate records an update its source rejected before handing it
// to the lifecycle. Errors the lifecycle produced are already recorded.
func (s *Server) RejectConfigUpdate(source configsnapshot.Source, cfg *config.RouterConfig, err error) error {
	return s.configManager().Reject(configsnapshot.Update{Origin: configsnapshot.Origin{Source: source}, Config: cfg}, err)
}

// sourceUpdate is the lifecycle update for a candidate from source. A file
// candidate carries the bytes the loader parsed and stays current while the
// file still holds them.
func sourceUpdate(source, path string, cfg *config.RouterConfig) configsnapshot.Update {
	update := configsnapshot.Update{Origin: configsnapshot.Origin{Source: configsnapshot.Source(source)}, Config: cfg}
	if update.Origin.Source == configsnapshot.SourceFile {
		update.Document = parsedDocument(path, cfg)
		update.Current = func() error { return checkFileReloadCandidate(path, cfg) }
	}
	return update
}

// parsedDocument returns the file's bytes when they are the document cfg was
// parsed from, and nil once the file holds another one.
func parsedDocument(path string, cfg *config.RouterConfig) []byte {
	if cfg == nil || cfg.DocumentHash == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil || documentDigest(data) != cfg.DocumentHash {
		return nil
	}
	return data
}

func documentDigest(data []byte) string {
	digest := sha256.Sum256(data)
	return hex.EncodeToString(digest[:])
}

// gatewayMode is the mode the server serves in; servers built without
// NewServer run behind Envoy.
func (s *Server) gatewayMode() config.GatewayMode {
	if s.gateway == "" {
		return config.GatewayExtProc
	}
	return s.gateway
}

// checkGatewayCapabilities rejects a candidate that needs a capability mode
// lacks, with one reason per use.
func checkGatewayCapabilities(candidate *config.RouterConfig, mode config.GatewayMode) error {
	violations := config.CheckGatewayCapabilities(candidate, mode)
	if len(violations) == 0 {
		return nil
	}
	reasons := make([]configsnapshot.Reason, 0, len(violations))
	for _, violation := range violations {
		reasons = append(reasons, configsnapshot.Reason{
			Stage: configsnapshot.StageValidate, Code: configsnapshot.CodeUnsupported,
			Path: violation.Path, Message: violation.Message,
		})
	}
	return configsnapshot.RejectReasons(configsnapshot.StageValidate, reasons)
}

// routerRuntime prepares and activates router generations for the lifecycle.
type routerRuntime struct{ s *Server }

func (r routerRuntime) Validate(_ context.Context, c *configsnapshot.Candidate) error {
	s := r.s
	if s.lifecycle.isStopping() {
		return configsnapshot.Reject(configsnapshot.StageValidate, configsnapshot.CodeShuttingDown, errServerShuttingDown)
	}
	candidate := c.Snapshot().Config()
	if err := config.ValidateRoutingPreviewReload(resolveServerConfig(s), candidate); err != nil {
		return configsnapshot.Reject(configsnapshot.StageValidate, configsnapshot.CodeRestartRequired, err)
	}
	if err := checkGatewayCapabilities(candidate, s.gatewayMode()); err != nil {
		return err
	}
	c.Step("artifacts")
	if err := modeldownload.ValidateReloadArtifacts(resolveServerConfig(s), candidate); err != nil {
		return configsnapshot.Reject(configsnapshot.StageValidate, configsnapshot.CodeArtifactUnavailable,
			fmt.Errorf("model artifact reload preflight failed: %w", err))
	}
	return c.Current()
}

func (r routerRuntime) Warm(ctx context.Context, c *configsnapshot.Candidate) (configsnapshot.Warmed, error) {
	s := r.s
	candidate := c.Snapshot().Config()
	// The Kubernetes source prepares models before it hands a candidate over,
	// reporting startup progress while it does.
	if c.Snapshot().Origin().Source != configsnapshot.SourceKubernetes {
		c.Step("model_download")
		if err := ensureReloadConfigModels(candidate); err != nil {
			return nil, configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeModelUnavailable,
				fmt.Errorf("model download preflight failed: %w", err))
		}
	}
	c.Step("model_prepare")
	router, err := s.buildCandidateRouter(c)
	if err != nil {
		return nil, configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeBuildFailed, err)
	}
	attachRuntimeRegistry(router, s.runtime)
	c.Step("warmup")
	if err := warmupReloadRouter(router); err != nil {
		_ = router.Close()
		return nil, configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeWarmupFailed,
			fmt.Errorf("runtime warmup failed: %w", err))
	}
	if err := ctx.Err(); err != nil {
		_ = router.Close()
		return nil, err
	}
	if s.lifecycle.isStopping() {
		_ = router.Close()
		return nil, configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeShuttingDown, errServerShuttingDown)
	}
	return &warmedRouter{s: s, candidate: c, router: router}, nil
}

// buildCandidateRouter builds the candidate's router. When the candidate's
// signal resources equal those the serving router was built from, the new
// router shares its signal runtime, so classifiers, embeddings and model
// runtime deployments are not built again.
func (s *Server) buildCandidateRouter(c *configsnapshot.Candidate) (*OpenAIRouter, error) {
	key := c.Snapshot().ComponentKey(configsnapshot.ComponentSignals)
	var router *OpenAIRouter
	var err error
	if shared, ok := s.service.GetRouter().signalRuntime().sharedWith(key); ok {
		router, err = buildReloadRouterSharingSignals(c.Snapshot().Config(), s.modelPool, shared)
		if err == nil {
			c.RecordReuse(configsnapshot.ComponentSignals)
		}
	} else {
		router, err = buildReloadRouter(c.Snapshot().Config(), s.modelPool)
	}
	if err != nil {
		return nil, err
	}
	router.nameSignals(key)
	return router, nil
}

// signalRuntime is the router's signal runtime, or nil.
func (r *OpenAIRouter) signalRuntime() *signalRuntime {
	if r == nil {
		return nil
	}
	return r.signals
}

// nameSignals records what the router's signal runtime was built from, so a
// later generation built from the same signal resources can share it.
func (r *OpenAIRouter) nameSignals(key string) {
	if r != nil && r.signals != nil {
		r.signals.key = key
	}
}

// warmedRouter is a built and warmed generation waiting to serve.
type warmedRouter struct {
	s         *Server
	candidate *configsnapshot.Candidate
	router    *OpenAIRouter
}

func (w *warmedRouter) Activate(context.Context) error {
	s := w.s
	snapshot := w.candidate.Snapshot()
	candidate := snapshot.Config()
	source := snapshot.Origin().Source
	w.candidate.Step("publication")
	if source == configsnapshot.SourceFile {
		release := s.runtime.LockConfigPublication()
		if err := w.candidate.Current(); err != nil {
			release()
			if closeErr := w.router.Close(); closeErr != nil {
				return fmt.Errorf("close discarded config generation: %w", closeErr)
			}
			return err
		}
		defer release()
	}
	inheritRouterLearningState(s.service.GetRouter(), w.router)

	// Registry-backed generations publish only after a successful swap.
	if source != configsnapshot.SourceKubernetes && s.runtime == nil {
		replaceReloadConfig(candidate)
	}
	logLoadedRouterConfig(s.configPath, candidate)
	router := w.router
	err := s.service.swapSnapshot(router, snapshot, func(acquire AcquireFunc) {
		if router != nil {
			router.WorkflowStateService.CommitStorePolicy()
		}
		publishSnapshotState(candidate, snapshot, router, s.runtime, acquire)
	})
	if err != nil {
		return configsnapshot.Reject(configsnapshot.StageActivate, configsnapshot.CodeShuttingDown, err)
	}
	return nil
}

func (w *warmedRouter) Discard() { _ = w.router.Close() }
