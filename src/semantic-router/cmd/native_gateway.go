package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"sort"
	"strconv"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// nativeDrainTimeout bounds how long in-flight native requests and streams may
// finish once shutdown starts.
const nativeDrainTimeout = 25 * time.Second

// startNativeGateway serves every configured listener through the routing core
// in process, with no ext_proc listener, until ctx ends or a listener fails.
// A non-empty bindAddress replaces every listener's configured address.
func startNativeGateway(ctx context.Context, server *extproc.Server, writer startupstatus.StatusWriter, bindAddress string) error {
	cfg := server.CurrentConfig()
	ready := &atomic.Bool{}
	engineOpts := routing.DefaultOptions
	// The upstream layer walks each call's fallback chain, so the response
	// phases never fall back a second time.
	engineOpts.ExecutesFallback = true
	listeners, err := listenNative(cfg, bindAddress, gateway.Options{
		Serving:   nativeServing{pin: server.Pin, engine: engineOpts},
		Ready:     ready.Load,
		AccessLog: logAccess,
	})
	if err != nil {
		return recordStartupError(writer, "start the standalone listeners", err)
	}
	if warning := config.UntrustedIdentityWarning(cfg); warning != "" {
		logging.ComponentWarnEvent("gateway", "standalone_identity_untrusted", map[string]interface{}{"message": warning})
	}
	failed := make(chan error, len(listeners))
	for _, l := range listeners {
		go func(l *nativeListener) {
			if serveErr := gateway.Serve(l.server, l.listener); serveErr != nil && !errors.Is(serveErr, http.ErrServerClosed) {
				failed <- fmt.Errorf("listener %s: %w", l.name, serveErr)
			}
		}(l)
	}
	served := make(chan error, 1)
	go func() {
		served <- server.StartContextWithoutExtProc(ctx, func() {
			ready.Store(true)
			if cfg.ConfigSource != config.ConfigSourceKubernetes {
				markRouterReady(writer, startupEmbeddingProviderStatus(server.EmbeddingRuntimeState()))
			}
		})
	}()
	select {
	case err = <-failed:
	case err = <-served:
	}
	ready.Store(false)
	drainCtx, cancel := context.WithTimeout(context.Background(), nativeDrainTimeout)
	defer cancel()
	for _, l := range listeners {
		err = errors.Join(err, l.server.Shutdown(drainCtx))
	}
	if err != nil {
		return recordStartupError(writer, "serve the standalone listeners", err)
	}
	return nil
}

type nativeListener struct {
	name     string
	server   *http.Server
	listener net.Listener
}

// listenNative binds every listener of cfg, on bindAddress when it is set; a
// failure closes those bound.
func listenNative(cfg *config.RouterConfig, bindAddress string, base gateway.Options) ([]*nativeListener, error) {
	if len(cfg.Listeners) == 0 {
		return nil, errors.New("standalone mode needs at least one listener in the configuration")
	}
	var bound []*nativeListener
	closeBound := func() {
		for _, l := range bound {
			_ = l.listener.Close()
		}
	}
	for _, spec := range cfg.Listeners {
		opts := base
		opts.Listener = spec.Name
		handler, err := gateway.NewHandler(opts)
		if err != nil {
			closeBound()
			return nil, err
		}
		serverOpts, err := listenerServerOptions(cfg, spec)
		if err != nil {
			closeBound()
			return nil, fmt.Errorf("listener %s: %w", spec.Name, err)
		}
		address := spec.Address
		if bindAddress != "" {
			address = bindAddress
		}
		ln, err := gateway.Listen(net.JoinHostPort(address, strconv.Itoa(spec.Port)), serverOpts)
		if err != nil {
			closeBound()
			return nil, fmt.Errorf("listener %s: %w", spec.Name, err)
		}
		bound = append(bound, &nativeListener{name: spec.Name, server: gateway.NewServer(handler, serverOpts), listener: ln})
		logging.ComponentEvent("router", "standalone_listener_started", map[string]interface{}{
			"listener": spec.Name, "address": ln.Addr().String(), "api_keys": len(spec.APIKeys) > 0,
			"models": spec.Models, "tls": serverOpts.TLS != nil,
		})
	}
	return bound, nil
}

// listenerServerOptions are a listener's connection settings: its idle
// timeout and, when it serves TLS, its certificate.
func listenerServerOptions(cfg *config.RouterConfig, spec config.Listener) (gateway.ServerOptions, error) {
	opts := gateway.ServerOptions{IdleTimeout: gateway.DefaultIdleTimeout}
	if spec.Timeout != "" {
		idle, err := time.ParseDuration(spec.Timeout)
		if err != nil {
			return opts, fmt.Errorf("invalid timeout %q: %w", spec.Timeout, err)
		}
		opts.IdleTimeout = idle
	}
	if spec.TLS != nil {
		certFile, keyFile := spec.TLS.Files(cfg.ConfigBaseDir)
		tlsConfig, err := gateway.LoadTLS(certFile, keyFile, certificateReloadLogger(spec.Name))
		if err != nil {
			return opts, fmt.Errorf("tls: %w", err)
		}
		opts.TLS = tlsConfig
	}
	return opts, nil
}

func certificateReloadLogger(listener string) func(error) {
	return func(err error) {
		if err != nil {
			logging.ComponentWarnEvent("router", "standalone_listener_certificate_reload_failed", map[string]interface{}{
				"listener": listener, "error": err.Error(),
			})
			return
		}
		logging.ComponentEvent("router", "standalone_listener_certificate_reloaded", map[string]interface{}{
			"listener": listener,
		})
	}
}

func logAccess(record gateway.AccessRecord) {
	logging.ComponentEventFunc("gateway", "access", func() map[string]interface{} {
		return map[string]interface{}{
			"start_time":            record.Start.UTC().Format(time.RFC3339Nano),
			"method":                record.Method,
			"path":                  record.Path,
			"protocol":              record.Protocol,
			"response_code":         record.Status,
			"bytes_received":        record.BytesReceived,
			"bytes_sent":            record.BytesSent,
			"duration_ms":           record.Duration.Milliseconds(),
			"upstream_service_time": record.UpstreamServiceTime.Milliseconds(),
			"x_forwarded_for":       record.ForwardedFor,
			"user_agent":            record.UserAgent,
			"request_id":            record.RequestID,
			"authority":             record.Authority,
			"upstream_host":         record.UpstreamHost,
		}
	})
}

// nativeServing pins one router generation per request: its routing pipeline,
// the upstream set its configuration snapshot owns, and the listener's keys
// from the same snapshot.
type nativeServing struct {
	pin    func() (*extproc.Lease, error)
	engine routing.Options
}

func (s nativeServing) Pin(_ context.Context, listener string) (gateway.Serving, func(), error) {
	lease, err := s.pin()
	if err != nil {
		return gateway.Serving{}, nil, err
	}
	cfg := lease.Snapshot.Config()
	nativeListener, _ := systemone.SelectListener(cfg.Listeners, listener)
	models, _ := lease.Snapshot.Part(configsnapshot.ComponentModelService).(*extproc.FrontendModels)
	serving := gateway.Serving{
		SystemOne: systemone.Handler(cfg, nativeListener, func(ctx context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
			result, err := models.SystemOne(ctx, deployment, body)
			return result.Status, result.Body, err
		}),
		RoutingDisabled: !cfg.RoutingEnabled(),
		APIKeys:         listenerAPIKeys(cfg, listener),
		Models:          listenerModels(cfg, listener),
		IdentityHeaders: []string{cfg.Authz.Identity.GetUserIDHeader(), cfg.Authz.Identity.GetUserGroupsHeader()},
		TrustIdentity:   listenerIdentityTrust(cfg, listener),
	}
	if cfg.RoutingEnabled() {
		set, ok := lease.Snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
		if !ok || lease.Router == nil {
			lease.Release()
			return gateway.Serving{}, nil, errors.New("the serving configuration has no routing pipeline")
		}
		serving.Engine = routing.NewEngine(lease.Router, s.engine)
		serving.Upstream = set
	}
	return serving, lease.Release, nil
}

func listenerAPIKeys(cfg *config.RouterConfig, name string) []string {
	for _, listener := range cfg.Listeners {
		if listener.Name == name {
			return listener.APIKeys
		}
	}
	return nil
}

func listenerModels(cfg *config.RouterConfig, name string) []string {
	for _, listener := range cfg.Listeners {
		if listener.Name == name {
			return listener.Models
		}
	}
	return nil
}

// listenerIdentityTrust is the identity trust of the named listener in cfg,
// which config validation has already checked.
func listenerIdentityTrust(cfg *config.RouterConfig, name string) gateway.IdentityTrust {
	for _, listener := range cfg.Listeners {
		if listener.Name != name || listener.Identity == nil {
			continue
		}
		peers, err := listener.Identity.PeerPrefixes()
		if err != nil {
			return gateway.IdentityTrust{}
		}
		return gateway.IdentityTrust{Headers: listener.Identity.TrustHeaders, Peers: peers}
	}
	return gateway.IdentityTrust{}
}

// upstreamPart makes the upstream layer part of every configuration snapshot,
// in both gateway modes: request graphs send their hops through it in either
// mode, and the standalone frontend sends client requests through it too. A
// snapshot whose endpoints, clusters and listeners are unchanged keeps its
// predecessor's set; otherwise it builds one that adopts the unchanged
// clusters and warms it before serving. A set closes when the last generation
// that serves it has drained. A standalone Router also refuses a reload that
// changes the listeners it bound.
func upstreamPart(mode config.GatewayMode) configsnapshot.PartBuilder {
	part := configsnapshot.PartBuilder{
		Component: configsnapshot.ComponentUpstream,
		Build: func(_ context.Context, candidate *configsnapshot.Snapshot, previous configsnapshot.Part) (configsnapshot.Part, error) {
			prev, _ := previous.(*upstream.Set)
			if !candidate.Config().RoutingEnabled() {
				return upstream.New(upstream.Topology{}, upstream.Options{})
			}
			set, err := buildUpstream(mode, candidate.Config(), prev)
			if err != nil && mode != config.GatewayStandalone {
				// Envoy carries the client traffic and accepts backends the
				// in-process client refuses, so only request-graph hops fail.
				logging.ComponentWarnEvent("router", "request_graph_upstream_unavailable", map[string]interface{}{
					"error": err.Error(),
					"fix":   "request-graph hops call providers.models backends in process; fix the backend the error names",
				})
				return upstream.New(upstream.Topology{}, upstream.Options{})
			}
			if err != nil {
				return nil, err
			}
			return set, nil
		},
		Warm: func(ctx context.Context, part configsnapshot.Part) error {
			return part.(*upstream.Set).Warm(ctx)
		},
	}
	if mode == config.GatewayStandalone {
		part.Validate = validateNativeListeners
	}
	return part
}

// buildUpstream builds one snapshot's upstream set. Active health checks
// belong to whoever owns the data path: the Router in standalone mode, Envoy
// in ext_proc mode, where the Router's hops keep passive outlier detection
// only.
func buildUpstream(mode config.GatewayMode, cfg *config.RouterConfig, previous *upstream.Set) (*upstream.Set, error) {
	topology, err := upstream.Compile(cfg)
	if err != nil {
		return nil, err
	}
	if mode != config.GatewayStandalone {
		topology = topology.WithoutActiveHealthChecks()
	}
	return upstream.New(topology, upstream.Options{Previous: previous})
}

// validateNativeListeners refuses a reload that changes what the native
// gateway bound at startup: the set of listeners, or a listener's address,
// port, timeout (also its connections' idle timeout) or TLS certificate.
func validateNativeListeners(candidate, active *configsnapshot.Snapshot) error {
	if active == nil {
		return nil
	}
	bound := make(map[string]config.Listener, len(active.Config().Listeners))
	for _, listener := range active.Config().Listeners {
		bound[listener.Name] = listener
	}
	var reasons []configsnapshot.Reason
	restart := func(path, message string) {
		reasons = append(reasons, configsnapshot.Reason{
			Stage: configsnapshot.StageValidate, Code: configsnapshot.CodeRestartRequired, Path: path,
			Message: message + "; standalone mode binds listeners at startup, so restart the Router to apply it",
		})
	}
	for _, next := range candidate.Config().Listeners {
		path := "listeners[" + next.Name + "]"
		previous, ok := bound[next.Name]
		delete(bound, next.Name)
		switch {
		case !ok:
			restart(path, "the listener is new")
		case next.Address != previous.Address:
			restart(path+".address", fmt.Sprintf("the address changes from %q to %q", previous.Address, next.Address))
		case next.Port != previous.Port:
			restart(path+".port", fmt.Sprintf("the port changes from %d to %d", previous.Port, next.Port))
		case next.Timeout != previous.Timeout:
			restart(path+".timeout", fmt.Sprintf("the timeout changes from %q to %q", previous.Timeout, next.Timeout))
		case !sameListenerTLS(next.TLS, previous.TLS):
			restart(path+".tls", "the TLS certificate changes")
		}
	}
	for _, removed := range sortedListenerNames(bound) {
		restart("listeners["+removed+"]", "the listener is removed")
	}
	if len(reasons) == 0 {
		return nil
	}
	return configsnapshot.RejectReasons(configsnapshot.StageValidate, reasons)
}

func sameListenerTLS(a, b *config.ListenerTLS) bool {
	if a == nil || b == nil {
		return a == b
	}
	return *a == *b
}

func sortedListenerNames(listeners map[string]config.Listener) []string {
	names := make([]string, 0, len(listeners))
	for name := range listeners {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}
