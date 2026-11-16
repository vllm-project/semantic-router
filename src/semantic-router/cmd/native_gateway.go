//go:build !windows

package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
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
		Ready:     func() bool { return ready.Load() && server.ServingActive() },
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
	err = errors.Join(err, drainNative(server.CurrentConfig().DrainTimeout(), listeners))
	if err != nil {
		return recordStartupError(writer, "serve the standalone listeners", err)
	}
	return nil
}

type nativeListener struct {
	name     string
	server   *http.Server
	handler  *gateway.Handler
	listener net.Listener
}

// drainNative stops every listener accepting and lets the requests in flight
// on them finish, for at most timeout; those still in flight then are cut
// when the process exits.
func drainNative(timeout time.Duration, listeners []*nativeListener) error {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	started, inFlight := time.Now(), 0
	for _, l := range listeners {
		inFlight += l.handler.InFlight()
	}
	logging.ComponentEvent("gateway", "standalone_drain_started", map[string]interface{}{
		"drain_timeout_seconds": int(timeout / time.Second), "in_flight": inFlight,
	})
	type drained struct {
		left int
		err  error
	}
	results := make(chan drained, len(listeners))
	for _, l := range listeners {
		go func(l *nativeListener) {
			// Shutdown waits for HTTP/1.1 and TLS requests only; Drain waits
			// for those of every protocol.
			err := l.server.Shutdown(ctx)
			if errors.Is(err, context.DeadlineExceeded) {
				err = nil
			} else if err != nil {
				err = fmt.Errorf("listener %s: %w", l.name, err)
			}
			results <- drained{left: l.handler.Drain(ctx), err: err}
		}(l)
	}
	var (
		left int
		err  error
	)
	for range listeners {
		result := <-results
		left += result.left
		err = errors.Join(err, result.err)
	}
	if left > 0 {
		logging.ComponentWarnEvent("gateway", "standalone_drain_incomplete", map[string]interface{}{
			"drain_timeout_seconds": int(timeout / time.Second), "in_flight": left,
		})
	} else {
		logging.ComponentEvent("gateway", "standalone_drain_completed", map[string]interface{}{
			"duration_ms": time.Since(started).Milliseconds(),
		})
	}
	return err
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
		serverOpts, err := listenerServerOptions(cfg, spec)
		if err != nil {
			closeBound()
			return nil, fmt.Errorf("listener %s: %w", spec.Name, err)
		}
		opts := base
		opts.Listener = spec.Name
		opts.ReadTimeout = serverOpts.IdleTimeout
		handler, err := gateway.NewHandler(opts)
		if err != nil {
			closeBound()
			return nil, err
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
		bound = append(bound, &nativeListener{name: spec.Name, server: gateway.NewServer(handler, serverOpts), handler: handler, listener: ln})
		logging.ComponentEvent("router", "standalone_listener_started", map[string]interface{}{
			"listener": spec.Name, "address": ln.Addr().String(), "api_keys": len(spec.APIKeys) > 0,
			"apis": spec.ServedAPIs(), "tls": serverOpts.TLS != nil,
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
	remote, _ := lease.Snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	local := func(ctx context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
		result, err := models.SystemOne(ctx, deployment, body)
		return result.Status, result.Body, err
	}
	var nativeRouter systemone.Router
	var rules systemone.ListenerRules
	if lease.Router != nil {
		nativeRouter, rules = lease.Router, lease.Router
	}
	serving := gateway.Serving{
		SystemOne:       systemone.ForClients(cfg, systemone.HandlerWithRules(cfg, nativeListener, systemone.ServingInvoker(cfg, listener, nativeRouter, local, remote, models), rules)),
		RoutingDisabled: !cfg.RoutingEnabled(),
		APIKeys:         listenerAPIKeys(cfg, listener),
		IdentityHeaders: []string{
			cfg.Authz.Identity.GetUserIDHeader(), cfg.Authz.Identity.GetUserGroupsHeader(),
			cfg.Authz.DelegatedCredential.GetHeader(),
		},
		TrustIdentity: listenerIdentityTrust(cfg, listener),
	}
	applyListenerAPIs(&serving, cfg, listener)
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

// applyListenerAPIs gives serving the named listener's API grant: the APIs it
// serves, the models each accepts and what GET /v1/models lists. A listener
// that names no apis serves the default APIs on every path, as before apis.
func applyListenerAPIs(serving *gateway.Serving, cfg *config.RouterConfig, name string) {
	serving.ListsLLM = true
	for _, listener := range cfg.Listeners {
		if listener.Name != name {
			continue
		}
		serving.Models, _ = listener.LLMModels()
		serving.ListsLLM = servesAnyLLM(listener)
		serving.ListsSystemOne = listener.Serves(apiregistry.SystemOne)
		serving.Gate = listenerGate(listener)
		return
	}
}

func servesAnyLLM(listener config.Listener) bool {
	for _, name := range listener.ServedAPIs() {
		if api, ok := apiregistry.Lookup(name); ok && api.Kind == apiregistry.KindLLM {
			return true
		}
	}
	return false
}

// listenerGate answers which models a listener accepts on a request path.
// With apis, a path of an API it does not serve, or of no API, answers 404
// with a detail naming the listener and the APIs it serves.
func listenerGate(listener config.Listener) func(string) ([]string, string, bool) {
	refuse := func(path string) ([]string, string, bool) {
		return nil, fmt.Sprintf("listener %q does not serve %s; it serves %s", listener.Name, path,
			strings.Join(listener.ServedAPIs(), ", ")), false
	}
	return func(path string) ([]string, string, bool) {
		api, isAPI := apiregistry.ForPath(path)
		if !isAPI {
			if listener.APIs == nil {
				return nil, "", true
			}
			return refuse(path)
		}
		entry, served := listener.API(api.Name)
		if !served {
			if listener.APIs == nil && api.Name == apiregistry.SystemOne {
				// The System One API answers its own 404 there, as before.
				return nil, "", true
			}
			return refuse(path)
		}
		return entry.Models, "", true
	}
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
// port, timeout (also its connections' idle timeout) or TLS file paths.
func validateNativeListeners(candidate, active *configsnapshot.Snapshot) error {
	if active == nil {
		return nil
	}
	var reasons []configsnapshot.Reason
	for _, restart := range config.ListenerRestarts(active.Config().Listeners, candidate.Config().Listeners) {
		reasons = append(reasons, configsnapshot.Reason{
			Stage: configsnapshot.StageValidate, Code: configsnapshot.CodeRestartRequired, Path: restart.Path,
			Message: restart.Message + "; standalone mode binds listeners at startup, so restart the Router to apply it",
		})
	}
	if len(reasons) == 0 {
		return nil
	}
	return configsnapshot.RejectReasons(configsnapshot.StageValidate, reasons)
}
