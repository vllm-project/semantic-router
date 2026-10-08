package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/netip"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// Upstream sends a planned call to the model backends, its fallback chain
// included; *upstream.Set implements it.
type Upstream interface {
	Execute(ctx context.Context, call *routing.Call, listener string) (*upstream.Result, error)
}

// Serving is what one request is served with, all from one configuration.
type Serving struct {
	// SystemOne serves native decision inference under this generation's own
	// listener grant. It does not enter the Chat routing pipeline.
	SystemOne http.Handler
	Engine    routing.Engine
	Upstream  Upstream
	// APIKeys, when set, are the only client keys the listener accepts.
	APIKeys []string
	// Models, when set, are the only request models the listener accepts;
	// the routing core enforces them on the model it parses.
	Models []string
	// IdentityHeaders are dropped from client requests in addition to the
	// built-in x-authz-* identity headers: the names the configuration reads
	// a client identity from, which no authenticator in front asserts.
	IdentityHeaders []string
	// TrustIdentity, when it trusts a request, keeps those headers: the
	// listener sits behind an authenticating proxy or a trusted application.
	TrustIdentity IdentityTrust
}

// IdentityTrust is a listener's trust in client identity headers.
type IdentityTrust struct {
	// Headers keeps the identity headers instead of dropping them.
	Headers bool
	// Peers, when set, keeps them only on connections from these networks.
	Peers []netip.Prefix
}

// trusts reports whether r's identity headers are kept. It reads the TCP
// peer, never X-Forwarded-For, which any client can set.
func (t IdentityTrust) trusts(r *http.Request) bool {
	if !t.Headers {
		return false
	}
	if len(t.Peers) == 0 {
		return true
	}
	peer, err := netip.ParseAddrPort(r.RemoteAddr)
	if err != nil {
		return false
	}
	for _, prefix := range t.Peers {
		if prefix.Contains(peer.Addr().Unmap()) {
			return true
		}
	}
	return false
}

// Pinner pins the Serving of one request on a listener. The handler calls
// release once the response, a streamed body included, is written, so a
// configuration change never splits a request, its fallback chain included.
type Pinner interface {
	Pin(ctx context.Context, listener string) (serving Serving, release func(), err error)
}

// Static serves every request with the same Serving.
func Static(serving Serving) Pinner { return staticServing(serving) }

type staticServing Serving

func (s staticServing) Pin(context.Context, string) (Serving, func(), error) {
	return Serving(s), func() {}, nil
}

// Options configure a Handler.
type Options struct {
	Serving Pinner
	// Listener names the listener whose route defaults the upstream layer
	// applies.
	Listener string
	// MaxRequestBodyBytes bounds a request body. The local Envoy template
	// buffers up to 500 MiB per connection.
	MaxRequestBodyBytes int64
	// Ready reports whether the gateway can serve; /ready answers 503 until
	// it does and again while it drains. Nil means always ready.
	Ready func() bool
	// AccessLog, when set, receives one record per request.
	AccessLog func(AccessRecord)
}

// DefaultMaxRequestBodyBytes matches the local Envoy template's buffer limit.
const DefaultMaxRequestBodyBytes = 500 << 20

// Handler serves inference traffic through the routing core.
type Handler struct {
	opts Options
}

// NewHandler returns a Handler for opts.
func NewHandler(opts Options) (*Handler, error) {
	if opts.Serving == nil {
		return nil, errors.New("gateway: a serving source is required")
	}
	if opts.MaxRequestBodyBytes <= 0 {
		opts.MaxRequestBodyBytes = DefaultMaxRequestBodyBytes
	}
	return &Handler{opts: opts}, nil
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if h.serveProbe(w, r) {
		return
	}
	cw := &countingWriter{ResponseWriter: w}
	x := &exchange{record: newAccessRecord(r)}
	defer h.logAccess(x, cw)
	h.serveRequest(cw, r, x)
}

func (h *Handler) serveRequest(w http.ResponseWriter, r *http.Request, x *exchange) {
	ctx := r.Context()
	serving, release, err := h.opts.Serving.Pin(ctx, h.opts.Listener)
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "unavailable", "The router is not serving.")
		return
	}
	defer release()
	if r.URL.Path == "/v1/systemone" || r.URL.Path == "/v1/decisions" || r.URL.Path == "/v1/systemone/models" {
		if serving.SystemOne == nil {
			http.NotFound(w, r)
			return
		}
		serving.SystemOne.ServeHTTP(w, r)
		return
	}
	if !newAPIKeys(serving.APIKeys).authorize(r) {
		writeUnauthorized(w)
		return
	}
	ctx = routing.WithListenerModels(ctx, serving.Models)
	body, err := readBody(r, h.opts.MaxRequestBodyBytes)
	if errors.Is(err, errBodyTooLarge) {
		writeError(w, http.StatusRequestEntityTooLarge, "request_too_large", "The request body is too large.")
		return
	}
	if err != nil {
		return
	}
	var strip map[string]bool
	if !serving.TrustIdentity.trusts(r) {
		strip = headerSet(append(append([]string(nil), identityHeaders...), serving.IdentityHeaders...))
	}
	req := routingRequest(r, body, strip)
	x.record.RequestID = req.Header.Get("x-request-id")
	plan, err := serving.Engine.Plan(ctx, req)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "routing_error", "The router could not process the request.")
		return
	}
	finishErr := h.serve(ctx, w, serving, plan, x)
	plan.Finish(finishErr)
	if finishErr != nil && ctx.Err() == nil && errors.Is(finishErr, errResponseStarted) {
		// The headers are out, so the client learns of the failure the way
		// Envoy tells it: the stream is reset rather than ended cleanly.
		panic(http.ErrAbortHandler)
	}
}

var errResponseStarted = errors.New("the response failed after it started")

// serve answers a plan and returns why the exchange ended early, or nil.
func (h *Handler) serve(ctx context.Context, w http.ResponseWriter, serving Serving, plan *routing.Plan, x *exchange) error {
	if plan.Immediate != nil {
		return transportError(ctx, writeResponse(w, plan.Immediate))
	}
	resp, closeBody, err := h.forward(ctx, serving, plan, x)
	if err != nil {
		if ctx.Err() == nil {
			writeError(w, http.StatusInternalServerError, "routing_error", "The router could not process the response.")
		}
		return transportError(ctx, err)
	}
	defer closeBody()
	return transportError(ctx, writeResponse(w, resp))
}

// forward sends the planned call upstream and runs the response phases. The
// caller closes the upstream body once the client response is written.
//
// A local reply skips the response phases: Envoy's ext_proc ends processing
// on Envoy's own local replies, so behind Envoy the client gets the reply as
// it is, and the session ends as the ext_proc stream then does.
func (h *Handler) forward(ctx context.Context, serving Serving, plan *routing.Plan, x *exchange) (*routing.Response, func(), error) {
	sent := time.Now()
	result, err := serving.Upstream.Execute(ctx, plan.Call, h.opts.Listener)
	if err != nil {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		x.record.UpstreamServiceTime = time.Since(sent)
		return verbatim(localReply(err)), func() {}, nil
	}
	x.record.UpstreamServiceTime = time.Since(sent)
	if result.Immediate != nil {
		// The fallback chain answered without a backend.
		return result.Immediate, func() {}, nil
	}
	resp := result.Response
	if resp.Endpoint.Host != "" {
		x.record.UpstreamHost = resp.Endpoint.Address()
	}
	closeBody := func() { _ = resp.Body.Close() }
	if resp.Local != nil {
		return verbatim(resp), closeBody, nil
	}
	out, err := serving.Engine.Respond(ctx, plan, &routing.UpstreamResponse{
		Status: resp.StatusCode,
		Header: routingHeader(resp.Header),
		Body:   responseBody(plan.Call, resp),
	})
	if err != nil {
		closeBody()
		return nil, nil, err
	}
	return out, closeBody, nil
}

// responseBody is nil for a response that has no body, as Envoy then ends the
// stream with the headers.
func responseBody(call *routing.Call, resp *upstream.Response) io.Reader {
	switch {
	case call.Request.Header.Get(":method") == http.MethodHead,
		resp.StatusCode == http.StatusNoContent, resp.StatusCode == http.StatusNotModified,
		resp.Header.Get("Content-Length") == "0":
		return nil
	default:
		return resp.Body
	}
}

// localReply stands in for a failure the upstream layer reports as an error
// rather than as Envoy's local reply, so the response phases still see it.
func localReply(err error) *upstream.Response {
	status := http.StatusServiceUnavailable
	var upstreamErr *upstream.Error
	if errors.As(err, &upstreamErr) {
		status = upstreamErr.StatusCode()
	}
	text := http.StatusText(status)
	return &upstream.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {"text/plain"}, "Content-Length": {strconv.Itoa(len(text))}},
		Body:       io.NopCloser(strings.NewReader(text)),
	}
}

// verbatim is a local reply as the client receives it, with no response
// phase applied. Local replies are short plain text, so the body is read
// whole.
func verbatim(resp *upstream.Response) *routing.Response {
	body, _ := io.ReadAll(resp.Body)
	return &routing.Response{Status: resp.StatusCode, Header: routingHeader(resp.Header), Body: body}
}

func routingHeader(h http.Header) routing.Header {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make(routing.Header, 0, len(names))
	for _, name := range names {
		for _, value := range h[name] {
			out = append(out, routing.HeaderField{Name: strings.ToLower(name), Value: value})
		}
	}
	return out
}

// writeResponse writes a client response; a streamed body is flushed chunk by
// chunk. A failure after the headers went out wraps errResponseStarted.
func writeResponse(w http.ResponseWriter, resp *routing.Response) error {
	header := w.Header()
	for _, field := range resp.Header {
		header.Add(field.Name, field.Value)
	}
	if !resp.Header.Has("content-type") {
		// Envoy sends no content type the response lacks; Go would sniff one.
		header["Content-Type"] = nil
	}
	w.WriteHeader(resp.Status)
	if resp.Stream == nil {
		if _, err := w.Write(resp.Body); err != nil {
			return errors.Join(errResponseStarted, err)
		}
		return nil
	}
	controller := http.NewResponseController(w)
	if err := controller.Flush(); err != nil {
		return errors.Join(errResponseStarted, err)
	}
	for {
		chunk, err := resp.Stream.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return errors.Join(errResponseStarted, err)
		}
		if _, err := w.Write(chunk); err != nil {
			return errors.Join(errResponseStarted, err)
		}
		if err := controller.Flush(); err != nil {
			return errors.Join(errResponseStarted, err)
		}
	}
}

// transportError reports why the exchange ended early: the client's
// cancellation or deadline wins over the write error it caused.
func transportError(ctx context.Context, err error) error {
	if err == nil {
		return nil
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return ctxErr
	}
	return err
}

var errBodyTooLarge = errors.New("request body too large")

func readBody(r *http.Request, limit int64) ([]byte, error) {
	if r.Body == nil || r.Body == http.NoBody {
		return nil, nil
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, err
	}
	if int64(len(body)) > limit {
		return nil, errBodyTooLarge
	}
	return body, nil
}

// writeError answers with an OpenAI-style error the gateway itself produces.
func writeError(w http.ResponseWriter, status int, code, message string) {
	kind := "server_error"
	if status < http.StatusInternalServerError {
		kind = "invalid_request_error"
	}
	body, _ := json.Marshal(map[string]map[string]string{
		"error": {"message": message, "type": kind, "code": code},
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
