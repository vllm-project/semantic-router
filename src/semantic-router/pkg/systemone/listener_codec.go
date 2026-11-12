package systemone

import (
	"bytes"
	"net/http"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apiregistry"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
)

// listenerDiagnostics counts what a listener's codec read and how it sent each
// request on: the vllm-sr/4.0 forms its clients used (action legacy_form, by
// form) and its plan for each request (field request, action passed_through
// or translated). A vllm-sr/4.0 listener whose clients no longer use a
// legacy form can switch to vllm-sr/5.0.
var listenerDiagnostics = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "vsr_systemone_listener_diagnostics_total",
	Help: "System One requests a listener read, by schema, field and action: the vllm-sr/4.0 forms clients used (legacy_form) and each request's plan (passed_through or translated).",
}, []string{"listener", "schema", "field", "action"})

// listenerCodec is the codec of the schema a listener speaks on its systemone
// API: vllm-sr/5.0 unless the listener names vllm-sr/4.0.
func listenerCodec(listener *routerconfig.Listener) systemonecodec.ListenerCodec {
	id := systemonecodec.SchemaVLLMSR50
	if listener != nil {
		if grant, ok := listener.API(apiregistry.SystemOne); ok && grant.Schema != "" {
			id = systemonecodec.SchemaID(grant.Schema)
		}
	}
	codec, err := systemonecodec.Listener(id)
	if err != nil {
		codec, _ = systemonecodec.Listener(systemonecodec.SchemaVLLMSR50)
	}
	return codec
}

func recordListenerDiagnostics(listener *routerconfig.Listener, codec systemonecodec.ListenerCodec, read systemonecodec.ClientRequest) {
	name := ""
	if listener != nil {
		name = listener.Name
	}
	for _, diagnostic := range read.Diagnostics {
		listenerDiagnostics.WithLabelValues(name, string(codec.ID()), diagnostic.Field, string(diagnostic.Action)).Inc()
	}
}

// codecWriter holds the answer the handler writes in the internal form, so the
// listener's codec writes it in its client's schema.
type codecWriter struct {
	header  http.Header
	status  int
	body    bytes.Buffer
	request *systemonecodec.ClientRequest
}

func newCodecWriter() *codecWriter { return &codecWriter{header: http.Header{}} }

func (c *codecWriter) Header() http.Header { return c.header }

func (c *codecWriter) Write(p []byte) (int, error) {
	if c.status == 0 {
		c.status = http.StatusOK
	}
	return c.body.Write(p)
}

func (c *codecWriter) WriteHeader(status int) {
	if c.status == 0 {
		c.status = status
	}
}

// flush writes the held answer through codec: a listing, a success or an
// error, each in the client's schema.
func (c *codecWriter) flush(w http.ResponseWriter, r *http.Request, codec systemonecodec.ListenerCodec) {
	status, body := c.status, c.body.Bytes()
	if status == 0 {
		status = http.StatusOK
	}
	switch {
	case status >= 400:
		status, body = codec.AnswerError(c.request, r.URL.Path, status, body)
	case r.Method == http.MethodGet && status == http.StatusOK:
		if listed, err := codec.AnswerListing(body); err == nil {
			body = listed
		}
	case r.Method == http.MethodPost && status >= 200 && status < 300:
		answered, err := codec.AnswerSuccess(c.request, body)
		if err != nil {
			fallback := newCodecWriter()
			publicSystemOneError(fallback, http.StatusBadGateway, "invalid_runtime_response")
			status, body = codec.AnswerError(c.request, r.URL.Path, http.StatusBadGateway, fallback.body.Bytes())
		} else {
			body = answered
		}
	}
	for name, values := range c.header {
		w.Header()[name] = values
	}
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}
