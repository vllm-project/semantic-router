package parity

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Wire records compare two running gateways (for example native and Envoy
// modes) on the same corpus: a Backend stands in for every model backend and
// records what reaches it, and Send plays the client. Go's HTTP stack does not
// keep header order, so wire messages list headers sorted by name.

// RequestIDPrefix marks corpus requests; the Backend finds a case by its
// x-request-id, which gateways forward unchanged.
const RequestIDPrefix = "parity-"

// Backend is a fake model backend for a corpus. It answers each case's request
// with the case's upstream fixture and records the request it received.
type Backend struct {
	cases map[string]Case
	mu    sync.Mutex
	seen  map[string][]*Message
}

// NewBackend serves the upstream fixtures of cases.
func NewBackend(cases []Case) *Backend {
	b := &Backend{cases: make(map[string]Case, len(cases)), seen: map[string][]*Message{}}
	for _, c := range cases {
		b.cases[RequestIDPrefix+c.Name] = c
	}
	return b
}

// ServeHTTP answers with the fixture of the case named by x-request-id, or 404.
func (b *Backend) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	id := r.Header.Get("x-request-id")
	body, _ := io.ReadAll(r.Body)
	message := &Message{Header: wireRequestHeader(r), Body: body}
	b.mu.Lock()
	b.seen[id] = append(b.seen[id], message)
	b.mu.Unlock()

	c, ok := b.cases[id]
	if !ok || c.Upstream == nil {
		http.Error(w, "no fixture for "+id, http.StatusNotFound)
		return
	}
	for _, field := range c.Upstream.Header() {
		w.Header().Add(field.Name, field.Value)
	}
	w.WriteHeader(c.Upstream.Status)
	flusher, _ := w.(http.Flusher)
	for _, part := range c.Upstream.Parts() {
		_, _ = io.WriteString(w, part)
		if flusher != nil {
			flusher.Flush()
		}
	}
}

// Received returns the requests the backend received for a case, in order.
func (b *Backend) Received(caseName string) []*Message {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]*Message(nil), b.seen[RequestIDPrefix+caseName]...)
}

// Reset forgets every received request.
func (b *Backend) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.seen = map[string][]*Message{}
}

func wireRequestHeader(r *http.Request) routing.Header {
	header := routing.Header{
		{Name: ":method", Value: r.Method},
		{Name: ":path", Value: r.URL.RequestURI()},
		{Name: ":authority", Value: r.Host},
	}
	return append(header, sortedHeader(r.Header)...)
}

func sortedHeader(h http.Header) routing.Header {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	var out routing.Header
	for _, name := range names {
		for _, value := range h[name] {
			out = append(out, routing.HeaderField{Name: strings.ToLower(name), Value: value})
		}
	}
	return out
}

// Send sends c to the gateway at baseURL as a client would, with the case's
// request ID, and returns the response the client received. A streamed body
// is read to its end and kept whole, since the wire does not preserve chunks.
func Send(ctx context.Context, client *http.Client, baseURL string, c Case) (*Message, error) {
	var body io.Reader
	if c.Request.Body != "" {
		body = strings.NewReader(c.Request.Body)
	}
	req, err := http.NewRequestWithContext(ctx, c.Request.Method, strings.TrimSuffix(baseURL, "/")+c.Request.Path, body)
	if err != nil {
		return nil, err
	}
	for _, pair := range c.Request.Headers {
		req.Header.Add(pair[0], pair[1])
	}
	if req.Header.Get("User-Agent") == "" {
		// An empty value stops Go's client from sending its own agent, which
		// a gateway would forward like any client header.
		req.Header.Set("User-Agent", "")
	}
	req.Header.Set("x-request-id", RequestIDPrefix+c.Name)
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response for %s: %w", c.Name, err)
	}
	return &Message{Status: resp.StatusCode, Header: sortedHeader(resp.Header), Body: data}, nil
}

// RecordWire sends every case through the gateway at baseURL and returns the
// normalized wire records: the client's response and what reached backend.
func RecordWire(ctx context.Context, client *http.Client, baseURL string, backend *Backend, cases []Case) []*Record {
	records := make([]*Record, 0, len(cases))
	for _, c := range cases {
		record := &Record{Case: c.Name}
		response, err := Send(ctx, client, baseURL, c)
		if err != nil {
			record.Error = err.Error()
		}
		record.Response = response
		if received := backend.Received(c.Name); len(received) > 0 {
			record.Upstream = received[len(received)-1]
			record.Route = record.Upstream.Header.Get(routing.RouteHeader)
		}
		records = append(records, record.Normalize())
	}
	return records
}

// WithoutHeaders returns a copy of r without the named headers in its wire
// messages, for differences between gateways that a design document names.
func (r *Record) WithoutHeaders(names ...string) *Record {
	out := *r
	strip := func(m *Message) *Message {
		if m == nil {
			return nil
		}
		copied := *m
		copied.Header = m.Header.Clone()
		for _, name := range names {
			copied.Header.Del(name)
		}
		return &copied
	}
	out.Upstream = strip(r.Upstream)
	out.Response = strip(r.Response)
	return &out
}

// ConfigWithBackend returns the corpus configuration with every provider
// backend pointed at the fake backend's base URL.
func (c *Corpus) ConfigWithBackend(baseURL string) ([]byte, error) {
	target, err := url.Parse(baseURL)
	if err != nil || target.Host == "" {
		return nil, fmt.Errorf("invalid backend URL %q", baseURL)
	}
	var document yaml.Node
	if err := yaml.Unmarshal(c.Config, &document); err != nil {
		return nil, err
	}
	for _, model := range mappingSequence(mappingValue(documentRoot(&document), "providers"), "models") {
		for _, ref := range mappingSequence(model, "backend_refs") {
			if baseURLNode := mappingValue(ref, "base_url"); baseURLNode != nil {
				original, err := url.Parse(baseURLNode.Value)
				if err != nil {
					return nil, err
				}
				baseURLNode.Value = target.Scheme + "://" + target.Host + original.Path
			}
			if endpoint := mappingValue(ref, "endpoint"); endpoint != nil {
				endpoint.Value = target.Host
			}
			if protocol := mappingValue(ref, "protocol"); protocol != nil {
				protocol.Value = target.Scheme
			}
		}
	}
	var out bytes.Buffer
	encoder := yaml.NewEncoder(&out)
	encoder.SetIndent(2)
	if err := encoder.Encode(&document); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}

func documentRoot(document *yaml.Node) *yaml.Node {
	if document.Kind == yaml.DocumentNode && len(document.Content) == 1 {
		return document.Content[0]
	}
	return document
}

func mappingValue(node *yaml.Node, key string) *yaml.Node {
	if node == nil || node.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(node.Content); i += 2 {
		if node.Content[i].Value == key {
			return node.Content[i+1]
		}
	}
	return nil
}

func mappingSequence(node *yaml.Node, key string) []*yaml.Node {
	sequence := mappingValue(node, key)
	if sequence == nil || sequence.Kind != yaml.SequenceNode {
		return nil
	}
	return sequence.Content
}
