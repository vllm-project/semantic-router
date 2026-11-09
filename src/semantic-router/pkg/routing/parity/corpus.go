// Package parity records how a gateway handles a request set, so the native
// gateway and the Envoy modes can be compared on identical requests.
//
// A Record holds each phase's effect, the upstream request after every
// mutation, its route, the client response and the routing evidence. Records
// come from the routing engine (Recorder), from the ext_proc adapter, or from
// the wire, and compare after Normalize, which rewrites volatile values only.
package parity

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// Corpus is a request set with the router configuration it runs against.
type Corpus struct {
	// Config is the canonical router configuration document.
	Config []byte
	Cases  []Case
}

// Case is one client request and the upstream response a fake backend gives.
type Case struct {
	Name    string      `yaml:"name"`
	Request CaseRequest `yaml:"request"`
	// UpstreamName names a fixture in the corpus's upstreams.
	UpstreamName string `yaml:"upstream"`
	// Upstream is the resolved fixture; nil when the case expects no call.
	Upstream *Upstream `yaml:"-"`
}

// CaseRequest is a client request; headers are ordered name/value pairs.
type CaseRequest struct {
	Method  string     `yaml:"method"`
	Path    string     `yaml:"path"`
	Headers [][]string `yaml:"headers"`
	Body    string     `yaml:"body"`
}

// Upstream is a canned backend response. Chunks, when set, arrive one per
// read, as a streaming backend sends them. Encoding "gzip" sends Body
// gzip-encoded under Content-Encoding: gzip.
type Upstream struct {
	Status   int        `yaml:"status"`
	Headers  [][]string `yaml:"headers"`
	Body     string     `yaml:"body"`
	Chunks   []string   `yaml:"chunks"`
	Encoding string     `yaml:"encoding"`

	// wire is a gzip Body as sent, encoded once by LoadCorpus.
	wire string
}

type corpusFile struct {
	Cases     []Case               `yaml:"cases"`
	Upstreams map[string]*Upstream `yaml:"upstreams"`
}

// LoadCorpus reads config.yaml and cases.yaml from dir.
func LoadCorpus(dir string) (*Corpus, error) {
	config, err := os.ReadFile(filepath.Join(dir, "config.yaml"))
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(filepath.Join(dir, "cases.yaml"))
	if err != nil {
		return nil, err
	}
	var file corpusFile
	if err := yaml.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("parse %s: %w", filepath.Join(dir, "cases.yaml"), err)
	}
	for name, upstream := range file.Upstreams {
		if upstream.Encoding == "" {
			continue
		}
		if upstream.Encoding != "gzip" || upstream.Body == "" || len(upstream.Chunks) > 0 {
			return nil, fmt.Errorf("upstream %q: encoding %q needs a gzip body without chunks", name, upstream.Encoding)
		}
		for _, pair := range upstream.Headers {
			if len(pair) > 0 && strings.EqualFold(pair[0], "content-encoding") {
				return nil, fmt.Errorf("upstream %q: encoding sets Content-Encoding; drop it from headers", name)
			}
		}
		upstream.wire = storedGzip(upstream.Body)
	}
	for i := range file.Cases {
		c := &file.Cases[i]
		for _, pair := range c.Request.Headers {
			if len(pair) != 2 {
				return nil, fmt.Errorf("case %q: header %v is not a name/value pair", c.Name, pair)
			}
		}
		if c.UpstreamName == "" {
			continue
		}
		upstream, ok := file.Upstreams[c.UpstreamName]
		if !ok {
			return nil, fmt.Errorf("case %q names unknown upstream %q", c.Name, c.UpstreamName)
		}
		c.Upstream = upstream
	}
	return &Corpus{Config: config, Cases: file.Cases}, nil
}

// GatewayRequest returns the request as Envoy's HTTP connection manager hands
// it to ext_proc: pseudo-headers, the client's headers without the
// proxy-control headers Envoy strips from clients, the body length, and the
// forwarded protocol and request ID Envoy adds.
func (c Case) GatewayRequest() *routing.Request {
	header := routing.Header{
		{Name: ":method", Value: c.Request.Method},
		{Name: ":path", Value: c.Request.Path},
		{Name: ":authority", Value: "localhost:8899"},
		{Name: ":scheme", Value: "http"},
	}
	for _, pair := range c.Request.Headers {
		if routing.IsProxyControlHeader(pair[0]) {
			continue
		}
		header = append(header, routing.HeaderField{Name: strings.ToLower(pair[0]), Value: pair[1]})
	}
	if c.Request.Body != "" {
		header = append(header, routing.HeaderField{Name: "content-length", Value: strconv.Itoa(len(c.Request.Body))})
	}
	header = append(header,
		routing.HeaderField{Name: "x-forwarded-proto", Value: "http"},
		routing.HeaderField{Name: "x-request-id", Value: "parity-" + c.Name},
	)
	return &routing.Request{Header: header, Body: []byte(c.Request.Body)}
}

// Header returns the fixture's response headers, with the content coding and
// the content length of a non-chunked body.
func (u *Upstream) Header() routing.Header {
	header := make(routing.Header, 0, len(u.Headers)+2)
	for _, pair := range u.Headers {
		header = append(header, routing.HeaderField{Name: strings.ToLower(pair[0]), Value: pair[1]})
	}
	if u.Encoding != "" {
		header = append(header, routing.HeaderField{Name: "content-encoding", Value: u.Encoding})
	}
	if u.Body != "" {
		header = append(header, routing.HeaderField{Name: "content-length", Value: strconv.Itoa(len(u.wireBody()))})
	}
	return header
}

// HasBody reports whether the fixture has a body.
func (u *Upstream) HasBody() bool {
	return u.Body != "" || len(u.Chunks) > 0
}

// Parts returns the body as the backend sends it: the chunks, or the body as
// one part.
func (u *Upstream) Parts() []string {
	if len(u.Chunks) > 0 {
		return u.Chunks
	}
	if u.Body == "" {
		return nil
	}
	return []string{u.wireBody()}
}

// wireBody returns Body as the backend sends it: a corpus fixture's gzip body
// as LoadCorpus encoded it, or encoded here for a fixture built in code.
func (u *Upstream) wireBody() string {
	switch {
	case u.Encoding != "gzip":
		return u.Body
	case u.wire != "":
		return u.wire
	}
	return storedGzip(u.Body)
}

// storedGzip stores body, not compressed, inside the gzip framing, so recorded
// bytes do not depend on the Go release's deflate output.
func storedGzip(body string) string {
	var buf bytes.Buffer
	writer, _ := gzip.NewWriterLevel(&buf, gzip.NoCompression)
	_, _ = writer.Write([]byte(body))
	_ = writer.Close()
	return buf.String()
}

// Response returns the fixture as an upstream response whose body reads one
// part per Read call.
func (u *Upstream) Response() *routing.UpstreamResponse {
	resp := &routing.UpstreamResponse{Status: u.Status, Header: u.Header()}
	if u.HasBody() {
		resp.Body = &partReader{parts: u.Parts()}
	}
	return resp
}

type partReader struct {
	parts   []string
	current *bytes.Reader
}

func (r *partReader) Read(p []byte) (int, error) {
	for r.current == nil || r.current.Len() == 0 {
		if len(r.parts) == 0 {
			return 0, io.EOF
		}
		r.current = bytes.NewReader([]byte(r.parts[0]))
		r.parts = r.parts[1:]
	}
	return r.current.Read(p)
}
