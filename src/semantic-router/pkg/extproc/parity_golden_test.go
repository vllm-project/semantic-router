package extproc

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"flag"
	"io"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"unicode/utf8"

	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protojson"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/parity"
)

var updateParityGolden = flag.Bool("update-parity-golden", false, "rewrite the parity golden files")

const (
	parityCorpusDir         = "../routing/parity/testdata/corpus"
	parityGoldenDir         = "testdata/parity/extproc"
	parityRecordsGoldenDir  = "testdata/parity/records"
	parityProtocolBodyModes = http_ext.ProcessingMode_BUFFERED
)

func loadParityCorpus(t *testing.T) (*parity.Corpus, *config.RouterConfig) {
	t.Helper()
	corpus, err := parity.LoadCorpus(parityCorpusDir)
	if err != nil {
		t.Fatalf("load parity corpus: %v", err)
	}
	cfg, err := config.ParseYAMLBytes(corpus.Config)
	if err != nil {
		t.Fatalf("parse parity corpus config: %v", err)
	}
	return corpus, cfg
}

// newParityRouter builds the corpus router serving its startup snapshot, as a
// Router's first generation serves configuration version 1.
func newParityRouter(t *testing.T, cfg *config.RouterConfig) *OpenAIRouter {
	t.Helper()
	router, err := buildOpenAIRouterFromConfig(cfg)
	if err != nil {
		t.Fatalf("build parity router: %v", err)
	}
	t.Cleanup(func() { _ = router.Close() })
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	if err != nil {
		t.Fatalf("install parity snapshot: %v", err)
	}
	newRouterGeneration(router, snapshot)
	return router
}

// extprocStreamProcessor opens routing sessions that drive a live ext_proc
// Process loop the way Envoy does, so a routing engine can run against the
// gRPC adapter itself. It keeps every message the Router sent.
type extprocStreamProcessor struct {
	router *OpenAIRouter
	mu     sync.Mutex
	sent   []*ext_proc.ProcessingResponse
}

func (p *extprocStreamProcessor) takeSent() []*ext_proc.ProcessingResponse {
	p.mu.Lock()
	defer p.mu.Unlock()
	sent := p.sent
	p.sent = nil
	return sent
}

func (p *extprocStreamProcessor) Open(ctx context.Context) (routing.Session, error) {
	stream := &channelProcessStream{
		ctx:      ctx,
		requests: make(chan *ext_proc.ProcessingRequest),
		replies:  make(chan *ext_proc.ProcessingResponse, 1),
		onSend: func(response *ext_proc.ProcessingResponse) {
			p.mu.Lock()
			p.sent = append(p.sent, response)
			p.mu.Unlock()
		},
	}
	session := &extprocStreamSession{stream: stream, done: make(chan error, 1)}
	go func() { session.done <- p.router.Process(stream) }()
	return session, nil
}

type channelProcessStream struct {
	ctx      context.Context
	requests chan *ext_proc.ProcessingRequest
	replies  chan *ext_proc.ProcessingResponse
	onSend   func(*ext_proc.ProcessingResponse)
	endErr   error
}

func (s *channelProcessStream) Recv() (*ext_proc.ProcessingRequest, error) {
	request, ok := <-s.requests
	if !ok {
		return nil, s.endErr
	}
	return request, nil
}

func (s *channelProcessStream) Send(response *ext_proc.ProcessingResponse) error {
	s.onSend(response)
	s.replies <- response
	return nil
}

func (s *channelProcessStream) Context() context.Context     { return s.ctx }
func (s *channelProcessStream) SendMsg(interface{}) error    { return nil }
func (s *channelProcessStream) RecvMsg(interface{}) error    { return nil }
func (s *channelProcessStream) SetHeader(metadata.MD) error  { return nil }
func (s *channelProcessStream) SendHeader(metadata.MD) error { return nil }
func (s *channelProcessStream) SetTrailer(metadata.MD)       {}

var _ ext_proc.ExternalProcessor_ProcessServer = (*channelProcessStream)(nil)

type extprocStreamSession struct {
	stream  *channelProcessStream
	done    chan error
	started bool
	ended   bool
}

func (s *extprocStreamSession) exchange(request *ext_proc.ProcessingRequest) (*routing.Effect, error) {
	if !s.started {
		// Envoy announces the configured body modes on the first message.
		request.ProtocolConfig = &ext_proc.ProtocolConfiguration{
			RequestBodyMode:  parityProtocolBodyModes,
			ResponseBodyMode: parityProtocolBodyModes,
		}
		s.started = true
	}
	s.stream.requests <- request
	select {
	case response := <-s.stream.replies:
		return routingEffect(response)
	case err := <-s.done:
		s.ended = true
		if err == nil {
			err = errors.New("ext_proc stream ended without a reply")
		}
		return nil, err
	}
}

func (s *extprocStreamSession) RequestHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	return s.exchange(&ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: extprocHeaderMap(header), EndOfStream: endOfStream},
	}})
}

func (s *extprocStreamSession) RequestBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	return s.exchange(&ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: endOfStream},
	}})
}

func (s *extprocStreamSession) ResponseHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	return s.exchange(&ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: extprocHeaderMap(header), EndOfStream: endOfStream},
	}})
}

func (s *extprocStreamSession) ResponseBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	return s.exchange(&ext_proc.ProcessingRequest{Request: &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: endOfStream},
	}})
}

// Evidence lives inside the Process loop; this side of the stream cannot see it.
func (s *extprocStreamSession) Evidence() routing.Evidence { return routing.Evidence{} }

// Close ends the stream as Envoy does: EOF after a delivered response, a
// cancelled stream otherwise.
func (s *extprocStreamSession) Close(err error) {
	if s.ended {
		return
	}
	s.ended = true
	s.stream.endErr = io.EOF
	if err != nil {
		s.stream.endErr = status.Error(codes.Canceled, "stream canceled")
	}
	close(s.stream.requests)
	<-s.done
}

// canonicalExtProcResponses renders sent ext_proc messages as stable JSON.
// protojson output is deliberately unstable, so it is re-encoded with sorted
// keys; byte fields become text, and only the parity normalizer's volatile
// values and removal order are rewritten.
func canonicalExtProcResponses(t *testing.T, responses []*ext_proc.ProcessingResponse) []byte {
	t.Helper()
	items := make([]interface{}, 0, len(responses))
	for _, response := range responses {
		raw, err := protojson.Marshal(response)
		if err != nil {
			t.Fatalf("marshal ext_proc response: %v", err)
		}
		var item interface{}
		if err := json.Unmarshal(raw, &item); err != nil {
			t.Fatalf("decode ext_proc response json: %v", err)
		}
		items = append(items, canonicalizeExtProcJSON(item))
	}
	var out bytes.Buffer
	encoder := json.NewEncoder(&out)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	if err := encoder.Encode(items); err != nil {
		t.Fatalf("encode golden: %v", err)
	}
	return out.Bytes()
}

func canonicalizeExtProcJSON(value interface{}) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		if header, ok := typed["header"].(map[string]interface{}); ok {
			if key, _ := header["key"].(string); parity.VolatileHeader(key) {
				for _, field := range []string{"rawValue", "value"} {
					if _, present := header[field]; present {
						header[field] = base64.StdEncoding.EncodeToString([]byte(parity.Placeholder))
					}
				}
			}
		}
		out := make(map[string]interface{}, len(typed))
		for key, field := range typed {
			text, isBytes := decodedBytesField(key, field)
			switch {
			case isBytes && key == "body":
				out["bodyText"] = string(parity.NormalizeBody([]byte(text)))
			case isBytes:
				out[key+"Text"] = text
			case key == "removeHeaders":
				out[key] = sortedStrings(field)
			default:
				out[key] = canonicalizeExtProcJSON(field)
			}
		}
		return out
	case []interface{}:
		out := make([]interface{}, len(typed))
		for i, item := range typed {
			out[i] = canonicalizeExtProcJSON(item)
		}
		return out
	default:
		return value
	}
}

// decodedBytesField decodes the protojson base64 of a bytes field when it is text.
func decodedBytesField(key string, field interface{}) (string, bool) {
	if key != "rawValue" && key != "body" {
		return "", false
	}
	encoded, ok := field.(string)
	if !ok {
		return "", false
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || !utf8.Valid(decoded) {
		return "", false
	}
	return string(decoded), true
}

func sortedStrings(field interface{}) interface{} {
	items, ok := field.([]interface{})
	if !ok {
		return field
	}
	out := make([]string, 0, len(items))
	for _, item := range items {
		text, _ := item.(string)
		out = append(out, text)
	}
	sort.Strings(out)
	return out
}

func checkParityGolden(t *testing.T, dir, name string, got []byte) {
	t.Helper()
	path := filepath.Join(dir, name+".json")
	if *updateParityGolden {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatal(err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read golden (run with -update-parity-golden to create it): %v", err)
	}
	if diff := parity.DiffText(want, got); diff != "" {
		t.Fatalf("%s changed:\n%s", path, diff)
	}
}

// TestExtProcParityGolden pins the exact ext_proc messages the Router sends for
// the parity corpus, so refactors of the routing core keep the ext_proc
// adapter byte for byte identical. The goldens were first recorded with
// pkg/extproc identical to main.
func TestExtProcParityGolden(t *testing.T) {
	corpus, cfg := loadParityCorpus(t)
	processor := &extprocStreamProcessor{router: newParityRouter(t, cfg)}
	recorder := parity.NewRecorder(processor, routing.DefaultOptions)
	for _, c := range corpus.Cases {
		t.Run(c.Name, func(t *testing.T) {
			if record := recorder.Run(context.Background(), c); record.Error != "" {
				t.Fatalf("run: %s", record.Error)
			}
			checkParityGolden(t, parityGoldenDir, c.Name, canonicalExtProcResponses(t, processor.takeSent()))
		})
	}
}
