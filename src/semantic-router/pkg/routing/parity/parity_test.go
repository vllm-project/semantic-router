package parity

import (
	"context"
	"encoding/json"
	"io"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestNormalizeBodyRewritesOnlyClockNumbers(t *testing.T) {
	tests := map[string]struct{ in, want string }{
		"json keeps every other byte": {
			in:   `{"id":"x", "created" : 1700000000,"data":[{"created":12,"name":"created"}],"text":"\"created\":5"}`,
			want: `{"id":"x", "created" : 0,"data":[{"created":0,"name":"created"}],"text":"\"created\":5"}`,
		},
		"sse events": {
			in:   "data: {\"created\":7,\"b\":1}\n\ndata: [DONE]\n\n",
			want: "data: {\"created\":0,\"b\":1}\n\ndata: [DONE]\n\n",
		},
		"string clock values stay": {in: `{"created_at":"today"}`, want: `{"created_at":"today"}`},
		"not json":                 {in: "plain text created 5", want: "plain text created 5"},
		"two documents":            {in: `{"created":1}{"created":2}`, want: `{"created":1}{"created":2}`},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if got := string(NormalizeBody([]byte(test.in))); got != test.want {
				t.Fatalf("NormalizeBody(%s) = %s, want %s", test.in, got, test.want)
			}
		})
	}
}

func TestNormalizeMutationSortsRemovalsAndHidesVolatileValues(t *testing.T) {
	in := &routing.HeaderMutation{
		Set:    []routing.HeaderOption{{Name: "x-vsr-routing-latency-ms", Value: "1.2"}, {Name: "x-a", Value: "1"}},
		Remove: []string{"b", "a"},
	}
	out := NormalizeMutation(in)
	if out.Set[0].Value != Placeholder || out.Set[1].Value != "1" || out.Remove[0] != "a" {
		t.Fatalf("normalized = %+v", out)
	}
	if in.Remove[0] != "b" || in.Set[0].Value != "1.2" {
		t.Fatal("NormalizeMutation must not modify its input")
	}
	if !VolatileHeader("TraceParent") || VolatileHeader("x-vsr-selected-model") {
		t.Fatal("volatile header classification is wrong")
	}
}

func TestTextEncodesUTF8AsAStringAndBinaryAsBase64(t *testing.T) {
	for _, value := range []Text{Text("<a & b>"), Text([]byte{0xff, 0x00})} {
		encoded, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		var decoded Text
		if err := json.Unmarshal(encoded, &decoded); err != nil {
			t.Fatal(err)
		}
		if string(decoded) != string(value) {
			t.Fatalf("round trip of %q via %s gave %q", value, encoded, decoded)
		}
	}
	record := &Record{Case: "<a & b>", Response: &Message{Body: Text("<a & b>")}}
	if encoded, _ := record.Encode(); !strings.Contains(string(encoded), `"body": "<a & b>"`) {
		t.Fatalf("records must not HTML-escape text: %s", encoded)
	}
}

func TestDiffText(t *testing.T) {
	if DiffText([]byte("a\nb\n"), []byte("a\nb\n")) != "" {
		t.Fatal("equal texts must have no diff")
	}
	diff := DiffText([]byte("a\nb\nc"), []byte("a\nx\nc"))
	if !strings.Contains(diff, "-2: b") || !strings.Contains(diff, "+2: x") {
		t.Fatalf("diff = %q", diff)
	}
}

func TestLoadCorpusResolvesUpstreams(t *testing.T) {
	corpus, err := LoadCorpus("testdata/corpus")
	if err != nil {
		t.Fatal(err)
	}
	if len(corpus.Config) == 0 || len(corpus.Cases) == 0 {
		t.Fatal("empty corpus")
	}
	seen := map[string]bool{}
	for _, c := range corpus.Cases {
		if seen[c.Name] {
			t.Fatalf("duplicate case %q", c.Name)
		}
		seen[c.Name] = true
		if (c.UpstreamName == "") != (c.Upstream == nil) {
			t.Fatalf("case %q: upstream %q not resolved", c.Name, c.UpstreamName)
		}
		request := c.GatewayRequest()
		if request.Header.Get(":method") != c.Request.Method || request.Header.Get("x-request-id") != "parity-"+c.Name {
			t.Fatalf("case %q: gateway headers %v", c.Name, request.Header)
		}
	}
}

func TestUpstreamResponseReadsOnePartPerRead(t *testing.T) {
	up := &Upstream{Status: 200, Headers: [][]string{{"Content-Type", "text/event-stream"}}, Chunks: []string{"one", "two"}}
	resp := up.Response()
	if resp.Header.Get("content-type") != "text/event-stream" || resp.Header.Has("content-length") {
		t.Fatalf("headers = %v", resp.Header)
	}
	buf := make([]byte, 64)
	var parts []string
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			parts = append(parts, string(buf[:n]))
		}
		if err == io.EOF {
			break
		}
	}
	if strings.Join(parts, "|") != "one|two" {
		t.Fatalf("parts = %q", parts)
	}
	buffered := (&Upstream{Status: 200, Body: "abc"}).Response()
	if buffered.Header.Get("content-length") != "3" {
		t.Fatalf("a buffered fixture declares its length: %v", buffered.Header)
	}
	if (&Upstream{Status: 204}).Response().Body != nil {
		t.Fatal("a fixture without a body has no body reader")
	}
}

// echoProcessor routes every request to "model-a" and tags the response.
type echoProcessor struct{}

func (echoProcessor) Open(context.Context) (routing.Session, error) { return &echoSession{}, nil }

type echoSession struct{ closed int }

func (s *echoSession) RequestHeaders(routing.Header, bool) (*routing.Effect, error) {
	return &routing.Effect{}, nil
}

func (s *echoSession) RequestBody(body []byte, _ bool) (*routing.Effect, error) {
	return &routing.Effect{
		Header:          &routing.HeaderMutation{Set: []routing.HeaderOption{{Name: routing.RouteHeader, Value: "model-a"}}},
		ClearRouteCache: true,
	}, nil
}

func (s *echoSession) ResponseHeaders(routing.Header, bool) (*routing.Effect, error) {
	return &routing.Effect{Header: &routing.HeaderMutation{Set: []routing.HeaderOption{{Name: "x-vsr-routing-latency-ms", Value: "3.1"}}}}, nil
}

func (s *echoSession) ResponseBody(body []byte, _ bool) (*routing.Effect, error) {
	return &routing.Effect{}, nil
}

func (s *echoSession) Evidence() routing.Evidence { return routing.Evidence{Model: "model-a"} }

func (s *echoSession) Close(error) { s.closed++ }

func TestRecorderRecordsPhasesCallAndResponse(t *testing.T) {
	c := Case{
		Name:     "echo",
		Request:  CaseRequest{Method: "POST", Path: "/v1/chat/completions", Body: `{"model":"auto"}`},
		Upstream: &Upstream{Status: 200, Body: `{"created":123,"ok":true}`},
	}
	record := NewRecorder(echoProcessor{}, routing.DefaultOptions).Run(context.Background(), c)
	if record.Error != "" {
		t.Fatal(record.Error)
	}
	if record.Route != "model-a" || record.Upstream == nil || record.Evidence.Model != "model-a" {
		t.Fatalf("record = %+v", record)
	}
	if len(record.Phases) != 4 || record.Phases[3].Phase != routing.PhaseResponseBody {
		t.Fatalf("phases = %+v", record.Phases)
	}
	if string(record.Response.Body) != `{"created":0,"ok":true}` || record.Response.Header.Get("x-vsr-routing-latency-ms") != Placeholder {
		t.Fatalf("response not normalized: %+v", record.Response)
	}
	missing := NewRecorder(echoProcessor{}, routing.DefaultOptions).Run(context.Background(), Case{Name: "no-fixture", Request: c.Request})
	if missing.Error == "" {
		t.Fatal("a planned call without an upstream fixture must be reported")
	}
}
