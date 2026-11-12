package routing

import (
	"errors"
	"strings"
	"testing"
)

func header(pairs ...string) Header {
	h := Header{}
	for i := 0; i+1 < len(pairs); i += 2 {
		h = append(h, HeaderField{Name: pairs[i], Value: pairs[i+1]})
	}
	return h
}

func TestHeaderKeepsPseudoHeadersFirst(t *testing.T) {
	h := header(":method", "POST", "content-type", "application/json")
	h.Add(":path", "/v1/chat/completions")
	h.Set("X-Trace", "a")
	h.Add("x-trace", "b")
	want := header(":method", "POST", ":path", "/v1/chat/completions", "content-type", "application/json", "x-trace", "a", "x-trace", "b")
	if got := h; !equalHeaders(got, want) {
		t.Fatalf("header = %v, want %v", got, want)
	}
	if got := h.Values("X-TRACE"); len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("values = %v", got)
	}
	h.Set("x-trace", "c")
	if got := h.Values("x-trace"); len(got) != 1 || got[0] != "c" || h[len(h)-1].Name != "x-trace" {
		t.Fatalf("set must replace every value and move the field last, got %v", h)
	}
	if got := h.WithoutPseudo(); got.Has(":path") || !got.Has("content-type") {
		t.Fatalf("WithoutPseudo = %v", got)
	}
}

func TestApplyHeaderMutationFollowsEnvoyRules(t *testing.T) {
	tests := []struct {
		name     string
		in       Header
		mutation HeaderMutation
		skipCL   bool
		want     Header
	}{
		{
			name:     "removals run before sets",
			in:       header("x-a", "1"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: "x-a", Value: "2"}}, Remove: []string{"x-a"}},
			want:     header("x-a", "2"),
		},
		{
			name:     "set replaces every value and append adds one",
			in:       header("x-a", "1", "x-a", "2", "x-b", "1"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: "X-A", Value: "3"}, {Name: "x-b", Value: "2", Append: true}}},
			want:     header("x-b", "1", "x-a", "3", "x-b", "2"),
		},
		{
			name: "routing headers are never changed",
			in:   header(":authority", "router", ":method", "POST", ":scheme", "http"),
			mutation: HeaderMutation{
				Set:    []HeaderOption{{Name: ":authority", Value: "other"}, {Name: "host", Value: "other"}, {Name: ":method", Value: "GET"}, {Name: ":scheme", Value: "https"}},
				Remove: []string{"host", ":authority"},
			},
			want: header(":authority", "router", ":method", "POST", ":scheme", "http"),
		},
		{
			name:     "path may be set but pseudo-headers are not removed or appended",
			in:       header(":path", "/a", "x-a", "1"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: ":path", Value: "/b"}, {Name: ":path", Value: "/c", Append: true}}, Remove: []string{":path"}},
			want:     header(":path", "/b", "x-a", "1"),
		},
		{
			name:     "x-envoy headers are left alone",
			in:       header("x-envoy-original-path", "/a"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: "x-envoy-decorator", Value: "x"}}, Remove: []string{"x-envoy-original-path"}},
			want:     header("x-envoy-original-path", "/a"),
		},
		{
			name:     "an invalid status is ignored",
			in:       header(":status", "200"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: ":status", Value: "99"}, {Name: ":status", Value: "abc"}}},
			want:     header(":status", "200"),
		},
		{
			name:     "content-length sets are skipped for streamed bodies",
			in:       header("content-length", "10"),
			mutation: HeaderMutation{Set: []HeaderOption{{Name: "Content-Length", Value: "12"}, {Name: "x-a", Value: "1"}}},
			skipCL:   true,
			want:     header("content-length", "10", "x-a", "1"),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got := test.in.Clone()
			if err := ApplyHeaderMutation(&got, &test.mutation, test.skipCL, DefaultLimits); err != nil {
				t.Fatalf("apply: %v", err)
			}
			if !equalHeaders(got, test.want) {
				t.Fatalf("header = %v, want %v", got, test.want)
			}
		})
	}
}

func TestApplyHeaderMutationRejectsInvalidMutations(t *testing.T) {
	tests := map[string]HeaderMutation{
		"bad set name":     {Set: []HeaderOption{{Name: "x a", Value: "1"}}},
		"bad set value":    {Set: []HeaderOption{{Name: "x-a", Value: "1\r\nx-b: 2"}}},
		"bad remove name":  {Remove: []string{"bad:name"}},
		"uppercase pseudo": {Set: []HeaderOption{{Name: ":Path", Value: "/"}}},
	}
	for name, mutation := range tests {
		t.Run(name, func(t *testing.T) {
			h := header("x-a", "1")
			if err := ApplyHeaderMutation(&h, &mutation, false, DefaultLimits); !errors.Is(err, ErrMutation) {
				t.Fatalf("err = %v, want ErrMutation", err)
			}
		})
	}
	h := Header{}
	big := HeaderMutation{Set: []HeaderOption{{Name: "x-big", Value: strings.Repeat("v", 70*1024)}}}
	if err := ApplyHeaderMutation(&h, &big, false, DefaultLimits); !errors.Is(err, ErrMutation) {
		t.Fatalf("oversized headers: err = %v, want ErrMutation", err)
	}
}

func TestCheckContentLength(t *testing.T) {
	if err := CheckContentLength(header("content-length", "3"), &BodyMutation{Body: []byte("abc")}); err != nil {
		t.Fatalf("matching length: %v", err)
	}
	if err := CheckContentLength(header("content-length", "3"), &BodyMutation{Body: []byte("abcd")}); !errors.Is(err, ErrMutation) {
		t.Fatalf("mismatched length: err = %v", err)
	}
	if err := CheckContentLength(header("content-length", "3"), &BodyMutation{Clear: true}); !errors.Is(err, ErrMutation) {
		t.Fatalf("clearing a declared body: err = %v", err)
	}
	if err := CheckContentLength(header(), &BodyMutation{Body: []byte("abcd")}); err != nil {
		t.Fatalf("no declared length: %v", err)
	}
}

func TestRenderImmediateMatchesEnvoyLocalReplies(t *testing.T) {
	got := RenderImmediate(&ImmediateResponse{
		Status: 400,
		Header: &HeaderMutation{Set: []HeaderOption{{Name: "content-type", Value: "application/json"}, {Name: "x-vsr-response-path", Value: "error"}}},
		Body:   []byte(`{"error":{}}`),
	}, DefaultLimits)
	want := header("content-type", "application/json", "x-vsr-response-path", "error", "content-length", "12")
	if got.Status != 400 || !equalHeaders(got.Header, want) || string(got.Body) != `{"error":{}}` {
		t.Fatalf("got %d %v %q", got.Status, got.Header, got.Body)
	}

	plain := RenderImmediate(&ImmediateResponse{Body: []byte("denied")}, DefaultLimits)
	if plain.Status != 200 || plain.Header.Get("content-type") != "text/plain" || plain.Header.Get("content-length") != "6" {
		t.Fatalf("a status below 200 becomes 200 and a body gets text/plain: %d %v", plain.Status, plain.Header)
	}

	empty := RenderImmediate(&ImmediateResponse{
		Status: 404,
		Header: &HeaderMutation{Set: []HeaderOption{{Name: "content-type", Value: "application/json"}, {Name: ":status", Value: "410"}}},
	}, DefaultLimits)
	if empty.Status != 410 || empty.Header.Has("content-type") || empty.Header.Has("content-length") {
		t.Fatalf("an empty reply drops body headers and honors :status: %d %v", empty.Status, empty.Header)
	}
}

func equalHeaders(a, b Header) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
