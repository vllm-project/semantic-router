package routing

import "testing"

func TestHeaderGetAndHasAreCaseInsensitive(t *testing.T) {
	var h Header
	h.Add("Content-Type", "application/json")
	h.Add("X-Empty", "")
	h.Add("X-Trace", "first")
	h.Add("x-trace", "second")

	if h.Get("CONTENT-TYPE") != "application/json" || h.Get("X-Trace") != "first" {
		t.Fatalf("Get = content-type %q, x-trace %q", h.Get("CONTENT-TYPE"), h.Get("X-Trace"))
	}
	if h.Get("missing") != "" || h.Has("missing") {
		t.Fatalf("absent header: Get=%q Has=%v", h.Get("missing"), h.Has("missing"))
	}
	if !h.Has("content-type") || !h.Has("X-EMPTY") || h.Get("x-empty") != "" {
		t.Fatalf("empty value is present: %v", h)
	}

	var absent Header
	if absent.Get("host") != "" || absent.Has("host") {
		t.Fatal("a nil header is absent")
	}
}

func TestHeaderValuesKeepDuplicatesInOrder(t *testing.T) {
	var h Header
	h.Add("X-A", "1")
	h.Add("X-B", "only")
	h.Add("x-a", "2")
	h.Add("X-A", "3")

	got := h.Values("x-A")
	if len(got) != 3 || got[0] != "1" || got[1] != "2" || got[2] != "3" {
		t.Fatalf("Values = %v, want [1 2 3]", got)
	}
	if only := h.Values("X-B"); len(only) != 1 || only[0] != "only" {
		t.Fatalf("Values = %v, want [only]", only)
	}
	if missing := h.Values("missing"); len(missing) != 0 {
		t.Fatalf("Values = %v, want none", missing)
	}
}

func TestHeaderAddPlacesPseudoHeadersAtTheEndOfTheBlock(t *testing.T) {
	h := header(":method", "POST", ":path", "/v1", "content-type", "application/json")
	h.Add(":Authority", "router.local")
	h.Add("X-Request-Id", "req-1")
	h.Add(":path", "/v1/chat/completions")
	want := header(
		":method", "POST",
		":path", "/v1",
		":authority", "router.local",
		":path", "/v1/chat/completions",
		"content-type", "application/json",
		"x-request-id", "req-1",
	)
	if !equalHeaders(h, want) {
		t.Fatalf("header = %v, want %v", h, want)
	}

	regular := header("content-type", "application/json")
	regular.Add(":method", "GET")
	if !equalHeaders(regular, header(":method", "GET", "content-type", "application/json")) {
		t.Fatalf("header = %v, want the pseudo-header ahead of the regular header", regular)
	}
}

func TestHeaderSetReplacesEveryValueAtTheTail(t *testing.T) {
	h := header("x-a", "1", "x-b", "keep", "x-a", "2", "x-c", "tail")
	h.Set("X-A", "3")
	want := header("x-b", "keep", "x-c", "tail", "x-a", "3")
	if !equalHeaders(h, want) {
		t.Fatalf("header = %v, want %v", h, want)
	}

	// setCopy puts the replacement where Add would. A pseudo-header therefore
	// lands at the end of the pseudo-header block, not after regular headers.
	h = header(":method", "POST", ":path", "/old", ":path", "/also", "x-host", "router")
	h.Set(":PATH", "/new")
	want = header(":method", "POST", ":path", "/new", "x-host", "router")
	if !equalHeaders(h, want) {
		t.Fatalf("pseudo-header set = %v, want %v", h, want)
	}
}

func TestHeaderDelRemovesEveryValue(t *testing.T) {
	h := header(":method", "POST", "x-a", "1", "x-ab", "keep", "x-a", "2", "x-c", "stay")
	h.Del("X-A")
	want := header(":method", "POST", "x-ab", "keep", "x-c", "stay")
	if !equalHeaders(h, want) {
		t.Fatalf("header = %v, want %v", h, want)
	}
	h.Del(":METHOD")
	want = header("x-ab", "keep", "x-c", "stay")
	if !equalHeaders(h, want) {
		t.Fatalf("header = %v, want %v", h, want)
	}
	h.Del("missing")
	if !equalHeaders(h, want) {
		t.Fatalf("deleting an absent name changed the header: %v", h)
	}
}

func TestHeaderWithoutPseudoAndCloneStayIndependent(t *testing.T) {
	original := header(":method", "POST", ":path", "/v1", "x-a", "1", "x-b", "2", "x-a", "3")
	kept := original.Clone()

	regular := original.WithoutPseudo()
	if !equalHeaders(regular, header("x-a", "1", "x-b", "2", "x-a", "3")) {
		t.Fatalf("WithoutPseudo = %v", regular)
	}
	regular.Set("x-a", "changed")
	regular.Del("x-b")
	if !equalHeaders(original, kept) {
		t.Fatalf("WithoutPseudo shares storage with the header: %v", original)
	}
	if onlyPseudo := header(":status", "200").WithoutPseudo(); len(onlyPseudo) != 0 {
		t.Fatalf("WithoutPseudo = %v, want no regular headers", onlyPseudo)
	}

	cloned := original.Clone()
	cloned[0].Value = "GET"
	cloned.Set("x-a", "from-clone")
	cloned.Del(":path")
	if !equalHeaders(original, kept) {
		t.Fatalf("clone mutation changed the original: %v", original)
	}
	original.Add("x-new", "1")
	original.Del("x-b")
	if cloned.Has("x-new") || !cloned.Has("x-b") || cloned.Get(":method") != "GET" || cloned.Has(":path") {
		t.Fatalf("original mutation changed the clone: %v", cloned)
	}
	if got := cloned.Values("x-a"); len(got) != 1 || got[0] != "from-clone" {
		t.Fatalf("clone values = %v, want [from-clone]", got)
	}

	var nilHeader Header
	if nilHeader.Clone() != nil {
		t.Fatal("Clone of a nil header must stay nil")
	}
	if got := nilHeader.WithoutPseudo(); len(got) != 0 {
		t.Fatalf("WithoutPseudo of nil = %v", got)
	}
}
