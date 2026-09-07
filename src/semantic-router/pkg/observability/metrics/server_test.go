package metrics

import (
	"net/http"
	"net/http/httptest"
	"testing"

	// Imported for its side effect: net/http/pprof registers the debug
	// endpoints on http.DefaultServeMux, which is exactly the leak the metrics
	// mux must not inherit.
	_ "net/http/pprof"
)

func TestMetricsServeMuxDoesNotExposeProfiling(t *testing.T) {
	// Guards against the assertion below passing vacuously: it is only a
	// regression test while the default mux really does carry pprof.
	defaultRecorder := httptest.NewRecorder()
	http.DefaultServeMux.ServeHTTP(defaultRecorder, httptest.NewRequest(http.MethodGet, "/debug/pprof/", nil))
	if defaultRecorder.Code == http.StatusNotFound {
		t.Fatal("net/http/pprof did not register on http.DefaultServeMux; the leak this test guards can no longer occur")
	}

	for _, path := range []string{"/debug/pprof/", "/debug/pprof/heap", "/debug/pprof/profile"} {
		recorder := httptest.NewRecorder()
		NewServeMux().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))

		if recorder.Code != http.StatusNotFound {
			t.Fatalf("GET %s on the metrics mux returned %d, want %d", path, recorder.Code, http.StatusNotFound)
		}
	}
}
