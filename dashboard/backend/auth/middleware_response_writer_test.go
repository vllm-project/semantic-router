package auth

import (
	"bufio"
	"errors"
	"io"
	"net"
	"net/http"
	"strings"
	"testing"
)

func TestProtectedResponseWriterPreservesStreamingAndUpgradeInterfaces(t *testing.T) {
	t.Parallel()

	underlying := newOptionalResponseWriter()
	wrapped := newProtectedResponseWriter(underlying)
	wrapped.Header().Set("Cache-Control", "public")
	wrapped.Header().Set("Pragma", "cache")

	if unwrapper, ok := any(wrapped).(interface{ Unwrap() http.ResponseWriter }); !ok {
		t.Fatal("protected writer does not expose Unwrap")
	} else if unwrapper.Unwrap() != underlying {
		t.Fatal("Unwrap did not return the underlying writer")
	}
	hijacker, ok := any(wrapped).(http.Hijacker)
	if !ok {
		t.Fatal("protected writer does not preserve http.Hijacker")
	}
	if _, _, err := hijacker.Hijack(); !errors.Is(err, http.ErrNotSupported) {
		t.Fatalf("Hijack() error = %v, want http.ErrNotSupported", err)
	}
	if !underlying.hijacked {
		t.Fatal("Hijack was not forwarded to the underlying writer")
	}
	pusher, ok := any(wrapped).(http.Pusher)
	if !ok {
		t.Fatal("protected writer does not preserve http.Pusher")
	}
	if err := pusher.Push("/asset.js", nil); !errors.Is(err, http.ErrNotSupported) {
		t.Fatalf("Push() error = %v, want http.ErrNotSupported", err)
	}
	if !underlying.pushed {
		t.Fatal("Push was not forwarded to the underlying writer")
	}
	readerFrom, ok := any(wrapped).(io.ReaderFrom)
	if !ok {
		t.Fatal("protected writer does not preserve io.ReaderFrom")
	}
	if _, err := readerFrom.ReadFrom(strings.NewReader("streamed")); err != nil {
		t.Fatalf("ReadFrom() error = %v", err)
	}
	if got := underlying.body.String(); got != "streamed" {
		t.Fatalf("streamed body = %q, want streamed", got)
	}

	wrapped.Header().Set("Cache-Control", "no-cache")
	wrapped.Header().Set("Pragma", "cache")
	if err := http.NewResponseController(wrapped).Flush(); err != nil {
		t.Fatalf("ResponseController.Flush() error = %v", err)
	}
	if !underlying.flushed {
		t.Fatal("Flush was not forwarded to the underlying writer")
	}
	if got := underlying.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control after streaming = %q, want no-store", got)
	}
	if got := underlying.Header().Get("Pragma"); got != "no-cache" {
		t.Fatalf("Pragma after streaming = %q, want no-cache", got)
	}
}

type optionalResponseWriter struct {
	header   http.Header
	body     strings.Builder
	status   int
	flushed  bool
	hijacked bool
	pushed   bool
}

func newOptionalResponseWriter() *optionalResponseWriter {
	return &optionalResponseWriter{header: make(http.Header)}
}

func (w *optionalResponseWriter) Header() http.Header { return w.header }

func (w *optionalResponseWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.body.Write(data)
}

func (w *optionalResponseWriter) WriteHeader(statusCode int) { w.status = statusCode }

func (w *optionalResponseWriter) ReadFrom(reader io.Reader) (int64, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return io.Copy(&w.body, reader)
}

func (w *optionalResponseWriter) Flush() { w.flushed = true }

func (w *optionalResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	w.hijacked = true
	return nil, nil, http.ErrNotSupported
}

func (w *optionalResponseWriter) Push(string, *http.PushOptions) error {
	w.pushed = true
	return http.ErrNotSupported
}
