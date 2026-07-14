package auth

import (
	"bufio"
	"io"
	"net"
	"net/http"
)

// protectedResponseWriter makes the authenticated-response cache policy a
// write-time invariant. Handlers and reverse proxies may replace response
// headers after authentication middleware starts, so setting no-store only
// before calling the next handler is not sufficient.
//
// Unwrap and the optional interfaces keep streaming, WebSocket upgrades, HTTP/2
// push, and ResponseController operations available through the middleware.
type protectedResponseWriter struct {
	http.ResponseWriter
}

func newProtectedResponseWriter(w http.ResponseWriter) *protectedResponseWriter {
	return &protectedResponseWriter{ResponseWriter: w}
}

func (w *protectedResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *protectedResponseWriter) enforceCachePolicy() {
	setProtectedResponseCachePolicy(w.ResponseWriter)
}

func (w *protectedResponseWriter) WriteHeader(statusCode int) {
	w.enforceCachePolicy()
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *protectedResponseWriter) Write(data []byte) (int, error) {
	w.enforceCachePolicy()
	return w.ResponseWriter.Write(data)
}

func (w *protectedResponseWriter) ReadFrom(reader io.Reader) (int64, error) {
	w.enforceCachePolicy()
	if readerFrom, ok := w.ResponseWriter.(io.ReaderFrom); ok {
		return readerFrom.ReadFrom(reader)
	}
	return io.Copy(struct{ io.Writer }{w.ResponseWriter}, reader)
}

func (w *protectedResponseWriter) Flush() {
	w.enforceCachePolicy()
	_ = http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *protectedResponseWriter) FlushError() error {
	w.enforceCachePolicy()
	return http.NewResponseController(w.ResponseWriter).Flush()
}

func (w *protectedResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	w.enforceCachePolicy()
	return http.NewResponseController(w.ResponseWriter).Hijack()
}

func (w *protectedResponseWriter) Push(target string, options *http.PushOptions) error {
	w.enforceCachePolicy()
	pusher, ok := w.ResponseWriter.(http.Pusher)
	if !ok {
		return http.ErrNotSupported
	}
	return pusher.Push(target, options)
}
