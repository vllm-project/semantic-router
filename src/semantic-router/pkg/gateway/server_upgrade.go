package gateway

import (
	"io"
	"net/http"
	"time"

	"golang.org/x/net/http/httpguts"
)

// boundH2CUpgrade bounds the body x/net's h2c adapter buffers before entering
// the gateway. Prior-knowledge HTTP/2 and ordinary HTTP/1 requests pass through.
func boundH2CUpgrade(handler, adapter http.Handler, timeout time.Duration) http.Handler {
	limit := int64(DefaultMaxRequestBodyBytes)
	if gateway, ok := handler.(*Handler); ok {
		limit, timeout = gateway.opts.MaxRequestBodyBytes, gateway.opts.ReadTimeout
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Match the adapter's Upgrade detection, including token lists.
		if !httpguts.HeaderValuesContainsToken(r.Header.Values("Upgrade"), "h2c") ||
			!httpguts.HeaderValuesContainsToken(r.Header.Values("Connection"), "HTTP2-Settings") {
			adapter.ServeHTTP(w, r)
			return
		}
		if r.ContentLength > limit {
			closeUpgradeRequest(w)
			writeBodyReadError(w, errBodyTooLarge)
			return
		}
		controller := http.NewResponseController(w)
		if timeout > 0 {
			_ = controller.SetReadDeadline(time.Now().Add(timeout))
		}
		body := &upgradeBody{ReadCloser: http.MaxBytesReader(w, r.Body, limit), controller: controller}
		r.Body = body
		adapter.ServeHTTP(&upgradeWriter{ResponseWriter: w, body: body}, r)
	})
}

// closeUpgradeRequest prevents net/http from draining an unread body while
// writing or closing the response, even when the configured read timeout is zero.
func closeUpgradeRequest(w http.ResponseWriter) {
	w.Header().Set("Connection", "close")
	_ = http.NewResponseController(w).SetReadDeadline(time.Now())
}

type upgradeBody struct {
	io.ReadCloser
	controller *http.ResponseController
	err        error
}

func (b *upgradeBody) Read(p []byte) (int, error) {
	n, err := b.ReadCloser.Read(p)
	if err == io.EOF {
		// The deadline must end before hijacking: a legitimate response, or
		// another stream on the upgraded connection, can take much longer.
		_ = b.controller.SetReadDeadline(time.Time{})
	} else if err != nil {
		b.err = err
	}
	return n, err
}

type upgradeWriter struct {
	http.ResponseWriter
	body *upgradeBody
}

func (w *upgradeWriter) Unwrap() http.ResponseWriter { return w.ResponseWriter }

func (w *upgradeWriter) WriteHeader(status int) {
	// The adapter reports every pre-hijack error as 500. Translate body
	// failures only; malformed settings and other adapter failures stay 500.
	if status == http.StatusInternalServerError {
		closeUpgradeRequest(w.ResponseWriter)
		if w.body.err != nil {
			writeBodyReadError(w.ResponseWriter, w.body.err)
			return
		}
	}
	w.ResponseWriter.WriteHeader(status)
}
