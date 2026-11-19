package router

import (
	"bytes"
	"compress/gzip"
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
)

// replayGzipMinBytes is the smallest Replay answer worth compressing: below
// it, gzip's header and a round of work outweigh the bytes saved.
const replayGzipMinBytes = 1024

type replayGzipKey struct{}

// withReplayGzip remembers that the browser accepts gzip before the proxy
// drops its Accept-Encoding, which lets Replay JSON reach the Dashboard
// uncompressed and redactable.
func withReplayGzip(r *http.Request) *http.Request {
	if !acceptsGzip(r.Header.Get("Accept-Encoding")) {
		return r
	}
	return r.WithContext(context.WithValue(r.Context(), replayGzipKey{}, true))
}

func acceptsGzip(header string) bool {
	for _, part := range strings.Split(header, ",") {
		name, params, _ := strings.Cut(strings.TrimSpace(part), ";")
		if !strings.EqualFold(strings.TrimSpace(name), "gzip") {
			continue
		}
		quality := strings.ReplaceAll(strings.ToLower(params), " ", "")
		return quality != "q=0" && quality != "q=0.0" && quality != "q=0.00" && quality != "q=0.000"
	}
	return false
}

// compressRouterReplayResponse gzips a Replay JSON answer for a browser that
// accepts it, after any redaction. A long agent run answers megabytes of
// JSON, which compress about tenfold: on a slow link that is most of the wait
// to open a session.
func compressRouterReplayResponse(resp *http.Response) error {
	if !shouldRedactRouterReplayResponse(resp) {
		return nil
	}
	if wanted, _ := resp.Request.Context().Value(replayGzipKey{}).(bool); !wanted {
		return nil
	}
	if resp.Header.Get("Content-Encoding") != "" ||
		!strings.HasPrefix(strings.ToLower(resp.Header.Get("Content-Type")), "application/json") {
		return nil
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	_ = resp.Body.Close()
	resp.Header.Add("Vary", "Accept-Encoding")
	if len(body) < replayGzipMinBytes {
		resp.Body = io.NopCloser(bytes.NewReader(body))
		return nil
	}
	var compressed bytes.Buffer
	writer, err := gzip.NewWriterLevel(&compressed, gzip.BestSpeed)
	if err != nil {
		return err
	}
	if _, err = writer.Write(body); err != nil {
		return err
	}
	if err = writer.Close(); err != nil {
		return err
	}
	resp.Body = io.NopCloser(&compressed)
	resp.ContentLength = int64(compressed.Len())
	resp.Header.Set("Content-Length", strconv.Itoa(compressed.Len()))
	resp.Header.Set("Content-Encoding", "gzip")
	return nil
}
