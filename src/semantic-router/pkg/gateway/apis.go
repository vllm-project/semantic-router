package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"
)

// writeDetail answers with a {"detail": ...} body, the error shape of a path
// the listener does not serve.
func writeDetail(w http.ResponseWriter, status int, detail string) {
	body, _ := json.Marshal(map[string]string{"detail": detail})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", strconv.Itoa(len(body)))
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

// bufferedResponse holds a response in memory, for a listing made of two.
type bufferedResponse struct {
	header http.Header
	status int
	body   bytes.Buffer
}

func newBufferedResponse() *bufferedResponse {
	return &bufferedResponse{header: http.Header{}, status: http.StatusOK}
}

func (b *bufferedResponse) Header() http.Header         { return b.header }
func (b *bufferedResponse) Write(p []byte) (int, error) { return b.body.Write(p) }
func (b *bufferedResponse) WriteHeader(status int)      { b.status = status }

// writeTo writes the held response.
func (b *bufferedResponse) writeTo(w http.ResponseWriter) {
	for name, values := range b.header {
		w.Header()[name] = values
	}
	w.WriteHeader(b.status)
	_, _ = w.Write(b.body.Bytes())
}
