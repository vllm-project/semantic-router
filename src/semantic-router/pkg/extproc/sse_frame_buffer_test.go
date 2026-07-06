package extproc

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestReassembleSSEFrames covers the SSE frame reassembly used to survive
// Envoy STREAMED-mode chunk boundaries that split events mid-frame (#2316).
func TestReassembleSSEFrames(t *testing.T) {
	cases := []struct {
		name          string
		pending       string
		chunk         string
		wantComplete  string
		wantRemainder string
	}{
		{
			name:          "single complete frame passes through",
			pending:       "",
			chunk:         "data: {\"a\":1}\n\n",
			wantComplete:  "data: {\"a\":1}\n\n",
			wantRemainder: "",
		},
		{
			name:          "frame split mid-json is held as remainder",
			pending:       "",
			chunk:         "data: {\"a\":",
			wantComplete:  "",
			wantRemainder: "data: {\"a\":",
		},
		{
			name:          "pending remainder completes on next chunk",
			pending:       "data: {\"a\":",
			chunk:         "1}\n\n",
			wantComplete:  "data: {\"a\":1}\n\n",
			wantRemainder: "",
		},
		{
			name:          "trailing partial after a complete frame is held",
			pending:       "",
			chunk:         "data: {\"a\":1}\n\ndata: {\"b\":",
			wantComplete:  "data: {\"a\":1}\n\n",
			wantRemainder: "data: {\"b\":",
		},
		{
			name:          "empty chunk with no pending yields nothing",
			pending:       "",
			chunk:         "",
			wantComplete:  "",
			wantRemainder: "",
		},
		{
			name:          "boundary split exactly between the two newlines",
			pending:       "data: {\"a\":1}\n",
			chunk:         "\ndata: {\"b\":2}\n\n",
			wantComplete:  "data: {\"a\":1}\n\ndata: {\"b\":2}\n\n",
			wantRemainder: "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			complete, remainder := reassembleSSEFrames([]byte(tc.pending), []byte(tc.chunk))
			assert.Equal(t, tc.wantComplete, string(complete), "complete frames")
			assert.Equal(t, tc.wantRemainder, string(remainder), "held remainder")
		})
	}
}

// TestReassembleSSEFrames_RemainderDoesNotAliasChunk asserts the returned
// remainder is copied out of the caller's chunk buffer, so retaining it
// across calls is safe even if Envoy reuses the underlying array.
func TestReassembleSSEFrames_RemainderDoesNotAliasChunk(t *testing.T) {
	chunk := []byte("data: {\"a\":")
	_, remainder := reassembleSSEFrames(nil, chunk)
	// Mutate the caller's buffer; the remainder must be unaffected.
	for i := range chunk {
		chunk[i] = 'X'
	}
	assert.Equal(t, "data: {\"a\":", string(remainder), "remainder must not alias the chunk buffer")
}
