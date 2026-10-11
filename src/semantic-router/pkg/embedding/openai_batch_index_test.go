package embedding

import (
	"context"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

type embeddingIndexTransport func(*http.Request) (*http.Response, error)

func (f embeddingIndexTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestOpenAICompatibleProviderBatchIndexes(t *testing.T) {
	for _, tc := range []struct {
		name     string
		response string
		input    []string
		want     [][]float32
		wantErr  string
	}{
		{
			name:     "ordered",
			response: `{"data":[{"index":0,"embedding":[1,0]},{"index":1,"embedding":[0,1]}]}`,
			want:     [][]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "reordered",
			response: `{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}`,
			want:     [][]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "duplicate_zero",
			response: `{"data":[{"index":0,"embedding":[1,0]},{"index":0,"embedding":[0,1]}]}`,
			wantErr:  "duplicate embedding index 0",
		},
		{
			name:     "duplicate_nonzero",
			response: `{"data":[{"index":1,"embedding":[0,1]},{"index":1,"embedding":[1,0]}]}`,
			wantErr:  "duplicate embedding index 1",
		},
		{
			name:     "missing_indexes",
			response: `{"data":[{"embedding":[1,0]},{"embedding":[0,1]}]}`,
			want:     [][]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "null_indexes",
			response: `{"data":[{"index":null,"embedding":[1,0]},{"index":null,"embedding":[0,1]}]}`,
			want:     [][]float32{{1, 0}, {0, 1}},
		},
		{
			name:     "single_index",
			response: `{"data":[{"index":0,"embedding":[1,0]}]}`,
			input:    []string{"single input"},
			want:     [][]float32{{1, 0}},
		},
		{
			name:     "single_missing_index",
			response: `{"data":[{"embedding":[1,0]}]}`,
			input:    []string{"single input"},
			want:     [][]float32{{1, 0}},
		},
		{
			name:     "single_null_index",
			response: `{"data":[{"index":null,"embedding":[1,0]}]}`,
			input:    []string{"single input"},
			want:     [][]float32{{1, 0}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			client := &http.Client{Transport: embeddingIndexTransport(func(r *http.Request) (*http.Response, error) {
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(tc.response)),
					Request:    r,
				}, nil
			})}
			provider := newTestOpenAIProvider(t, OpenAICompatibleConfig{
				BaseURL:           "http://embedding.invalid/v1",
				Model:             "fixture",
				ExpectedDimension: 2,
				HTTPClient:        client,
			})
			defer provider.Close()
			input := tc.input
			if input == nil {
				input = []string{"first input", "second input"}
			}
			got, err := provider.EmbedBatch(context.Background(), input)
			t.Logf("vectors=%v error=%v", got, err)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("EmbedBatch() error = %v, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("EmbedBatch() = %v, %v; want vectors in input-index order %v", got, err, tc.want)
			}
		})
	}
}
