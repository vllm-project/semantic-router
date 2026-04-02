package handlers

import "testing"

func TestShouldPreferJinaFetch(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		url  string
		req  OpenWebRequest
		want bool
	}{
		{
			name: "ordinary html stays on direct-first path",
			url:  "https://example.com/article",
			req:  OpenWebRequest{},
			want: false,
		},
		{
			name: "explicit force_jina wins",
			url:  "https://example.com/article",
			req: OpenWebRequest{
				ForceJina: true,
			},
			want: true,
		},
		{
			name: "with_images keeps jina path",
			url:  "https://example.com/article",
			req: OpenWebRequest{
				WithImages: true,
			},
			want: true,
		},
		{
			name: "pdf urls keep jina path",
			url:  "https://example.com/guide.pdf",
			req:  OpenWebRequest{},
			want: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			if got := shouldPreferJinaFetch(tt.url, tt.req); got != tt.want {
				t.Fatalf("shouldPreferJinaFetch(%q) = %v, want %v", tt.url, got, tt.want)
			}
		})
	}
}

func TestNormalizeOpenWebMaxLength(t *testing.T) {
	t.Parallel()

	if got := normalizeOpenWebMaxLength(0); got != openWebMaxContentLength {
		t.Fatalf("normalizeOpenWebMaxLength(0) = %d, want %d", got, openWebMaxContentLength)
	}

	if got := normalizeOpenWebMaxLength(1200); got != 1200 {
		t.Fatalf("normalizeOpenWebMaxLength(1200) = %d, want 1200", got)
	}

	if got := normalizeOpenWebMaxLength(openWebMaxContentLength + 1); got != openWebMaxContentLength {
		t.Fatalf(
			"normalizeOpenWebMaxLength(max+1) = %d, want %d",
			got,
			openWebMaxContentLength,
		)
	}
}
