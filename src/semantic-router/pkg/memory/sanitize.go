package memory

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const maxMemoryContentBytes = 16384 // 16 KB hard limit per memory chunk

// sanitizeMemoryContent validates memory content before storage.
// Returns the (possibly trimmed) content and an error if the content
// is structurally invalid (bad UTF-8, empty). Adversarial content
// blocking is handled by SR's jailbreak classifier on the request path.
func sanitizeMemoryContent(content string) (string, error) {
	if !utf8.ValidString(content) {
		return "", fmt.Errorf("content contains invalid UTF-8")
	}

	content = strings.TrimSpace(content)
	if content == "" {
		return "", fmt.Errorf("content is empty after trimming")
	}

	if len(content) > maxMemoryContentBytes {
		logging.Debugf("sanitize: truncating memory from %d to %d bytes", len(content), maxMemoryContentBytes)
		content = truncateUTF8(content, maxMemoryContentBytes)
	}

	return content, nil
}

// truncateUTF8 truncates a string to at most maxBytes while preserving
// valid UTF-8 boundaries.
func truncateUTF8(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	for maxBytes > 0 && !utf8.RuneStart(s[maxBytes]) {
		maxBytes--
	}
	return s[:maxBytes]
}
