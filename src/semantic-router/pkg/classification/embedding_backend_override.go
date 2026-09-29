package classification

import (
	"os"
	"strings"
)

func embeddingBackendOverride() string {
	return strings.ToLower(strings.TrimSpace(os.Getenv("EMBEDDING_BACKEND_OVERRIDE")))
}
