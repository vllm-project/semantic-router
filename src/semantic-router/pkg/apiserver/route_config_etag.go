//go:build !windows && cgo

package apiserver

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
)

const configPreconditionRequiredStatus = 428

func configDocumentETag(data []byte) string {
	digest := sha256.Sum256(data)
	return `"` + hex.EncodeToString(digest[:]) + `"`
}

// checkConfigPrecondition implements optimistic concurrency for every config
// mutation. All writers share one canonical document, so every mutation must
// prove which document it intends to replace.
func checkConfigPrecondition(
	w http.ResponseWriter,
	r *http.Request,
	currentData []byte,
) bool {
	ifMatch := strings.TrimSpace(r.Header.Get("If-Match"))
	if ifMatch == "" {
		writeConfigPreconditionError(
			w,
			configPreconditionRequiredStatus,
			"PRECONDITION_REQUIRED",
			"If-Match is required; plan or read the current config before mutating it",
		)
		return false
	}

	currentETag := configDocumentETag(currentData)
	for _, candidate := range strings.Split(ifMatch, ",") {
		candidate = strings.TrimSpace(candidate)
		if candidate == currentETag {
			return true
		}
	}

	w.Header().Set("ETag", currentETag)
	writeConfigPreconditionError(
		w,
		http.StatusPreconditionFailed,
		"CONFIG_CHANGED",
		"The router config changed since it was read; refetch and retry with the latest ETag",
	)
	return false
}

func writeConfigPreconditionError(w http.ResponseWriter, status int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write([]byte(`{"error":{"code":"` + code + `","message":"` + message + `"}}` + "\n"))
}
