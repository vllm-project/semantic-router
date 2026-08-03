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

// checkConfigPrecondition implements optimistic concurrency for config
// mutations. Existing whole-document APIs remain backward compatible when
// require is false; focused subresource APIs require If-Match because they
// read-modify-write one shared canonical document.
func checkConfigPrecondition(
	w http.ResponseWriter,
	r *http.Request,
	currentData []byte,
	require bool,
) bool {
	ifMatch := strings.TrimSpace(r.Header.Get("If-Match"))
	if ifMatch == "" {
		if !require {
			return true
		}
		writeConfigPreconditionError(
			w,
			configPreconditionRequiredStatus,
			"PRECONDITION_REQUIRED",
			"If-Match is required; fetch the recipe collection ETag before mutating it",
		)
		return false
	}

	currentETag := configDocumentETag(currentData)
	for _, candidate := range strings.Split(ifMatch, ",") {
		candidate = strings.TrimSpace(candidate)
		if candidate == "*" || candidate == currentETag {
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
