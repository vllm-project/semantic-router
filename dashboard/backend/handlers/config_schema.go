package handlers

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/routerauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

const (
	configSchemaSourceHeader = "X-Vllm-Sr-Schema-Source"
	configSchemaMatchHeader  = "X-Vllm-Sr-Schema-Match"
	maxConfigSchemaBytes     = 4 << 20
)

// ConfigSchemaHandler serves the deployed Router contract when it is
// reachable and falls back to the exact contract bundled with this Dashboard.
// Both sources support the same progressive full/index/section/surface views.
func ConfigSchemaHandler(routerAPIURL string, credentialProvider routerauth.CredentialProvider) http.HandlerFunc {
	client := &http.Client{Timeout: 5 * time.Second}
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if strings.TrimSpace(routerAPIURL) != "" {
			served, _ := serveRuntimeConfigSchema(
				w,
				r,
				strings.TrimSpace(routerAPIURL),
				client,
				credentialProvider,
			)
			if served {
				return
			}
		}
		serveBundledConfigSchema(w, r)
	}
}

func serveRuntimeConfigSchema(
	w http.ResponseWriter,
	r *http.Request,
	routerAPIURL string,
	client *http.Client,
	credentialProvider routerauth.CredentialProvider,
) (bool, error) {
	targetURL := strings.TrimSuffix(routerAPIURL, "/") + configschema.SchemaEndpoint
	if r.URL.RawQuery != "" {
		targetURL += "?" + r.URL.RawQuery
	}
	request, err := http.NewRequestWithContext(r.Context(), http.MethodGet, targetURL, nil)
	if err != nil {
		return false, err
	}
	request.Header.Set("Accept", "application/schema+json, application/json")
	if authErr := routerauth.RewriteAuthorization(request, credentialProvider); authErr != nil {
		return false, authErr
	}
	response, err := client.Do(request)
	if err != nil {
		return false, err
	}
	defer response.Body.Close()
	body, err := io.ReadAll(io.LimitReader(response.Body, maxConfigSchemaBytes+1))
	if err != nil {
		return false, err
	}
	if len(body) > maxConfigSchemaBytes {
		return false, errors.New("runtime config schema exceeds size limit")
	}
	if response.StatusCode == http.StatusNotFound || response.StatusCode >= http.StatusInternalServerError {
		return false, errors.New("runtime config schema is unavailable")
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		copyConfigSchemaHeaders(w.Header(), response.Header)
		w.Header().Set(configSchemaSourceHeader, "runtime")
		w.WriteHeader(response.StatusCode)
		_, _ = w.Write(body)
		return true, nil
	}
	etag := response.Header.Get("ETag")
	if etag == "" {
		etag = contentETag(body)
	}
	copyConfigSchemaHeaders(w.Header(), response.Header)
	w.Header().Set("ETag", etag)
	w.Header().Set(configSchemaSourceHeader, "runtime")
	w.Header().Set(configSchemaMatchHeader, runtimeSchemaMatch(r, etag))
	if r.Header.Get("If-None-Match") == etag {
		w.WriteHeader(http.StatusNotModified)
		return true, nil
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
	return true, nil
}

func serveBundledConfigSchema(w http.ResponseWriter, r *http.Request) {
	representation, err := configschema.Render(configschema.ViewOptions{
		View:        r.URL.Query().Get("view"),
		Path:        r.URL.Query().Get("path"),
		SurfaceKind: r.URL.Query().Get("kind"),
		SurfaceName: r.URL.Query().Get("name"),
	})
	if err != nil {
		var viewError *configschema.ViewError
		if errors.As(err, &viewError) {
			http.Error(w, viewError.Error(), http.StatusBadRequest)
			return
		}
		http.Error(w, "Config schema is unavailable", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", representation.ContentType)
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("ETag", representation.ETag)
	w.Header().Set(configSchemaSourceHeader, "bundled")
	w.Header().Set(configSchemaMatchHeader, "unknown")
	if r.Header.Get("If-None-Match") == representation.ETag {
		w.WriteHeader(http.StatusNotModified)
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(representation.Body)
}

func runtimeSchemaMatch(r *http.Request, runtimeETag string) string {
	local, err := configschema.Render(configschema.ViewOptions{
		View:        r.URL.Query().Get("view"),
		Path:        r.URL.Query().Get("path"),
		SurfaceKind: r.URL.Query().Get("kind"),
		SurfaceName: r.URL.Query().Get("name"),
	})
	if err != nil {
		return "unknown"
	}
	if local.ETag == runtimeETag {
		return "true"
	}
	return "false"
}

func copyConfigSchemaHeaders(destination, source http.Header) {
	for _, key := range []string{"Content-Type", "Cache-Control"} {
		if value := source.Get(key); value != "" {
			destination.Set(key, value)
		}
	}
	if destination.Get("Content-Type") == "" {
		destination.Set("Content-Type", "application/schema+json")
	}
	destination.Set("Cache-Control", "no-cache")
}

func contentETag(body []byte) string {
	digest := sha256.Sum256(body)
	return `"sha256:` + hex.EncodeToString(digest[:]) + `"`
}
