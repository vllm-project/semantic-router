package managementserver

import (
	"context"
	"mime"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

const (
	maximumAcceptBytes  = 8 << 10
	maximumAcceptRanges = 32
)

type negotiatedMediaContextKey struct{}

type managementTransport struct {
	next *http.ServeMux
}

func newManagementTransport(next *http.ServeMux) http.Handler {
	if next == nil {
		panic("Management transport requires a route mux")
	}
	return &managementTransport{next: next}
}

func (transport *managementTransport) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	requestID := managementRequestID(request)
	if requestHasBody(request) && !isManagementContentType(request.Header.Values("Content-Type")) {
		writeProviderError(response, http.StatusUnsupportedMediaType, "unsupported_media_type",
			"Use the Management API media type for the request body.", requestID)
		return
	}
	supported := supportedManagementResponseMedia(request)
	selected, ok := selectManagementResponseMedia(request.Header.Values("Accept"), supported)
	if !ok {
		writeProviderError(response, http.StatusNotAcceptable, "not_acceptable",
			"Accept a supported Management API media type.", requestID)
		return
	}
	ctx := context.WithValue(request.Context(), negotiatedMediaContextKey{}, selected)
	transport.next.ServeHTTP(response, request.WithContext(ctx))
}

func supportedManagementResponseMedia(request *http.Request) []string {
	if request != nil && request.Method == http.MethodGet && request.URL != nil && request.URL.Path == routingCurrentExportPath {
		return []string{managementapi.YAMLMediaType}
	}
	if isAgentEventRequest(request) {
		return []string{managementapi.JSONMediaType, managementapi.EventStreamMediaType}
	}
	return []string{managementapi.JSONMediaType}
}

func requestHasBody(request *http.Request) bool {
	if request == nil || request.Body == nil || request.Body == http.NoBody {
		return false
	}
	return request.ContentLength != 0 || len(request.TransferEncoding) > 0
}

func isManagementContentType(values []string) bool {
	if len(values) != 1 {
		return false
	}
	mediaType, parameters, err := mime.ParseMediaType(values[0])
	if err != nil || !strings.EqualFold(mediaType, managementapi.JSONMediaType) {
		return false
	}
	if len(parameters) == 0 {
		return true
	}
	return len(parameters) == 1 && strings.EqualFold(parameters["charset"], "utf-8")
}

func selectManagementResponseMedia(values, supported []string) (string, bool) {
	if len(values) == 0 {
		return "", false
	}
	totalBytes, ranges := 0, 0
	selected, selectedQuality := "", -1
	for _, value := range values {
		totalBytes += len(value)
		if totalBytes > maximumAcceptBytes {
			return "", false
		}
		for _, rawRange := range strings.Split(value, ",") {
			ranges++
			if ranges > maximumAcceptRanges {
				return "", false
			}
			mediaType, parameters, err := mime.ParseMediaType(strings.TrimSpace(rawRange))
			if err != nil {
				return "", false
			}
			quality, ok := acceptQuality(parameters)
			if !ok {
				return "", false
			}
			for _, candidate := range supported {
				if strings.EqualFold(mediaType, candidate) && quality > selectedQuality {
					selected, selectedQuality = candidate, quality
				}
			}
		}
	}
	return selected, selected != "" && selectedQuality > 0
}

func acceptQuality(parameters map[string]string) (int, bool) {
	if len(parameters) == 0 {
		return 1000, true
	}
	if len(parameters) != 1 {
		return 0, false
	}
	value, ok := parameters["q"]
	if !ok {
		return 0, false
	}
	if value == "0" {
		return 0, true
	}
	if value == "1" {
		return 1000, true
	}
	if len(value) < 2 || len(value) > 5 || value[1] != '.' || (value[0] != '0' && value[0] != '1') {
		return 0, false
	}
	quality := 0
	for index := 2; index < len(value); index++ {
		if value[index] < '0' || value[index] > '9' || (value[0] == '1' && value[index] != '0') {
			return 0, false
		}
		quality = quality*10 + int(value[index]-'0')
	}
	for index := len(value) - 2; index < 3; index++ {
		quality *= 10
	}
	if value[0] == '1' {
		return 1000, true
	}
	return quality, true
}

func isAgentEventRequest(request *http.Request) bool {
	if request == nil || request.Method != http.MethodGet || request.URL == nil {
		return false
	}
	parts := strings.Split(strings.TrimPrefix(request.URL.Path, agentSessionsPath+"/"), "/")
	return len(parts) == 2 && parts[0] != "" && parts[1] == "events"
}

func negotiatedManagementMedia(request *http.Request) string {
	if request == nil {
		return ""
	}
	value, _ := request.Context().Value(negotiatedMediaContextKey{}).(string)
	return value
}
