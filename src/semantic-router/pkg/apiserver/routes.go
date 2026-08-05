//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
)

type apiRouteHandler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)

type requestBodyKind string

const (
	requestBodyNone      requestBodyKind = ""
	requestBodyJSON      requestBodyKind = "application/json"
	requestBodyMultipart requestBodyKind = "multipart/form-data"
)

type apiRequestBody struct {
	Kind        requestBodyKind
	Required    bool
	LimitBytes  int64
	Description string
}

type apiRoute struct {
	EndpointMetadata
	Handler     apiRouteHandler
	RequestBody apiRequestBody
	Permission  RoutePermission
	Sensitivity RouteSensitivity
	AuditAction RouteAuditAction
}

func (r apiRoute) pattern() string {
	return fmt.Sprintf("%s %s", r.Method, r.Path)
}

func (r apiRoute) bind(s *ClassificationAPIServer) http.HandlerFunc {
	handler := func(w http.ResponseWriter, req *http.Request) {
		r.Handler(s, w, req)
	}
	return s.wrapRouteHandler(r, handler)
}

func jsonBody() apiRequestBody {
	return jsonBodyWithLimit(defaultJSONRequestBodyLimit)
}

func jsonBodyWithLimit(limit int64) apiRequestBody {
	return apiRequestBody{
		Kind:       requestBodyJSON,
		Required:   true,
		LimitBytes: limit,
	}
}

func multipartBody(limit int64, description string) apiRequestBody {
	return apiRequestBody{
		Kind:        requestBodyMultipart,
		Required:    true,
		LimitBytes:  limit,
		Description: description,
	}
}

func apiEndpointMetadata() []EndpointMetadata {
	routes := apiRoutes()
	metadata := make([]EndpointMetadata, 0, len(routes))
	for _, route := range routes {
		metadata = append(metadata, route.EndpointMetadata)
	}
	return metadata
}

func apiRoutes() []apiRoute {
	return appendAPIRoutes(
		make([]apiRoute, 0, 64),
		apiHealthRoutes(),
		apiClassifyRoutes(),
		apiInfoRoutes(),
		apiConfigRoutes(),
		apiMemoryRoutes(),
		apiVectorStoreRoutes(),
		apiFileRoutes(),
	)
}
