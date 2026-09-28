//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"strconv"
)

// apiRouteOption keeps request and response contracts next to their handler.
// Response contracts never depend on URL spelling or HTTP method heuristics.
type apiRouteOption interface{ applyRoute(*apiRoute) }

func (body apiRequestBody) applyRoute(route *apiRoute) { route.RequestBody = body }

type routeOptionFunc func(*apiRoute)

func (option routeOptionFunc) applyRoute(route *apiRoute) { option(route) }

func jsonResponse[T any](status int, description string) apiRouteOption {
	return mediaResponse(status, description, "application/json", *openAPIRequestSchemaFor[T]())
}

func mediaResponse(status int, description, mediaType string, schema OpenAPISchema) apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) {
		response := route.Responses[strconv.Itoa(status)]
		response.Description = description
		if response.Content == nil {
			response.Content = make(map[string]OpenAPIMedia)
		}
		response.Content[mediaType] = OpenAPIMedia{Schema: &schema}
		setRouteResponse(route, status, response)
	})
}

func emptyResponse(status int, description string) apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) { setRouteResponse(route, status, OpenAPIResponse{Description: description}) })
}

func errorResponses(statuses ...int) apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) {
		for _, status := range statuses {
			setRouteResponse(route, status, openAPIErrorResponse(http.StatusText(status)))
		}
	})
}

func errorResponse(status int, description string) apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) { setRouteResponse(route, status, openAPIErrorResponse(description)) })
}

func responseHeaders(status int, headers map[string]OpenAPIHeader) apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) {
		response := route.Responses[strconv.Itoa(status)]
		response.Headers = headers
		setRouteResponse(route, status, response)
	})
}

func setRouteResponse(route *apiRoute, status int, response OpenAPIResponse) {
	if route.Responses == nil {
		route.Responses = make(map[string]OpenAPIResponse)
	}
	route.Responses[strconv.Itoa(status)] = response
}

func routeOpenAPIResponses(route apiRoute) map[string]OpenAPIResponse {
	responses := make(map[string]OpenAPIResponse, len(route.Responses)+2)
	for status, response := range route.Responses {
		headers := make(map[string]OpenAPIHeader, len(response.Headers)+1)
		for name, header := range response.Headers {
			headers[name] = header
		}
		headers[managementRequestIDHeader] = OpenAPIHeader{Description: "Correlation identifier for this management request.", Schema: OpenAPISchema{Type: "string"}}
		response.Headers = headers
		responses[status] = response
	}
	if route.Permission != PermHealthRead {
		responses["401"] = openAPIErrorResponse("A management bearer token is required or invalid when bearer authentication is enabled")
		responses["403"] = openAPIErrorResponse("The authenticated role lacks the operation permission")
		if _, exists := responses["500"]; !exists {
			responses["500"] = openAPIErrorResponse("Management authentication configuration is invalid")
		}
	}
	return responses
}

// jsonResponseOrError records operations that can return a structured result
// or the common error envelope at the same HTTP status.
func jsonResponseOrError[T any](status int, description string) apiRouteOption {
	return mediaResponse(status, description, "application/json", OpenAPISchema{OneOf: []OpenAPISchema{
		*openAPIRequestSchemaFor[T](), *openAPIRequestSchemaFor[managementErrorResponse](),
	}})
}

func etagResponseHeaders(status int) apiRouteOption {
	return responseHeaders(status, map[string]OpenAPIHeader{"ETag": {Description: "Current document version; use as If-Match for a conditional mutation or If-None-Match for a schema read.", Schema: OpenAPISchema{Type: "string"}}})
}

func configMutationResponses() apiRouteOption {
	return routeOptionFunc(func(route *apiRoute) {
		for _, status := range []int{http.StatusOK, http.StatusAccepted} {
			jsonResponse[RouterConfigUpdateResponse](status, "Configuration persisted; activation_status and generated_runtime_hash identify its runtime publication").applyRoute(route)
			etagResponseHeaders(status).applyRoute(route)
		}
		jsonResponseOrError[RouterConfigUpdateResponse](http.StatusServiceUnavailable, "Configuration persisted but activation failed; the active runtime remains on its previous generation").applyRoute(route)
		etagResponseHeaders(http.StatusServiceUnavailable).applyRoute(route)
		errorResponses(http.StatusBadRequest, http.StatusForbidden, http.StatusNotFound, http.StatusConflict, http.StatusPreconditionFailed, http.StatusPreconditionRequired, http.StatusInternalServerError).applyRoute(route)
		etagResponseHeaders(http.StatusPreconditionFailed).applyRoute(route)
	})
}
