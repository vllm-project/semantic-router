package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

var traceSummariesOperation = apicontract.Operation{
	ID:      "listTraceSummaries",
	Summary: "Summaries of the traces a Jaeger search finds",
	Description: "Runs the search against Jaeger at TARGET_JAEGER_URL on the server and answers one summary per trace, newest first, " +
		"instead of the traces themselves; fetch a full trace from /embedded/jaeger/api/traces/{traceID}. " +
		"Query: service (required), operation, lookback (a Go duration, default 1h, at most 720h), limit (1 to 100, default 20), " +
		"minDuration and maxDuration (Go durations). traceID is Jaeger's full ID; startTimeMicros is microseconds since the Unix epoch, " +
		"and durationMicros runs from the first span's start to the last span's end. The root is the earliest span without a CHILD_OF " +
		"or FOLLOWS_FROM parent in the trace; services lists the root's service first, then the busiest. errorCount counts spans tagged " +
		"error=true or otel.status_code=ERROR. Jaeger's answer is read for at most 20 s and 64 MiB.",
	Responses: map[int]apicontract.Response{
		http.StatusOK:                 apicontract.JSONResponse[handlers.TraceSummariesResponse]("One summary per trace, newest first; total counts data"),
		http.StatusBadRequest:         apicontract.JSONResponse[handlers.TraceSummariesErrorResponse]("error invalid_request for a missing service, a malformed parameter, or a search Jaeger rejected"),
		http.StatusBadGateway:         apicontract.JSONResponse[handlers.TraceSummariesErrorResponse]("error jaeger_unreachable, jaeger_bad_response when the answer is not a JSON search result, jaeger_response_too_large over 64 MiB, or jaeger_error when Jaeger failed the search"),
		http.StatusServiceUnavailable: apicontract.JSONResponse[handlers.TraceSummariesErrorResponse]("error jaeger_not_configured when TARGET_JAEGER_URL is empty or not an absolute URL"),
		http.StatusGatewayTimeout:     apicontract.JSONResponse[handlers.TraceSummariesErrorResponse]("error jaeger_timeout when Jaeger does not finish the search within 20 s"),
	},
}
