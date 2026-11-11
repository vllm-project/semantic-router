package router

import (
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

var webSearchStatsOperation = apicontract.Operation{
	ID:      "getWebSearchRateLimitStats",
	Summary: "WebSearch rate-limit snapshot",
	Responses: map[int]apicontract.Response{
		http.StatusOK: apicontract.JSONResponse[handlers.WebSearchRateLimitStatsResponse]("Limiter counts for the current window and the fixed limits"),
	},
}
