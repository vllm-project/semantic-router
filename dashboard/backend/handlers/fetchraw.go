package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

const (
	fetchRawMaxSize = 2 * 1024 * 1024 // 2 MB
	fetchRawTimeout = 15 * time.Second
)

// FetchRawRequest is the request body for the fetch-raw endpoint.
type FetchRawRequest struct {
	URL string `json:"url"`
}

// FetchRawResponse is the response body for the fetch-raw endpoint.
type FetchRawResponse struct {
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

// FetchRawHandler returns an HTTP handler that proxies a GET request to the
// given URL and returns the raw text body without any HTML cleaning or
// truncation. Designed for fetching YAML/JSON config files from remote URLs,
// bypassing browser CORS restrictions.
func FetchRawHandler() http.HandlerFunc {
	return fetchRawHandlerWithClient(newPublicOutboundHTTPClient(fetchRawTimeout))
}

func fetchRawHandlerWithClient(client outboundHTTPClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")

		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "method not allowed, use POST"})
			return
		}

		var req FetchRawRequest
		if status, err := decodeBoundedJSON(w, r, outboundMaxRequestBodyBytes, &req); err != nil {
			w.WriteHeader(status)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid request body"})
			return
		}

		parsedURL, err := parseOutboundHTTPURL(req.URL)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid URL, must be public http or https"})
			return
		}
		targetURL := parsedURL.String()
		if validationErr := client.ValidateURL(r.Context(), targetURL); validationErr != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "invalid URL, must be public http or https"})
			return
		}

		log.Printf("[FetchRaw] Fetching: %s", redactURLForLog(targetURL))

		httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodGet, targetURL, nil)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "failed to create request"})
			return
		}

		httpReq.Header.Set("User-Agent", getRandomUserAgent())
		httpReq.Header.Set("Accept", "text/plain, application/x-yaml, application/json, */*")

		resp, err := client.Do(httpReq)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "fetch failed"})
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: fmt.Sprintf("remote returned HTTP %d", resp.StatusCode)})
			return
		}

		body, err := readBoundedOutboundBody(resp.Body, fetchRawMaxSize)
		if err != nil {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "failed to read response"})
			return
		}

		if len(body) == 0 {
			w.WriteHeader(http.StatusBadGateway)
			_ = json.NewEncoder(w).Encode(FetchRawResponse{Error: "remote returned empty content"})
			return
		}

		log.Printf("[FetchRaw] Success, %d bytes from %s", len(body), redactURLForLog(targetURL))

		_ = json.NewEncoder(w).Encode(FetchRawResponse{Content: string(body)})
	}
}
