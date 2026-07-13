package handlers

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"
)

type openWebFetchPlan struct {
	request   OpenWebRequest
	timeout   time.Duration
	format    string
	maxLength int
	forceJina bool
}

func OpenWebHandler() http.HandlerFunc {
	return openWebHandlerWithClient(newPublicOutboundHTTPClient(openWebMaxTimeout))
}

func openWebHandlerWithClient(client outboundHTTPClient) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		setOpenWebCORSHeaders(w)

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		req, status, err := decodeOpenWebRequest(w, r)
		if err != nil {
			log.Printf("[OpenWeb] Failed to parse request")
			writeOpenWebJSON(w, status, OpenWebResponse{Error: "Invalid request format"})
			return
		}

		if invalidResponse, ok := validateOpenWebRequest(r.Context(), client, &req); ok {
			writeOpenWebJSON(w, http.StatusBadRequest, invalidResponse)
			return
		}

		plan := buildOpenWebFetchPlan(req)
		logOpenWebFetchPlan(plan)

		result, fetchErr := fetchOpenWeb(r.Context(), client, plan)
		if fetchErr != nil {
			log.Printf("[OpenWeb] All fetch methods failed for %s", redactURLForLog(req.URL))
			writeOpenWebJSON(w, http.StatusBadGateway, OpenWebResponse{
				URL:   redactURLForLog(req.URL),
				Error: "Unable to fetch web content",
			})
			return
		}

		writeOpenWebJSON(w, http.StatusOK, *result)
	}
}

func setOpenWebCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
}

func decodeOpenWebRequest(w http.ResponseWriter, r *http.Request) (OpenWebRequest, int, error) {
	var req OpenWebRequest
	if status, err := decodeBoundedJSON(w, r, outboundMaxRequestBodyBytes, &req); err != nil {
		return OpenWebRequest{}, status, err
	}
	return req, 0, nil
}

func validateOpenWebRequest(
	ctx context.Context,
	client outboundHTTPClient,
	req *OpenWebRequest,
) (OpenWebResponse, bool) {
	if req == nil || req.URL == "" {
		return OpenWebResponse{Error: "URL cannot be empty"}, true
	}

	parsedURL, err := parseOutboundHTTPURL(req.URL)
	if err != nil {
		return OpenWebResponse{Error: "Invalid or non-public URL"}, true
	}
	req.URL = parsedURL.String()
	if err := client.ValidateURL(ctx, req.URL); err != nil {
		return OpenWebResponse{Error: "Invalid or non-public URL"}, true
	}

	return OpenWebResponse{}, false
}

func buildOpenWebFetchPlan(req OpenWebRequest) openWebFetchPlan {
	timeout := openWebDefaultTimeout
	if req.Timeout > 0 {
		timeout = time.Duration(req.Timeout) * time.Second
		if timeout > openWebMaxTimeout {
			timeout = openWebMaxTimeout
		}
	}

	return openWebFetchPlan{
		request:   req,
		timeout:   timeout,
		format:    normalizeOpenWebFormat(req.Format),
		maxLength: normalizeOpenWebMaxLength(req.MaxLength),
		forceJina: shouldPreferJinaFetch(req.URL, req),
	}
}

func logOpenWebFetchPlan(plan openWebFetchPlan) {
	log.Printf(
		"[OpenWeb] Request: url=%s, timeout=%v, force_jina=%v, format=%s, max_length=%d, with_images=%v",
		redactURLForLog(plan.request.URL),
		plan.timeout,
		plan.forceJina,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
}

func fetchOpenWeb(
	ctx context.Context,
	client outboundHTTPClient,
	plan openWebFetchPlan,
) (*OpenWebResponse, error) {
	if !plan.forceJina {
		log.Printf("[OpenWeb] Strategy 1: Trying direct fetch...")
		result, err := fetchWebDirect(ctx, client, plan.request.URL, plan.timeout, plan.maxLength)
		if err == nil {
			log.Printf("[OpenWeb] Direct fetch succeeded")
			return result, nil
		}
		log.Printf("[OpenWeb] Direct fetch failed")
		log.Printf("[OpenWeb] Strategy 2: Falling back to Jina Reader...")
	} else {
		log.Printf("[OpenWeb] Skipping direct fetch, using Jina Reader directly")
	}

	result, err := fetchWebWithJina(
		ctx,
		client,
		plan.request.URL,
		plan.timeout,
		plan.format,
		plan.maxLength,
		plan.request.WithImages,
	)
	if err != nil {
		return nil, err
	}

	log.Printf("[OpenWeb] Jina Reader fetch succeeded")
	return result, nil
}

func writeOpenWebJSON(w http.ResponseWriter, status int, response OpenWebResponse) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(response)
}
