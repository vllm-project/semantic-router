package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

type openWebFetchedContent struct {
	url     string
	title   string
	content string
}

type jinaReaderResponse struct {
	Data struct {
		URL     string `json:"url"`
		Title   string `json:"title"`
		Content string `json:"content"`
	} `json:"data"`
	URL     string `json:"url"`
	Title   string `json:"title"`
	Content string `json:"content"`
}

func fetchWebWithJina(targetURL string, timeout time.Duration, outputFormat string, maxLength int, withImages bool) (*OpenWebResponse, error) {
	log.Printf("[OpenWeb:Jina] Starting fetch: %s", targetURL)
	startTime := time.Now()

	req, err := newJinaRequest(targetURL, timeout, outputFormat, withImages)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := (&http.Client{Timeout: timeout}).Do(req)
	if err != nil {
		return nil, wrapOpenWebRequestError(err)
	}
	defer resp.Body.Close()

	log.Printf("[OpenWeb:Jina] Response status: %d, elapsed: %v", resp.StatusCode, time.Since(startTime))

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(body))
	}

	fetchedContent, err := parseJinaResponse(resp.Body, outputFormat, targetURL)
	if err != nil {
		return nil, err
	}

	log.Printf("[OpenWeb:Jina] Got title: %s", fetchedContent.title)
	log.Printf("[OpenWeb:Jina] Original content length: %d characters", len(fetchedContent.content))

	result := buildOpenWebResponse(fetchedContent, maxLength, "jina")
	if result.Truncated {
		log.Printf("[OpenWeb:Jina] Content truncated to %d characters", maxLength)
	}

	log.Printf("[OpenWeb:Jina] ✅ Fetch succeeded, total elapsed: %v", time.Since(startTime))
	return result, nil
}

func newJinaRequest(targetURL string, timeout time.Duration, outputFormat string, withImages bool) (*http.Request, error) {
	jinaURL := fmt.Sprintf("%s/%s", jinaReaderBaseURL, targetURL)
	log.Printf("[OpenWeb:Jina] Jina URL: %s", jinaURL)

	req, err := http.NewRequest("GET", jinaURL, nil)
	if err != nil {
		return nil, err
	}

	if outputFormat == "json" {
		req.Header.Set("Accept", "application/json")
	}
	req.Header.Set("X-Timeout", fmt.Sprintf("%d", int(timeout.Seconds())))
	req.Header.Set("X-No-Cache", "true")
	if withImages {
		req.Header.Set("X-With-Generated-Alt", "true")
	}

	return req, nil
}

func parseJinaResponse(body io.Reader, outputFormat string, targetURL string) (openWebFetchedContent, error) {
	if outputFormat == "json" {
		return parseJinaJSONResponse(body, targetURL)
	}
	return parseJinaMarkdownResponse(body, targetURL)
}

func parseJinaJSONResponse(body io.Reader, targetURL string) (openWebFetchedContent, error) {
	var result jinaReaderResponse
	if err := json.NewDecoder(body).Decode(&result); err != nil {
		return openWebFetchedContent{}, fmt.Errorf("failed to parse response: %w", err)
	}

	content := result.Data.Content
	if content == "" {
		content = result.Content
	}
	if content == "" {
		return openWebFetchedContent{}, fmt.Errorf("response content is empty")
	}

	title := result.Data.Title
	if title == "" {
		title = result.Title
	}

	actualURL := targetURL
	if result.Data.URL != "" {
		actualURL = result.Data.URL
	} else if result.URL != "" {
		actualURL = result.URL
	}

	return openWebFetchedContent{
		url:     actualURL,
		title:   title,
		content: content,
	}, nil
}

func parseJinaMarkdownResponse(body io.Reader, targetURL string) (openWebFetchedContent, error) {
	contentBytes, err := io.ReadAll(body)
	if err != nil {
		return openWebFetchedContent{}, fmt.Errorf("failed to read response: %w", err)
	}

	content := string(contentBytes)
	if content == "" {
		return openWebFetchedContent{}, fmt.Errorf("response content is empty")
	}

	return openWebFetchedContent{
		url:     targetURL,
		title:   extractMarkdownTitle(content),
		content: content,
	}, nil
}

func buildOpenWebResponse(content openWebFetchedContent, maxLength int, method string) *OpenWebResponse {
	title := content.title
	if title == "" {
		title = "Untitled"
	}

	truncatedContent, truncated := truncateOpenWebContent(content.content, maxLength)
	return &OpenWebResponse{
		URL:       content.url,
		Title:     title,
		Content:   truncatedContent,
		Length:    len(truncatedContent),
		Truncated: truncated,
		Method:    method,
	}
}

func wrapOpenWebRequestError(err error) error {
	if err == nil {
		return nil
	}
	if containsOpenWebTimeout(err) {
		return fmt.Errorf("request timeout")
	}
	return fmt.Errorf("request failed: %w", err)
}

func containsOpenWebTimeout(err error) bool {
	return err != nil && strings.Contains(err.Error(), "timeout")
}
