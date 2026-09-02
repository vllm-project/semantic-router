// Package connector provides the shared HTTP mechanics used by remote router
// model protocol adapters.
package connector

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"
)

// Operation describes one static operation in a remote model protocol.
type Operation struct {
	Name      string
	Method    string
	Path      string
	RetrySafe bool
}

// Options defines bounded transport behavior. Byte limits and AttemptTimeout
// must be positive; MaxRetries is the number of attempts after the first one.
type Options struct {
	AttemptTimeout   time.Duration
	MaxRetries       int
	MaxRequestBytes  int64
	MaxResponseBytes int64
	MaxErrorBytes    int64
}

// Client binds one deployment endpoint, its authentication hook, and bounded
// transport policy. Protocol payloads remain owned by callers.
type Client struct {
	baseURL   *url.URL
	authorize func(context.Context, *http.Request) error
	options   Options
	http      *http.Client
}

func New(
	baseURL string,
	authorize func(context.Context, *http.Request) error,
	options Options,
) (*Client, error) {
	parsed, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return nil, fmt.Errorf("parse connector base URL: %w", err)
	}
	if (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.Host == "" {
		return nil, fmt.Errorf("connector base URL must be an absolute HTTP(S) URL")
	}
	if parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return nil, fmt.Errorf("connector base URL must not contain user info, query, or fragment")
	}
	if err := validateOptions(options); err != nil {
		return nil, err
	}

	transport := http.DefaultTransport.(*http.Transport).Clone()
	return &Client{
		baseURL:   parsed,
		authorize: authorize,
		options:   options,
		http:      &http.Client{Transport: transport},
	}, nil
}

func validateOptions(options Options) error {
	if options.AttemptTimeout <= 0 {
		return fmt.Errorf("connector attempt timeout must be positive")
	}
	if options.MaxRetries < 0 {
		return fmt.Errorf("connector max retries must not be negative")
	}
	if options.MaxRequestBytes <= 0 || options.MaxResponseBytes <= 0 || options.MaxErrorBytes <= 0 {
		return fmt.Errorf("connector body limits must be positive")
	}
	return nil
}

// Do invokes an operation and returns its bounded successful response body.
func (c *Client) Do(ctx context.Context, operation Operation, body []byte) ([]byte, error) {
	if err := validateOperation(operation); err != nil {
		return nil, &Error{Kind: KindRequest, Operation: operation.Name, Cause: err}
	}
	if int64(len(body)) > c.options.MaxRequestBytes {
		return nil, &Error{
			Kind:      KindRequest,
			Operation: operation.Name,
			Cause: fmt.Errorf(
				"request body is %d bytes, exceeding the limit of %d bytes",
				len(body), c.options.MaxRequestBytes,
			),
		}
	}

	for attempt := 1; ; attempt++ {
		responseBody, connectorErr := c.doAttempt(ctx, operation, body, attempt)
		if connectorErr == nil {
			return responseBody, nil
		}
		if !connectorErr.Retryable || attempt > c.options.MaxRetries {
			return nil, connectorErr
		}
		if err := waitBeforeRetry(ctx, attempt); err != nil {
			return nil, &Error{
				Kind:      KindTransport,
				Operation: operation.Name,
				Attempt:   attempt,
				Cause:     err,
			}
		}
	}
}

func validateOperation(operation Operation) error {
	if strings.TrimSpace(operation.Name) == "" {
		return fmt.Errorf("operation name is required")
	}
	if strings.TrimSpace(operation.Method) == "" {
		return fmt.Errorf("operation method is required")
	}
	if !strings.HasPrefix(operation.Path, "/") || strings.ContainsAny(operation.Path, "?#") {
		return fmt.Errorf("operation path must be an absolute path without query or fragment")
	}
	return nil
}

func (c *Client) doAttempt(
	ctx context.Context,
	operation Operation,
	body []byte,
	attempt int,
) ([]byte, *Error) {
	if ctx == nil {
		return nil, &Error{Kind: KindRequest, Operation: operation.Name, Attempt: attempt, Cause: fmt.Errorf("context is nil")}
	}
	attemptCtx, cancel := context.WithTimeout(ctx, c.options.AttemptTimeout)
	defer cancel()

	request, connectorErr := c.newRequest(attemptCtx, operation, body, attempt)
	if connectorErr != nil {
		return nil, connectorErr
	}
	response, err := c.http.Do(request)
	if err != nil {
		return nil, &Error{
			Kind:      KindTransport,
			Operation: operation.Name,
			Attempt:   attempt,
			Retryable: operation.RetrySafe && ctx.Err() == nil && retryableTransportError(err),
			Cause:     err,
		}
	}
	defer response.Body.Close()
	return c.readResponse(ctx, operation, response, attempt)
}

func (c *Client) newRequest(
	ctx context.Context,
	operation Operation,
	body []byte,
	attempt int,
) (*http.Request, *Error) {
	target := *c.baseURL
	target.Path = path.Join(c.baseURL.Path, operation.Path)
	target.RawPath = ""
	request, err := http.NewRequestWithContext(ctx, operation.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		return nil, &Error{Kind: KindRequest, Operation: operation.Name, Attempt: attempt, Cause: err}
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Accept", "application/json")
	if c.authorize != nil {
		if err := c.authorize(ctx, request); err != nil {
			return nil, &Error{Kind: KindAuthorization, Operation: operation.Name, Attempt: attempt, Cause: err}
		}
	}
	return request, nil
}

func (c *Client) readResponse(
	ctx context.Context,
	operation Operation,
	response *http.Response,
	attempt int,
) ([]byte, *Error) {
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		errorBody, truncated, readErr := readBounded(response.Body, c.options.MaxErrorBytes)
		if readErr != nil {
			return nil, &Error{
				Kind:      KindResponse,
				Operation: operation.Name,
				Attempt:   attempt,
				Retryable: operation.RetrySafe && ctx.Err() == nil,
				Cause:     readErr,
			}
		}
		return nil, &Error{
			Kind:       KindStatus,
			Operation:  operation.Name,
			StatusCode: response.StatusCode,
			Attempt:    attempt,
			Retryable:  operation.RetrySafe && retryableStatus(response.StatusCode),
			body:       errorBody,
			truncated:  truncated,
		}
	}

	responseBody, exceeded, err := readBounded(response.Body, c.options.MaxResponseBytes)
	if err != nil {
		return nil, &Error{
			Kind:      KindResponse,
			Operation: operation.Name,
			Attempt:   attempt,
			Retryable: operation.RetrySafe && ctx.Err() == nil,
			Cause:     err,
		}
	}
	if exceeded {
		return nil, &Error{
			Kind:      KindResponse,
			Operation: operation.Name,
			Attempt:   attempt,
			Cause: fmt.Errorf(
				"response body exceeds limit of %d bytes",
				c.options.MaxResponseBytes,
			),
		}
	}
	return responseBody, nil
}

func readBounded(reader io.Reader, limit int64) ([]byte, bool, error) {
	body, err := io.ReadAll(io.LimitReader(reader, limit+1))
	if err != nil {
		return nil, false, err
	}
	if int64(len(body)) <= limit {
		return body, false, nil
	}
	return body[:limit], true, nil
}

// Close releases idle connections owned by this client. In-flight requests
// remain governed by their contexts.
func (c *Client) Close() error {
	if c != nil && c.http != nil {
		c.http.CloseIdleConnections()
	}
	return nil
}
