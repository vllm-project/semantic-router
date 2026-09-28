// Package connector provides the shared HTTP mechanics used by remote router
// model protocol adapters.
package connector

import (
	"bytes"
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Operation describes one static operation in a remote model protocol. An
// empty Path targets the configured base URL exactly. Query, when set, is the
// already-encoded query string without its leading "?". SuccessStatusCode
// narrows success to one 2xx status; zero accepts any 2xx.
type Operation struct {
	Name              string
	Method            string
	Path              string
	Query             string
	SuccessStatusCode int
	RetrySafe         bool
}

// Options defines bounded transport behavior. Byte limits and AttemptTimeout
// must be positive; MaxRetries is the number of attempts after the first one.
// TLSConfig, when set, replaces the default TLS client configuration so a
// caller can carry an explicit operator opt-in such as skipped verification.
type Options struct {
	AttemptTimeout   time.Duration
	MaxRetries       int
	MaxRequestBytes  int64
	MaxResponseBytes int64
	MaxErrorBytes    int64
	TLSConfig        *tls.Config
}

// Request carries one operation payload plus headers that belong to that
// call alone, such as trace context or a per-request identifier. Headers are
// applied after the connector defaults and before the authentication hook.
type Request struct {
	Body    []byte
	Headers map[string]string
}

// Result reports a successful operation together with the transport facts a
// caller may want to record about it.
type Result struct {
	Body       []byte
	StatusCode int
	Attempts   int
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
	if options.TLSConfig != nil {
		transport.TLSClientConfig = options.TLSConfig.Clone()
	}
	return &Client{
		baseURL:   parsed,
		authorize: authorize,
		options:   options,
		http: &http.Client{
			Transport:     transport,
			CheckRedirect: rejectRedirect,
		},
	}, nil
}

// rejectRedirect stops the HTTP client from following any redirect. The
// connector binds one configured origin; following a redirect would replay
// the request body and its credential headers to whatever origin the remote
// named, which is outside the trust boundary the caller configured. Returning
// ErrUseLastResponse hands the redirect response back to readResponse, where
// it is classified as KindRedirect without a second request being sent.
func rejectRedirect(*http.Request, []*http.Request) error {
	return http.ErrUseLastResponse
}

// isRedirectStatus reports the status codes for which the standard client
// would have issued a follow-up request.
func isRedirectStatus(statusCode int) bool {
	switch statusCode {
	case http.StatusMovedPermanently,
		http.StatusFound,
		http.StatusSeeOther,
		http.StatusTemporaryRedirect,
		http.StatusPermanentRedirect:
		return true
	default:
		return false
	}
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
	result, err := c.DoRequest(ctx, operation, Request{Body: body})
	if err != nil {
		return nil, err
	}
	return result.Body, nil
}

// connectorOutcome maps a call's result onto the metrics outcome label: the
// success constant, or the connector's own error kind, so a dashboard can tell
// a down remote from an overloaded or misconfigured one.
//
// Every failure path returns an *Error, so the fallback is unreachable today;
// it stays because a metrics label must never be the reason a call panics, and
// it is named in the metric's help so an operator who does see it knows it
// means "an error this code did not classify".
func connectorOutcome(err error) string {
	if err == nil {
		return metrics.RemoteConnectorOutcomeSuccess
	}
	var connectorErr *Error
	if errors.As(err, &connectorErr) {
		return string(connectorErr.Kind)
	}
	return metrics.RemoteConnectorOutcomeUnclassified
}

// DoRequest invokes an operation with per-call headers and returns the bounded
// successful response together with its status code and attempt count.
//
// Every call is recorded once, however it ended, with the wall time the caller
// actually waited - retries included - so a remote classifier's health is
// visible to whoever operates it rather than only to the signal that failed.
// The measurement sits here rather than on Do so a caller that needs
// per-request headers is measured too.
func (c *Client) DoRequest(ctx context.Context, operation Operation, request Request) (Result, error) {
	start := time.Now()
	result, err := c.doRequest(ctx, operation, request)
	metrics.RecordRemoteConnectorRequest(operation.Name, connectorOutcome(err), time.Since(start).Seconds())
	return result, err
}

func (c *Client) doRequest(ctx context.Context, operation Operation, request Request) (Result, error) {
	if err := validateOperation(operation); err != nil {
		return Result{}, &Error{Kind: KindRequest, Operation: operation.Name, Cause: err}
	}
	if int64(len(request.Body)) > c.options.MaxRequestBytes {
		return Result{}, &Error{
			Kind:      KindRequest,
			Operation: operation.Name,
			Cause: fmt.Errorf(
				"request body is %d bytes, exceeding the limit of %d bytes",
				len(request.Body), c.options.MaxRequestBytes,
			),
		}
	}

	for attempt := 1; ; attempt++ {
		result, connectorErr := c.doAttempt(ctx, operation, request, attempt)
		if connectorErr == nil {
			result.Attempts = attempt
			return result, nil
		}
		if !connectorErr.Retryable || attempt > c.options.MaxRetries {
			return Result{}, connectorErr
		}
		if err := waitBeforeRetry(ctx, attempt); err != nil {
			return Result{}, &Error{
				Kind:      KindTransport,
				Operation: operation.Name,
				Attempt:   attempt,
				Cause:     err,
			}
		}
		// Counted only once the wait completed: a context cancelled during the
		// backoff means the retry never happens, and counting it would put the
		// retry rate above the real attempt count - the very ratio a dashboard
		// reads as "flapping under the retry budget".
		metrics.RecordRemoteConnectorRetry(operation.Name)
	}
}

func validateOperation(operation Operation) error {
	if strings.TrimSpace(operation.Name) == "" {
		return fmt.Errorf("operation name is required")
	}
	if strings.TrimSpace(operation.Method) == "" {
		return fmt.Errorf("operation method is required")
	}
	if operation.Path != "" &&
		(!strings.HasPrefix(operation.Path, "/") || strings.ContainsAny(operation.Path, "?#")) {
		return fmt.Errorf("operation path must be an absolute path without query or fragment")
	}
	if operation.SuccessStatusCode != 0 &&
		(operation.SuccessStatusCode < http.StatusOK || operation.SuccessStatusCode >= http.StatusMultipleChoices) {
		return fmt.Errorf("operation success status code must be between 200 and 299")
	}
	if strings.ContainsAny(operation.Query, "?#") {
		return fmt.Errorf("operation query must not contain a fragment or a second query marker")
	}
	if _, err := url.ParseQuery(operation.Query); err != nil {
		return fmt.Errorf("operation query is not a valid query string: %w", err)
	}
	return nil
}

func (c *Client) doAttempt(
	ctx context.Context,
	operation Operation,
	request Request,
	attempt int,
) (Result, *Error) {
	if ctx == nil {
		return Result{}, &Error{Kind: KindRequest, Operation: operation.Name, Attempt: attempt, Cause: fmt.Errorf("context is nil")}
	}
	attemptCtx, cancel := context.WithTimeout(ctx, c.options.AttemptTimeout)
	defer cancel()

	httpRequest, connectorErr := c.newRequest(attemptCtx, operation, request, attempt)
	if connectorErr != nil {
		return Result{}, connectorErr
	}
	response, err := c.http.Do(httpRequest)
	if err != nil {
		return Result{}, &Error{
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
	request Request,
	attempt int,
) (*http.Request, *Error) {
	target := *c.baseURL
	if operation.Path != "" {
		target.Path = path.Join(c.baseURL.Path, operation.Path)
		target.RawPath = ""
	}
	target.RawQuery = operation.Query
	httpRequest, err := http.NewRequestWithContext(ctx, operation.Method, target.String(), bytes.NewReader(request.Body))
	if err != nil {
		return nil, &Error{Kind: KindRequest, Operation: operation.Name, Attempt: attempt, Cause: err}
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "application/json")
	for key, value := range request.Headers {
		httpRequest.Header.Set(key, value)
	}
	if c.authorize != nil {
		if err := c.authorize(ctx, httpRequest); err != nil {
			return nil, &Error{Kind: KindAuthorization, Operation: operation.Name, Attempt: attempt, Cause: err}
		}
	}
	return httpRequest, nil
}

func (c *Client) readResponse(
	ctx context.Context,
	operation Operation,
	response *http.Response,
	attempt int,
) (Result, *Error) {
	if isRedirectStatus(response.StatusCode) {
		return Result{}, &Error{
			Kind:       KindRedirect,
			Operation:  operation.Name,
			StatusCode: response.StatusCode,
			Attempt:    attempt,
			Cause:      fmt.Errorf("%w: %s", ErrRedirectRejected, redirectTarget(response)),
		}
	}
	if !operationAcceptsStatus(operation, response.StatusCode) {
		errorBody, truncated, readErr := readBounded(response.Body, c.options.MaxErrorBytes)
		if readErr != nil {
			return Result{}, &Error{
				Kind:      KindResponse,
				Operation: operation.Name,
				Attempt:   attempt,
				Retryable: operation.RetrySafe && ctx.Err() == nil,
				Cause:     readErr,
			}
		}
		return Result{}, &Error{
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
		return Result{}, &Error{
			Kind:      KindResponse,
			Operation: operation.Name,
			Attempt:   attempt,
			Retryable: operation.RetrySafe && ctx.Err() == nil,
			Cause:     err,
		}
	}
	if exceeded {
		return Result{}, &Error{
			Kind:      KindResponse,
			Operation: operation.Name,
			Attempt:   attempt,
			Cause:     fmt.Errorf("%w of %d bytes", ErrResponseTooLarge, c.options.MaxResponseBytes),
		}
	}
	return Result{Body: responseBody, StatusCode: response.StatusCode}, nil
}

// redirectTarget names where a rejected redirect pointed, reduced to its
// scheme and host so the diagnostic never carries a path or query.
func redirectTarget(response *http.Response) string {
	location := strings.TrimSpace(response.Header.Get("Location"))
	if location == "" {
		return "no location"
	}
	parsed, err := url.Parse(location)
	if err != nil || parsed.Host == "" {
		return "relative location"
	}
	return parsed.Scheme + "://" + parsed.Host
}

func operationAcceptsStatus(operation Operation, statusCode int) bool {
	if operation.SuccessStatusCode != 0 {
		return statusCode == operation.SuccessStatusCode
	}
	return statusCode >= http.StatusOK && statusCode < http.StatusMultipleChoices
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
