package routerauth

import (
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	defaultManagementRetryAfter = time.Second
	maximumManagementRetryAfter = 10 * time.Minute
)

var (
	ErrManagementSessionReauthentication = errors.New("router Management session requires reauthentication")
	ErrManagementSessionRateLimited      = errors.New("router Management session exchange is rate limited")
)

// ManagementSessionError is the bounded failure contract between Router
// session exchange and the Dashboard BFF. It carries no upstream body or
// assertion data.
type ManagementSessionError struct {
	Status     int
	RetryAfter time.Duration
	RequestID  string
}

func (err *ManagementSessionError) Error() string {
	if err == nil {
		return ErrManagementSessionUnavailable.Error()
	}
	switch err.Status {
	case http.StatusUnauthorized:
		return ErrManagementSessionReauthentication.Error()
	case http.StatusTooManyRequests:
		return ErrManagementSessionRateLimited.Error()
	default:
		return ErrManagementSessionUnavailable.Error()
	}
}

func (err *ManagementSessionError) Unwrap() []error {
	causes := []error{ErrManagementSessionUnavailable}
	if err == nil {
		return causes
	}
	switch err.Status {
	case http.StatusUnauthorized:
		causes = append(causes, ErrManagementSessionReauthentication)
	case http.StatusTooManyRequests:
		causes = append(causes, ErrManagementSessionRateLimited)
	}
	return causes
}

func (err *ManagementSessionError) RetryAfterHeader() string {
	if err == nil || err.Status != http.StatusTooManyRequests {
		return ""
	}
	return managementRetryAfterHeader(err.RetryAfter)
}

func (err *ManagementSessionError) HTTPStatus() int {
	if err == nil {
		return http.StatusServiceUnavailable
	}
	switch err.Status {
	case http.StatusUnauthorized, http.StatusTooManyRequests, http.StatusServiceUnavailable:
		return err.Status
	default:
		return http.StatusServiceUnavailable
	}
}

type routerManagementResponseError struct {
	Status     int
	RetryAfter time.Duration
	RequestID  string
}

func (err *routerManagementResponseError) Error() string {
	if err == nil {
		return ErrManagementSessionUnavailable.Error()
	}
	return fmt.Sprintf("router Management request returned HTTP %d", err.Status)
}

func classifyManagementSessionError(err error) *ManagementSessionError {
	var classified *ManagementSessionError
	if errors.As(err, &classified) {
		return classified
	}
	var upstream *routerManagementResponseError
	if errors.As(err, &upstream) {
		switch upstream.Status {
		case http.StatusUnauthorized:
			return &ManagementSessionError{Status: http.StatusUnauthorized, RequestID: upstream.RequestID}
		case http.StatusTooManyRequests:
			return &ManagementSessionError{
				Status: http.StatusTooManyRequests, RetryAfter: boundedManagementRetryAfter(upstream.RetryAfter),
				RequestID: upstream.RequestID,
			}
		}
	}
	return &ManagementSessionError{Status: http.StatusServiceUnavailable}
}

func boundedManagementRetryAfter(value time.Duration) time.Duration {
	if value <= 0 {
		return defaultManagementRetryAfter
	}
	if value > maximumManagementRetryAfter {
		return maximumManagementRetryAfter
	}
	return value
}

func parseManagementRetryAfter(value string, now time.Time) time.Duration {
	value = strings.TrimSpace(value)
	if seconds, err := strconv.ParseInt(value, 10, 32); err == nil && seconds > 0 {
		return boundedManagementRetryAfter(time.Duration(seconds) * time.Second)
	}
	if retryAt, err := http.ParseTime(value); err == nil && retryAt.After(now) {
		return boundedManagementRetryAfter(retryAt.Sub(now))
	}
	return defaultManagementRetryAfter
}

func managementRetryAfterHeader(value time.Duration) string {
	seconds := int64((boundedManagementRetryAfter(value) + time.Second - 1) / time.Second)
	return strconv.FormatInt(seconds, 10)
}

func invitationAuthorityFromManagementError(err error) error {
	var upstream *routerManagementResponseError
	if errors.As(err, &upstream) {
		code := ""
		switch upstream.Status {
		case http.StatusUnauthorized:
			code = "unauthenticated"
		case http.StatusTooManyRequests:
			code = "challenge_capacity_exceeded"
		case http.StatusServiceUnavailable:
			code = "authentication_unavailable"
		}
		return newInvitationAuthorityError(upstream.Status, code, upstream.RequestID, upstream.RetryAfter)
	}
	var session *ManagementSessionError
	if errors.As(err, &session) {
		code := ""
		switch session.Status {
		case http.StatusUnauthorized:
			code = "unauthenticated"
		case http.StatusTooManyRequests:
			code = "challenge_capacity_exceeded"
		case http.StatusServiceUnavailable:
			code = "authentication_unavailable"
		}
		return newInvitationAuthorityError(session.Status, code, session.RequestID, session.RetryAfter)
	}
	return nil
}
