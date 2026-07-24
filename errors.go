package ai

import (
	"errors"
	"fmt"
	"time"
)

// Error taxonomy. Providers normalize their wire errors onto these sentinels so
// callers can detect conditions uniformly with errors.Is, regardless of which
// provider produced the error:
//
//	if errors.Is(err, ai.ErrRateLimited) { ... }
//
// For provider-specific detail (status code, retry-after, raw body), unwrap an
// *APIError with errors.As.
var (
	// ErrRateLimited indicates the provider rejected the request due to rate
	// limiting (typically HTTP 429).
	ErrRateLimited = errors.New("ai: rate limited")
	// ErrContextLengthExceeded indicates the prompt plus requested output
	// exceeds the model's context window.
	ErrContextLengthExceeded = errors.New("ai: context length exceeded")
	// ErrProviderUnavailable indicates a server-side failure (typically HTTP
	// 5xx) where the request may succeed on retry.
	ErrProviderUnavailable = errors.New("ai: provider unavailable")
	// ErrOverloaded indicates the provider is temporarily overloaded
	// (Anthropic HTTP 529).
	ErrOverloaded = errors.New("ai: provider overloaded")
	// ErrInvalidRequest indicates the request was malformed or invalid
	// (typically HTTP 400).
	ErrInvalidRequest = errors.New("ai: invalid request")
	// ErrAuthentication indicates missing or invalid credentials (typically
	// HTTP 401 or 403).
	ErrAuthentication = errors.New("ai: authentication failed")
	// ErrNotFound indicates the requested model or resource does not exist
	// (typically HTTP 404).
	ErrNotFound = errors.New("ai: not found")
)

// APIError is a normalized provider error. It wraps one of the package sentinels
// (accessible via Unwrap and therefore errors.Is) while retaining provider
// detail for callers that need it via errors.As.
type APIError struct {
	// Provider names the provider that produced the error, e.g. "anthropic".
	Provider string
	// StatusCode is the HTTP status code, or 0 if the error occurred before a
	// response was received.
	StatusCode int
	// Type is the provider's own error type string, when available.
	Type string
	// Message is the provider's human-readable error message.
	Message string
	// RetryAfter is the suggested wait before retrying, parsed from the
	// Retry-After header when present.
	RetryAfter time.Duration
	// Raw is the raw error response body.
	Raw []byte

	// sentinel is the normalized error this APIError unwraps to.
	sentinel error
}

// NewAPIError constructs an APIError that unwraps to the given sentinel. Provider
// packages use it to normalize wire errors. If sentinel is nil the APIError
// unwraps to nothing.
func NewAPIError(provider string, status int, sentinel error) *APIError {
	return &APIError{Provider: provider, StatusCode: status, sentinel: sentinel}
}

// Error implements the error interface.
func (e *APIError) Error() string {
	kind := "error"
	if e.sentinel != nil {
		kind = e.sentinel.Error()
	}
	if e.Message != "" {
		return fmt.Sprintf("%s (%s, status %d): %s", kind, e.Provider, e.StatusCode, e.Message)
	}
	return fmt.Sprintf("%s (%s, status %d)", kind, e.Provider, e.StatusCode)
}

// Unwrap returns the normalized sentinel so errors.Is(err, ai.ErrRateLimited)
// and friends work across providers.
func (e *APIError) Unwrap() error { return e.sentinel }
