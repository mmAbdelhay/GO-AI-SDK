package anthropic

import (
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

const providerName = "anthropic"

// parseError converts a non-200 Anthropic HTTP response into a normalized
// *ai.APIError whose Unwrap returns the matching package sentinel, so callers can
// use errors.Is uniformly across providers.
func parseError(resp *http.Response, body []byte) error {
	var we wireErrorResponse
	_ = json.Unmarshal(body, &we) // best effort; body may not be JSON

	sentinel := sentinelFor(resp.StatusCode, we.Error.Type, we.Error.Message)

	apiErr := ai.NewAPIError(providerName, resp.StatusCode, sentinel)
	apiErr.Type = we.Error.Type
	apiErr.Message = we.Error.Message
	apiErr.Raw = body
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			apiErr.RetryAfter = time.Duration(secs) * time.Second
		}
	}
	return apiErr
}

// sentinelFor maps an HTTP status and Anthropic error type/message onto a
// normalized sentinel error.
func sentinelFor(status int, errType, message string) error {
	// A 400 that is really a context-length problem maps to the more specific
	// sentinel so callers can trim history and retry.
	if status == http.StatusBadRequest && isContextLength(errType, message) {
		return ai.ErrContextLengthExceeded
	}
	switch status {
	case http.StatusTooManyRequests:
		return ai.ErrRateLimited
	case http.StatusBadRequest, http.StatusUnprocessableEntity:
		return ai.ErrInvalidRequest
	case http.StatusUnauthorized, http.StatusForbidden:
		return ai.ErrAuthentication
	case http.StatusNotFound:
		return ai.ErrNotFound
	case 529: // Anthropic-specific "overloaded" status.
		return ai.ErrOverloaded
	}
	if status >= 500 {
		return ai.ErrProviderUnavailable
	}
	return ai.ErrInvalidRequest
}

// sentinelForType maps an Anthropic error type string (as it appears in error
// response bodies and streaming error events) onto a normalized sentinel. It is
// used when no HTTP status is available, such as errors delivered mid-stream.
func sentinelForType(errType, message string) error {
	switch errType {
	case "rate_limit_error":
		return ai.ErrRateLimited
	case "authentication_error", "permission_error":
		return ai.ErrAuthentication
	case "not_found_error":
		return ai.ErrNotFound
	case "overloaded_error":
		return ai.ErrOverloaded
	case "api_error":
		return ai.ErrProviderUnavailable
	case "invalid_request_error":
		if isContextLength(errType, message) {
			return ai.ErrContextLengthExceeded
		}
		return ai.ErrInvalidRequest
	default:
		return ai.ErrProviderUnavailable
	}
}

func isContextLength(errType, message string) bool {
	m := strings.ToLower(errType + " " + message)
	return strings.Contains(m, "context") ||
		strings.Contains(m, "too long") ||
		strings.Contains(m, "maximum") && strings.Contains(m, "token")
}
