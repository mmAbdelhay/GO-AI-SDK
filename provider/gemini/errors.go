package gemini

import (
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// parseError converts a non-200 Gemini response into a normalized *ai.APIError.
// Gemini errors carry a gRPC-style status string alongside the HTTP code.
func parseError(resp *http.Response, body []byte) error {
	var we gErrorResponse
	_ = json.Unmarshal(body, &we) // best effort

	sentinel := gSentinel(resp.StatusCode, we.Error.Status)
	apiErr := ai.NewAPIError("gemini", resp.StatusCode, sentinel)
	apiErr.Type = we.Error.Status
	apiErr.Message = we.Error.Message
	apiErr.Raw = body
	if ra := resp.Header.Get("Retry-After"); ra != "" {
		if secs, err := strconv.Atoi(ra); err == nil {
			apiErr.RetryAfter = time.Duration(secs) * time.Second
		}
	}
	return apiErr
}

func gSentinel(status int, grpcStatus string) error {
	switch grpcStatus {
	case "RESOURCE_EXHAUSTED":
		return ai.ErrRateLimited
	case "UNAUTHENTICATED", "PERMISSION_DENIED":
		return ai.ErrAuthentication
	case "NOT_FOUND":
		return ai.ErrNotFound
	case "INVALID_ARGUMENT", "FAILED_PRECONDITION":
		return ai.ErrInvalidRequest
	case "UNAVAILABLE", "INTERNAL", "DEADLINE_EXCEEDED":
		return ai.ErrProviderUnavailable
	}
	switch status {
	case http.StatusTooManyRequests:
		return ai.ErrRateLimited
	case http.StatusBadRequest:
		return ai.ErrInvalidRequest
	case http.StatusUnauthorized, http.StatusForbidden:
		return ai.ErrAuthentication
	case http.StatusNotFound:
		return ai.ErrNotFound
	}
	if status >= 500 {
		return ai.ErrProviderUnavailable
	}
	return ai.ErrInvalidRequest
}
