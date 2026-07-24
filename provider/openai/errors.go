package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// parseError converts a non-200 response into a normalized *ai.APIError.
func (c *Client) parseError(resp *http.Response, body []byte) error {
	var we oaErrorResponse
	_ = json.Unmarshal(body, &we) // best effort

	code := fmt.Sprint(we.Error.Code)
	sentinel := oaSentinel(resp.StatusCode, code, we.Error.Message)

	apiErr := ai.NewAPIError(c.provider, resp.StatusCode, sentinel)
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

func oaSentinel(status int, code, message string) error {
	m := strings.ToLower(code + " " + message)
	if strings.Contains(m, "context_length") || strings.Contains(m, "context length") ||
		strings.Contains(m, "maximum context") {
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
	}
	if status >= 500 {
		return ai.ErrProviderUnavailable
	}
	return ai.ErrInvalidRequest
}
