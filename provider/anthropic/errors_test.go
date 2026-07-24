package anthropic

import (
	"errors"
	"net/http"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestParseErrorMapping(t *testing.T) {
	tests := []struct {
		name    string
		status  int
		errType string
		message string
		want    error
	}{
		{"rate limit", 429, "rate_limit_error", "slow down", ai.ErrRateLimited},
		{"auth", 401, "authentication_error", "bad key", ai.ErrAuthentication},
		{"forbidden", 403, "permission_error", "nope", ai.ErrAuthentication},
		{"not found", 404, "not_found_error", "no model", ai.ErrNotFound},
		{"overloaded", 529, "overloaded_error", "busy", ai.ErrOverloaded},
		{"server", 500, "api_error", "oops", ai.ErrProviderUnavailable},
		{"bad request", 400, "invalid_request_error", "bad", ai.ErrInvalidRequest},
		{"context length", 400, "invalid_request_error", "prompt is too long: maximum 200000 tokens", ai.ErrContextLengthExceeded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := `{"type":"error","error":{"type":"` + tt.errType + `","message":"` + tt.message + `"}}`
			resp := &http.Response{StatusCode: tt.status, Header: http.Header{}}
			err := parseError(resp, []byte(body))
			if !errors.Is(err, tt.want) {
				t.Errorf("errors.Is(err, %v) = false; err = %v", tt.want, err)
			}
			var apiErr *ai.APIError
			if !errors.As(err, &apiErr) {
				t.Fatalf("errors.As(*APIError) failed")
			}
			if apiErr.Type != tt.errType || apiErr.Message != tt.message {
				t.Errorf("APIError detail = %+v", apiErr)
			}
		})
	}
}

func TestParseErrorRetryAfter(t *testing.T) {
	resp := &http.Response{StatusCode: 429, Header: http.Header{}}
	resp.Header.Set("Retry-After", "7")
	err := parseError(resp, []byte(`{"error":{"type":"rate_limit_error","message":"x"}}`))
	var apiErr *ai.APIError
	if !errors.As(err, &apiErr) {
		t.Fatal("not an APIError")
	}
	if apiErr.RetryAfter.Seconds() != 7 {
		t.Errorf("RetryAfter = %v, want 7s", apiErr.RetryAfter)
	}
}
