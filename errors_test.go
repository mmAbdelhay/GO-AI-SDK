package ai_test

import (
	"errors"
	"testing"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

func TestAPIErrorUnwrapsToSentinel(t *testing.T) {
	apiErr := ai.NewAPIError("anthropic", 429, ai.ErrRateLimited)
	apiErr.RetryAfter = 5 * time.Second
	apiErr.Message = "slow down"

	var err error = apiErr
	if !errors.Is(err, ai.ErrRateLimited) {
		t.Errorf("errors.Is(err, ErrRateLimited) = false, want true")
	}
	if errors.Is(err, ai.ErrAuthentication) {
		t.Errorf("errors.Is matched the wrong sentinel")
	}

	var target *ai.APIError
	if !errors.As(err, &target) {
		t.Fatalf("errors.As(*APIError) = false")
	}
	if target.StatusCode != 429 || target.RetryAfter != 5*time.Second {
		t.Errorf("APIError detail not preserved: %+v", target)
	}
}

func TestAPIErrorMessageFormat(t *testing.T) {
	e := ai.NewAPIError("anthropic", 400, ai.ErrInvalidRequest)
	e.Message = "bad field"
	if got := e.Error(); got == "" {
		t.Errorf("Error() returned empty string")
	}
}
