//go:build integration

// Package anthropic integration tests hit the real Anthropic API. They are
// excluded from the default build and only run with:
//
//	ANTHROPIC_API_KEY=... go test -tags integration ./provider/anthropic
package anthropic_test

import (
	"context"
	"os"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

func integrationClient(t *testing.T) *anthropic.Client {
	t.Helper()
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		t.Skip("ANTHROPIC_API_KEY not set")
	}
	model := os.Getenv("ANTHROPIC_MODEL")
	if model == "" {
		model = "claude-sonnet-4-20250514"
	}
	return anthropic.New(key, anthropic.WithModel(model))
}

func TestIntegrationGenerate(t *testing.T) {
	client := integrationClient(t)
	resp, err := ai.Generate(context.Background(), client,
		"Reply with exactly the word: pong", ai.WithMaxTokens(16))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() == "" {
		t.Errorf("empty response")
	}
	if resp.Usage.TotalTokens() == 0 {
		t.Errorf("no usage reported")
	}
}

func TestIntegrationStream(t *testing.T) {
	client := integrationClient(t)
	text, err := ai.GenerateStream(context.Background(), client,
		"Count from 1 to 3.", ai.WithMaxTokens(32)).Text()
	if err != nil {
		t.Fatal(err)
	}
	if text == "" {
		t.Errorf("empty streamed response")
	}
}
