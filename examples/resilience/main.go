// Command resilience demonstrates provider failover: Anthropic primary with
// retry, falling back to OpenAI, with the failover event surfaced to the
// caller.
//
//	ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... go run ./examples/resilience
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
	"github.com/mmabdelhay/go-ai-sdk/resilience"
)

func main() {
	anthropicKey, openaiKey := os.Getenv("ANTHROPIC_API_KEY"), os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY and OPENAI_API_KEY")
	}

	primary := resilience.NewRetry(
		anthropic.New(anthropicKey, anthropic.WithModel("claude-sonnet-4-20250514")),
		resilience.RetryConfig{
			MaxAttempts: 2,
			BaseDelay:   time.Second,
			OnRetry: func(attempt int, err error, delay time.Duration) {
				fmt.Printf("  [retry %d after %v: %v]\n", attempt, delay, err)
			},
		})

	model := resilience.NewFallback(resilience.FallbackConfig{
		OnFailover: func(from, to string, err error) {
			fmt.Printf("  [failover %s -> %s: %v]\n", from, to, err)
		},
	},
		primary,
		openai.New(openaiKey, openai.WithModel("gpt-4o-mini")),
	)

	resp, err := ai.Generate(context.Background(), model,
		"Say hello in one short sentence.", ai.WithMaxTokens(50))
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Printf("%s\n(served by %s)\n", resp.Text(), resp.Model)
}
