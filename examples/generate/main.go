// Command generate is a minimal example of one-shot text generation.
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/generate
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("set ANTHROPIC_API_KEY")
	}

	model := anthropic.New(key, anthropic.WithModel("claude-sonnet-4-20250514"))

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := ai.Generate(ctx, model, "Write a haiku about Go's goroutines.",
		ai.WithTemperature(0.7),
		ai.WithMaxTokens(200),
	)
	if err != nil {
		if errors.Is(err, ai.ErrRateLimited) {
			log.Fatal("rate limited; back off and retry")
		}
		log.Fatalf("generate: %v", err)
	}

	fmt.Println(resp.Text())
	fmt.Printf("\n[%s] tokens: in=%d out=%d\n",
		resp.StopReason, resp.Usage.InputTokens, resp.Usage.OutputTokens)
}
