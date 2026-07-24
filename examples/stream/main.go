// Command stream is a minimal example of streaming text generation using the
// range-over-func iterator.
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/stream
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("set ANTHROPIC_API_KEY")
	}

	model := anthropic.New(key, anthropic.WithModel("claude-sonnet-4-20250514"))

	ctx := context.Background()
	stream := ai.GenerateStream(ctx, model, "Tell me a short story about a gopher.",
		ai.WithMaxTokens(400))

	for chunk, err := range stream {
		if err != nil {
			log.Fatalf("\nstream: %v", err)
		}
		switch chunk.Type {
		case ai.ChunkText:
			fmt.Print(chunk.Text)
		case ai.ChunkDone:
			fmt.Printf("\n\n[%s] tokens: in=%d out=%d\n",
				chunk.StopReason, chunk.Usage.InputTokens, chunk.Usage.OutputTokens)
		}
	}
}
