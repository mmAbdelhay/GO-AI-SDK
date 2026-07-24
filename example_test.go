package ai_test

import (
	"context"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
)

// Example demonstrates generating text against a scripted fake model, so it runs
// with no network access or API key. Swap the fake for a provider client (for
// example anthropic.New) to talk to a real model.
func Example() {
	model := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse("Hello from a fake model!"),
	}}

	resp, err := ai.Generate(context.Background(), model, "Say hello",
		ai.WithTemperature(0.2))
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
	// Output: Hello from a fake model!
}

// ExampleGenerateStream demonstrates consuming a stream with range-over-func.
func ExampleGenerateStream() {
	model := &aitest.Model{StreamFunc: func(context.Context, ai.Request) ai.Stream {
		return aitest.StreamOf("one ", "two ", "three")
	}}

	for chunk, err := range ai.GenerateStream(context.Background(), model, "count") {
		if err != nil {
			panic(err)
		}
		if chunk.Type == ai.ChunkText {
			fmt.Print(chunk.Text)
		}
	}
	fmt.Println()
	// Output: one two three
}
