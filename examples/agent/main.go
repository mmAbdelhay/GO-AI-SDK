// Command agent demonstrates the agent loop: a model with tools, memory, and
// safeguards completing a multi-step task.
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/agent
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strconv"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/agent"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
	"github.com/mmabdelhay/go-ai-sdk/tool"
)

type weatherInput struct {
	City string `json:"city" desc:"City name, e.g. Cairo"`
}

type convertInput struct {
	Celsius float64 `json:"celsius" desc:"Temperature in Celsius"`
}

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("set ANTHROPIC_API_KEY")
	}
	model := anthropic.New(key, anthropic.WithModel("claude-sonnet-4-20250514"))

	weather := tool.MustNew("get_weather", "Current temperature for a city, in Celsius",
		func(_ context.Context, in weatherInput) (string, error) {
			return "31", nil // a real implementation would call a weather API
		})
	toF := tool.MustNew("celsius_to_fahrenheit", "Convert Celsius to Fahrenheit",
		func(_ context.Context, in convertInput) (string, error) {
			return strconv.FormatFloat(in.Celsius*9/5+32, 'f', 1, 64), nil
		})

	a := agent.New(model,
		agent.WithTools(weather, toF),
		agent.WithSystem("Use the tools; do not guess numbers."),
		agent.WithMaxIterations(6),
		agent.WithTokenBudget(20_000),
		agent.WithApproval(func(_ context.Context, call ai.ToolCall) error {
			fmt.Printf("  [approving tool call: %s(%s)]\n", call.Name, call.Input)
			return nil // return an error here to block the call
		}),
	)

	res, err := a.Run(context.Background(),
		"What is the temperature in Cairo in Fahrenheit right now?")
	if err != nil {
		log.Fatalf("run: %v", err)
	}
	fmt.Println(res.Output)
	fmt.Printf("\n(%d iterations, %d tool calls, %d tokens)\n",
		res.Iterations, res.ToolCalls, res.Usage.TotalTokens())
}
