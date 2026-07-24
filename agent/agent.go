// Package agent implements the agentic loop: call the model, detect tool use,
// execute tools, feed results back, and repeat until the model answers or a
// safeguard trips. Safeguards — an iteration cap, a token budget, context
// cancellation, and a human-approval hook — are mandatory parts of the loop, not
// options bolted on.
package agent

import (
	"context"
	"errors"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/memory"
	"github.com/mmabdelhay/go-ai-sdk/tool"
)

// Safeguard sentinels. A tripped safeguard returns a *LimitError wrapping one of
// these, carrying the partial Result.
var (
	// ErrMaxIterations means the loop hit its iteration cap without the model
	// producing a final answer.
	ErrMaxIterations = errors.New("agent: max iterations reached")
	// ErrTokenBudget means cumulative token usage exceeded the configured
	// budget.
	ErrTokenBudget = errors.New("agent: token budget exceeded")
	// ErrApprovalDenied means the approval hook rejected a tool call.
	ErrApprovalDenied = errors.New("agent: tool call denied")
)

// LimitError is returned when a safeguard stops the run. It wraps the matching
// sentinel and carries the partial Result so callers can inspect what happened
// before the stop.
type LimitError struct {
	Sentinel error
	Detail   string
	Result   *Result
}

func (e *LimitError) Error() string {
	if e.Detail != "" {
		return e.Sentinel.Error() + ": " + e.Detail
	}
	return e.Sentinel.Error()
}

func (e *LimitError) Unwrap() error { return e.Sentinel }

// Agent runs a model with tools until completion. Construct with [New]; the
// zero value is not usable.
type Agent struct {
	model         ai.ChatModel
	registry      *tool.Registry
	system        string
	maxIterations int
	tokenBudget   int
	approve       func(ctx context.Context, call ai.ToolCall) error
	store         memory.Store
	conversation  string
	strategy      memory.Strategy
	contextBudget int
	requestOpts   []ai.Option
}

// Option configures an [Agent].
type Option func(*Agent)

// WithTools registers the agent's tools.
func WithTools(tools ...tool.Tool) Option {
	return func(a *Agent) {
		r, err := tool.NewRegistry(tools...)
		if err != nil {
			panic(err) // duplicate tool names are a programming error
		}
		a.registry = r
	}
}

// WithSystem sets the agent's system prompt.
func WithSystem(s string) Option { return func(a *Agent) { a.system = s } }

// WithMaxIterations caps model-call iterations per run (default 10).
func WithMaxIterations(n int) Option { return func(a *Agent) { a.maxIterations = n } }

// WithTokenBudget caps cumulative token usage per run; 0 means unlimited.
func WithTokenBudget(n int) Option { return func(a *Agent) { a.tokenBudget = n } }

// WithApproval installs a hook invoked before every tool execution. Returning an
// error aborts the run with a *LimitError wrapping [ErrApprovalDenied]. Use it
// to gate side-effecting tools behind human review.
func WithApproval(fn func(ctx context.Context, call ai.ToolCall) error) Option {
	return func(a *Agent) { a.approve = fn }
}

// WithMemory persists the conversation in store under conversationID, so
// successive Run calls share history.
func WithMemory(store memory.Store, conversationID string) Option {
	return func(a *Agent) { a.store = store; a.conversation = conversationID }
}

// WithContextStrategy applies a context-management strategy before each model
// call, keeping the conversation within budgetTokens (estimated).
func WithContextStrategy(s memory.Strategy, budgetTokens int) Option {
	return func(a *Agent) { a.strategy = s; a.contextBudget = budgetTokens }
}

// WithRequestOptions applies ai options (temperature, max tokens, provider
// options, ...) to every model call the agent makes.
func WithRequestOptions(opts ...ai.Option) Option {
	return func(a *Agent) { a.requestOpts = append(a.requestOpts, opts...) }
}

// New constructs an Agent for the given model.
func New(model ai.ChatModel, opts ...Option) *Agent {
	a := &Agent{model: model, maxIterations: 10}
	for _, opt := range opts {
		opt(a)
	}
	if a.registry == nil {
		a.registry, _ = tool.NewRegistry()
	}
	return a
}

// Result is the outcome of a run.
type Result struct {
	// Output is the model's final text answer.
	Output string
	// Response is the final model response.
	Response ai.Response
	// Messages is the transcript of this run (excluding prior stored history).
	Messages []ai.Message
	// Usage is cumulative token usage across all iterations.
	Usage ai.Usage
	// Iterations is how many model calls were made.
	Iterations int
	// ToolCalls is how many tool executions occurred.
	ToolCalls int
}

// Run executes the agent loop for prompt and returns the final result. Cancel
// via ctx; cancellation is honored between iterations and during tool
// execution.
func (a *Agent) Run(ctx context.Context, prompt string) (*Result, error) {
	res := &Result{}

	var history []ai.Message
	if a.store != nil {
		var err error
		history, err = a.store.Messages(ctx, a.conversation)
		if err != nil {
			return nil, fmt.Errorf("agent: loading memory: %w", err)
		}
	}

	userMsg := ai.UserText(prompt)
	res.Messages = append(res.Messages, userMsg)
	msgs := append(append([]ai.Message{}, history...), userMsg)

	defs := a.registry.Defs()

	for res.Iterations < a.maxIterations {
		if err := ctx.Err(); err != nil {
			return res, err
		}

		if a.strategy != nil {
			fitted, err := a.strategy.Fit(ctx, msgs, a.contextBudget)
			if err != nil {
				return res, fmt.Errorf("agent: context strategy: %w", err)
			}
			msgs = fitted
		}

		req := ai.Request{System: a.system, Messages: msgs, Tools: defs}
		for _, opt := range a.requestOpts {
			opt(&req)
		}

		resp, err := a.model.Generate(ctx, req)
		if err != nil {
			return res, err
		}
		res.Iterations++
		res.Usage.InputTokens += resp.Usage.InputTokens
		res.Usage.OutputTokens += resp.Usage.OutputTokens
		res.Usage.CacheCreationTokens += resp.Usage.CacheCreationTokens
		res.Usage.CacheReadTokens += resp.Usage.CacheReadTokens
		res.Response = resp

		msgs = append(msgs, resp.Message)
		res.Messages = append(res.Messages, resp.Message)

		if a.tokenBudget > 0 && res.Usage.TotalTokens() > a.tokenBudget {
			return nil, &LimitError{
				Sentinel: ErrTokenBudget,
				Detail:   fmt.Sprintf("used %d of %d tokens", res.Usage.TotalTokens(), a.tokenBudget),
				Result:   res,
			}
		}

		calls := toolCalls(resp.Message)
		if len(calls) == 0 {
			res.Output = resp.Text()
			if err := a.persist(ctx, res); err != nil {
				return res, err
			}
			return res, nil
		}

		results := make([]ai.Part, 0, len(calls))
		for _, call := range calls {
			if a.approve != nil {
				if err := a.approve(ctx, call); err != nil {
					return nil, &LimitError{
						Sentinel: ErrApprovalDenied,
						Detail:   fmt.Sprintf("tool %q: %v", call.Name, err),
						Result:   res,
					}
				}
			}
			tr, err := a.registry.Execute(ctx, call)
			if err != nil {
				return res, err // context cancellation mid-loop
			}
			res.ToolCalls++
			results = append(results, tr)
		}
		resultMsg := ai.Message{Role: ai.RoleTool, Content: results}
		msgs = append(msgs, resultMsg)
		res.Messages = append(res.Messages, resultMsg)
	}

	return nil, &LimitError{
		Sentinel: ErrMaxIterations,
		Detail:   fmt.Sprintf("%d iterations", res.Iterations),
		Result:   res,
	}
}

func (a *Agent) persist(ctx context.Context, res *Result) error {
	if a.store == nil {
		return nil
	}
	if err := a.store.Append(ctx, a.conversation, res.Messages...); err != nil {
		return fmt.Errorf("agent: saving memory: %w", err)
	}
	return nil
}

func toolCalls(m ai.Message) []ai.ToolCall {
	var calls []ai.ToolCall
	for _, p := range m.Content {
		if c, ok := p.(ai.ToolCall); ok {
			calls = append(calls, c)
		}
	}
	return calls
}
