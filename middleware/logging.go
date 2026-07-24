package middleware

import (
	"context"
	"log/slog"
	"time"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Logging returns a middleware that logs every model call through logger:
// successes at Info with model, duration, tokens, and stop reason; failures at
// Error with the error. Streams are logged once, on completion.
func Logging(logger *slog.Logger) Middleware {
	if logger == nil {
		logger = slog.Default()
	}
	return func(next ai.ChatModel) ai.ChatModel {
		name := next.Name()
		return WithHooks(Hooks{
			After: func(ctx context.Context, req ai.Request, resp ai.Response, err error, d time.Duration) {
				model := req.Model
				if model == "" {
					model = name
				}
				if err != nil {
					logger.LogAttrs(ctx, slog.LevelError, "ai.call",
						slog.String("model", model),
						slog.Duration("duration", d),
						slog.String("error", err.Error()),
					)
					return
				}
				logger.LogAttrs(ctx, slog.LevelInfo, "ai.call",
					slog.String("model", model),
					slog.Duration("duration", d),
					slog.Int("input_tokens", resp.Usage.InputTokens),
					slog.Int("output_tokens", resp.Usage.OutputTokens),
					slog.String("stop_reason", string(resp.StopReason)),
				)
			},
		})(next)
	}
}
