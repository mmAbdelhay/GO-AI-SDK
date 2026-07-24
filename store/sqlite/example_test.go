package sqlite_test

import (
	"context"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/store/sqlite"
)

// Example persists a conversation to SQLite and reads it back. Use a file path
// instead of ":memory:" to persist across process restarts.
func Example() {
	store, err := sqlite.Open(":memory:")
	if err != nil {
		panic(err)
	}
	defer store.Close()

	ctx := context.Background()
	_ = store.Append(ctx, "conv-1", ai.UserText("Hi"), ai.AssistantText("Hello!"))

	msgs, _ := store.Messages(ctx, "conv-1")
	for _, m := range msgs {
		fmt.Printf("%s: %s\n", m.Role, msgText(m))
	}
	// Output:
	// user: Hi
	// assistant: Hello!
}

func msgText(m ai.Message) string {
	for _, p := range m.Content {
		if t, ok := p.(ai.Text); ok {
			return t.Text
		}
	}
	return ""
}
