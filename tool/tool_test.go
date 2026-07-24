package tool_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/tool"
)

type addInput struct {
	A int `json:"a"`
	B int `json:"b"`
}

func newAdd(t *testing.T) tool.Tool {
	t.Helper()
	return tool.MustNew("add", "Adds two integers",
		func(_ context.Context, in addInput) (string, error) {
			return fmt.Sprint(in.A + in.B), nil
		})
}

func TestFuncToolCall(t *testing.T) {
	add := newAdd(t)
	out, err := add.Call(context.Background(), json.RawMessage(`{"a":2,"b":3}`))
	if err != nil {
		t.Fatal(err)
	}
	if out != "5" {
		t.Errorf("out = %q", out)
	}
}

func TestFuncToolValidatesInput(t *testing.T) {
	add := newAdd(t)
	tests := []struct {
		name  string
		input string
	}{
		{"missing field", `{"a":2}`},
		{"wrong type", `{"a":"two","b":3}`},
		{"unknown field", `{"a":1,"b":2,"c":3}`},
		{"not json", `nope`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := add.Call(context.Background(), json.RawMessage(tt.input)); err == nil {
				t.Errorf("Call(%s) succeeded, want validation error", tt.input)
			}
		})
	}
}

func TestDef(t *testing.T) {
	def := tool.Def(newAdd(t))
	if def.Name != "add" || def.Description == "" {
		t.Errorf("def = %+v", def)
	}
	if !strings.Contains(string(def.InputSchema), `"integer"`) {
		t.Errorf("schema = %s", def.InputSchema)
	}
}

func TestRegistryDuplicate(t *testing.T) {
	a := newAdd(t)
	if _, err := tool.NewRegistry(a, a); err == nil {
		t.Fatal("expected duplicate-name error")
	}
}

func TestRegistryExecute(t *testing.T) {
	reg, err := tool.NewRegistry(newAdd(t))
	if err != nil {
		t.Fatal(err)
	}

	tr, err := reg.Execute(context.Background(),
		ai.ToolCall{ID: "c1", Name: "add", Input: json.RawMessage(`{"a":1,"b":1}`)})
	if err != nil {
		t.Fatal(err)
	}
	if tr.IsError || tr.CallID != "c1" || tr.Content[0].(ai.Text).Text != "2" {
		t.Errorf("result = %+v", tr)
	}

	// Unknown tools become error results the model can react to, not Go errors.
	tr, err = reg.Execute(context.Background(), ai.ToolCall{ID: "c2", Name: "nope"})
	if err != nil {
		t.Fatal(err)
	}
	if !tr.IsError {
		t.Errorf("unknown tool should produce IsError result")
	}

	// Tool failures likewise surface to the model.
	failing := tool.MustNew("boom", "always fails",
		func(context.Context, struct{}) (string, error) { return "", errors.New("kaput") })
	reg2, _ := tool.NewRegistry(failing)
	tr, err = reg2.Execute(context.Background(), ai.ToolCall{ID: "c3", Name: "boom", Input: json.RawMessage(`{}`)})
	if err != nil {
		t.Fatal(err)
	}
	if !tr.IsError || !strings.Contains(tr.Content[0].(ai.Text).Text, "kaput") {
		t.Errorf("failure result = %+v", tr)
	}
}

func TestRegistryExecuteCancelled(t *testing.T) {
	reg, _ := tool.NewRegistry(newAdd(t))
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := reg.Execute(ctx, ai.ToolCall{Name: "add"}); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}
