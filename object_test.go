package ai_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
)

type person struct {
	Name string `json:"name"`
	Age  int    `json:"age"`
}

func TestGenerateObject(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse(`{"name":"Ada","age":36}`),
	}}
	got, err := ai.GenerateObject[person](context.Background(), m, "who?")
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "Ada" || got.Age != 36 {
		t.Errorf("got = %+v", got)
	}
	// The schema instruction must be in the system prompt.
	last, _ := m.LastRequest()
	if !strings.Contains(last.System, "JSON Schema") {
		t.Errorf("schema instruction missing from system prompt")
	}
}

func TestGenerateObjectStripsFences(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse("Here you go:\n```json\n{\"name\":\"Ada\",\"age\":36}\n```\n"),
	}}
	got, err := ai.GenerateObject[person](context.Background(), m, "who?")
	if err != nil {
		t.Fatal(err)
	}
	if got.Name != "Ada" {
		t.Errorf("got = %+v", got)
	}
}

func TestGenerateObjectRepairLoop(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse(`{"name":"Ada"}`),          // missing required "age"
		aitest.TextResponse(`{"name":"Ada","age":36}`), // repaired
	}}
	got, err := ai.GenerateObject[person](context.Background(), m, "who?")
	if err != nil {
		t.Fatal(err)
	}
	if got.Age != 36 {
		t.Errorf("got = %+v", got)
	}
	if m.CallCount() != 2 {
		t.Errorf("calls = %d, want 2", m.CallCount())
	}
	// The repair request must include the validation feedback.
	last, _ := m.LastRequest()
	feedback := last.Messages[len(last.Messages)-1].Content[0].(ai.Text).Text
	if !strings.Contains(feedback, "age") {
		t.Errorf("repair feedback missing failing field: %q", feedback)
	}
}

func TestGenerateObjectExhaustsRepairs(t *testing.T) {
	m := &aitest.Model{GenerateFunc: func(context.Context, ai.Request) (ai.Response, error) {
		return aitest.TextResponse(`not json at all`), nil
	}}
	_, err := ai.GenerateObject[person](context.Background(), m, "who?", ai.WithObjectRepairs(1))
	if !errors.Is(err, ai.ErrInvalidSchema) {
		t.Fatalf("err = %v, want ErrInvalidSchema", err)
	}
}

func TestGenerateObjectNoRepairs(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{aitest.TextResponse(`{}`)}}
	_, err := ai.GenerateObject[person](context.Background(), m, "who?", ai.WithObjectRepairs(-1))
	if !errors.Is(err, ai.ErrInvalidSchema) {
		t.Fatal("expected schema error")
	}
	if m.CallCount() != 1 {
		t.Errorf("calls = %d, want 1 (repairs disabled)", m.CallCount())
	}
}

func TestExtractJSON(t *testing.T) {
	tests := []struct{ in, want string }{
		{`{"a":1}`, `{"a":1}`},
		{"```json\n{\"a\":1}\n```", `{"a":1}`},
		{"Sure! Here it is: {\"a\":1} hope that helps", `{"a":1}`},
		{"prose [1,2] more", `[1,2]`},
		{"no json here", "no json here"},
	}
	for _, tt := range tests {
		if got := string(ai.ExtractJSON(tt.in)); got != tt.want {
			t.Errorf("ExtractJSON(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
