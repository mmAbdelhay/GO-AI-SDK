package aitest_test

import (
	"context"
	"errors"
	"testing"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/aitest"
)

func TestModelScriptedResponses(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{
		aitest.TextResponse("first"),
		aitest.TextResponse("second"),
	}}

	r1, err := ai.Generate(context.Background(), m, "a")
	if err != nil || r1.Text() != "first" {
		t.Fatalf("r1 = %q, err = %v", r1.Text(), err)
	}
	r2, _ := ai.Generate(context.Background(), m, "b")
	if r2.Text() != "second" {
		t.Errorf("r2 = %q", r2.Text())
	}

	// Queue exhausted.
	if _, err := ai.Generate(context.Background(), m, "c"); !errors.Is(err, aitest.ErrNoResponse) {
		t.Errorf("exhausted err = %v, want ErrNoResponse", err)
	}

	if m.CallCount() != 3 {
		t.Errorf("CallCount = %d, want 3", m.CallCount())
	}
	last, ok := m.LastRequest()
	if !ok || last.Messages[0].Content[0].(ai.Text).Text != "c" {
		t.Errorf("LastRequest = %+v", last)
	}
}

func TestModelErr(t *testing.T) {
	sentinel := errors.New("scripted failure")
	m := &aitest.Model{Err: sentinel}
	if _, err := ai.Generate(context.Background(), m, "x"); !errors.Is(err, sentinel) {
		t.Errorf("err = %v", err)
	}
}

func TestModelGenerateFunc(t *testing.T) {
	m := &aitest.Model{GenerateFunc: func(_ context.Context, req ai.Request) (ai.Response, error) {
		return aitest.TextResponse("echo:" + req.Messages[0].Content[0].(ai.Text).Text), nil
	}}
	r, _ := ai.Generate(context.Background(), m, "hi")
	if r.Text() != "echo:hi" {
		t.Errorf("r = %q", r.Text())
	}
}

func TestModelStreamDerivedFromResponse(t *testing.T) {
	m := &aitest.Model{Responses: []ai.Response{aitest.TextResponse("streamed")}}
	text, err := ai.GenerateStream(context.Background(), m, "x").Text()
	if err != nil {
		t.Fatal(err)
	}
	if text != "streamed" {
		t.Errorf("text = %q", text)
	}
}

func TestStreamOfHelper(t *testing.T) {
	resp, err := aitest.StreamOf("a", "b", "c").Collect()
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text() != "abc" || resp.StopReason != ai.StopEndTurn {
		t.Errorf("resp = %+v", resp)
	}
}

func TestErrorStreamHelper(t *testing.T) {
	sentinel := errors.New("mid-stream")
	_, err := aitest.ErrorStream(sentinel, "partial").Text()
	if !errors.Is(err, sentinel) {
		t.Errorf("err = %v", err)
	}
}
