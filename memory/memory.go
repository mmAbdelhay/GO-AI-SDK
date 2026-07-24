// Package memory provides conversation persistence (the Store interface with an
// in-memory implementation) and context-management strategies that keep long
// conversations within a model's context window.
package memory

import (
	"context"
	"sync"

	ai "github.com/mmabdelhay/go-ai-sdk"
)

// Store persists conversation history keyed by conversation ID. SQL-backed
// implementations live in separate submodules so the core stays dependency-free.
type Store interface {
	// Append adds messages to the conversation.
	Append(ctx context.Context, conversationID string, msgs ...ai.Message) error
	// Messages returns the conversation's messages in order. A missing
	// conversation yields an empty slice, not an error.
	Messages(ctx context.Context, conversationID string) ([]ai.Message, error)
	// Clear removes the conversation.
	Clear(ctx context.Context, conversationID string) error
}

// InMemory is a Store keeping conversations in process memory. It is safe for
// concurrent use. Useful for tests, CLIs, and single-process servers.
type InMemory struct {
	mu    sync.RWMutex
	convs map[string][]ai.Message
}

// NewInMemory returns an empty in-memory store.
func NewInMemory() *InMemory {
	return &InMemory{convs: make(map[string][]ai.Message)}
}

// Append implements [Store].
func (s *InMemory) Append(_ context.Context, id string, msgs ...ai.Message) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.convs[id] = append(s.convs[id], msgs...)
	return nil
}

// Messages implements [Store].
func (s *InMemory) Messages(_ context.Context, id string) ([]ai.Message, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	msgs := s.convs[id]
	out := make([]ai.Message, len(msgs))
	copy(out, msgs)
	return out, nil
}

// Clear implements [Store].
func (s *InMemory) Clear(_ context.Context, id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.convs, id)
	return nil
}

var _ Store = (*InMemory)(nil)
