package postgres

import (
	"context"
	"encoding/json"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/memory"
)

// Append implements memory.Store, storing one row per message ordered by an
// increasing sequence, using the canonical ai.Message JSON encoding.
func (s *Store) Append(ctx context.Context, conversationID string, msgs ...ai.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("postgres: append: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	var next int64
	if err := tx.QueryRow(ctx,
		`SELECT COALESCE(MAX(seq)+1, 0) FROM conversations WHERE conversation_id = $1`,
		conversationID).Scan(&next); err != nil {
		return fmt.Errorf("postgres: append: %w", err)
	}
	for i, m := range msgs {
		data, err := json.Marshal(m)
		if err != nil {
			return fmt.Errorf("postgres: marshaling message: %w", err)
		}
		if _, err := tx.Exec(ctx,
			`INSERT INTO conversations (conversation_id, seq, message_json) VALUES ($1, $2, $3)`,
			conversationID, next+int64(i), string(data)); err != nil {
			return fmt.Errorf("postgres: append: %w", err)
		}
	}
	return tx.Commit(ctx)
}

// Messages implements memory.Store.
func (s *Store) Messages(ctx context.Context, conversationID string) ([]ai.Message, error) {
	rows, err := s.pool.Query(ctx,
		`SELECT message_json FROM conversations WHERE conversation_id = $1 ORDER BY seq`,
		conversationID)
	if err != nil {
		return nil, fmt.Errorf("postgres: messages: %w", err)
	}
	defer rows.Close()

	var msgs []ai.Message
	for rows.Next() {
		var data []byte
		if err := rows.Scan(&data); err != nil {
			return nil, fmt.Errorf("postgres: messages: %w", err)
		}
		var m ai.Message
		if err := json.Unmarshal(data, &m); err != nil {
			return nil, fmt.Errorf("postgres: unmarshaling message: %w", err)
		}
		msgs = append(msgs, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("postgres: messages: %w", err)
	}
	return msgs, nil
}

// Clear implements memory.Store.
func (s *Store) Clear(ctx context.Context, conversationID string) error {
	if _, err := s.pool.Exec(ctx,
		`DELETE FROM conversations WHERE conversation_id = $1`, conversationID); err != nil {
		return fmt.Errorf("postgres: clear: %w", err)
	}
	return nil
}

var _ memory.Store = (*Store)(nil)
