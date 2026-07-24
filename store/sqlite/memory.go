package sqlite

import (
	"context"
	"encoding/json"
	"fmt"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/memory"
)

// Append implements memory.Store. Messages are stored one row per message,
// ordered by an increasing sequence number, using the canonical ai.Message JSON
// encoding so all content part types round-trip.
func (s *Store) Append(ctx context.Context, conversationID string, msgs ...ai.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: append: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	var next int
	err = tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(seq)+1, 0) FROM conversations WHERE conversation_id = ?`,
		conversationID).Scan(&next)
	if err != nil {
		return fmt.Errorf("sqlite: append: %w", err)
	}
	stmt, err := tx.PrepareContext(ctx,
		`INSERT INTO conversations (conversation_id, seq, message_json) VALUES (?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("sqlite: append: %w", err)
	}
	defer stmt.Close()

	for i, m := range msgs {
		data, err := json.Marshal(m)
		if err != nil {
			return fmt.Errorf("sqlite: marshaling message: %w", err)
		}
		if _, err := stmt.ExecContext(ctx, conversationID, next+i, string(data)); err != nil {
			return fmt.Errorf("sqlite: append: %w", err)
		}
	}
	return tx.Commit()
}

// Messages implements memory.Store.
func (s *Store) Messages(ctx context.Context, conversationID string) ([]ai.Message, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT message_json FROM conversations WHERE conversation_id = ? ORDER BY seq`,
		conversationID)
	if err != nil {
		return nil, fmt.Errorf("sqlite: messages: %w", err)
	}
	defer rows.Close()

	var msgs []ai.Message
	for rows.Next() {
		var data string
		if err := rows.Scan(&data); err != nil {
			return nil, fmt.Errorf("sqlite: messages: %w", err)
		}
		var m ai.Message
		if err := json.Unmarshal([]byte(data), &m); err != nil {
			return nil, fmt.Errorf("sqlite: unmarshaling message: %w", err)
		}
		msgs = append(msgs, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: messages: %w", err)
	}
	return msgs, nil
}

// Clear implements memory.Store.
func (s *Store) Clear(ctx context.Context, conversationID string) error {
	if _, err := s.db.ExecContext(ctx,
		`DELETE FROM conversations WHERE conversation_id = ?`, conversationID); err != nil {
		return fmt.Errorf("sqlite: clear: %w", err)
	}
	return nil
}

var _ memory.Store = (*Store)(nil)
