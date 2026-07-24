package postgres

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/mmabdelhay/go-ai-sdk/vector"
)

// Upsert implements vector.Store, recording the embedding model with each
// vector. Every embedding must have the store's configured dimension (see
// WithDimension); pgvector enforces this at the column level.
func (s *Store) Upsert(ctx context.Context, model string, docs ...vector.Document) error {
	if model == "" {
		return fmt.Errorf("postgres: embedding model is required on upsert")
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("postgres: upsert: %w", err)
	}
	defer func() { _ = tx.Rollback(ctx) }()

	for _, d := range docs {
		if d.ID == "" {
			return fmt.Errorf("postgres: document ID is required")
		}
		if len(d.Embedding) == 0 {
			return fmt.Errorf("postgres: document %q has no embedding", d.ID)
		}
		if len(d.Embedding) != s.dim {
			return fmt.Errorf("%w: document %q has %d dims, store expects %d",
				vector.ErrDimensionMismatch, d.ID, len(d.Embedding), s.dim)
		}
		meta, err := json.Marshal(d.Metadata)
		if err != nil {
			return fmt.Errorf("postgres: marshaling metadata: %w", err)
		}
		if _, err := tx.Exec(ctx, `
INSERT INTO documents (id, model, text, metadata, embedding) VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (id) DO UPDATE SET model=EXCLUDED.model, text=EXCLUDED.text,
	metadata=EXCLUDED.metadata, embedding=EXCLUDED.embedding`,
			d.ID, model, d.Text, string(meta), vectorLiteral(d.Embedding)); err != nil {
			return fmt.Errorf("postgres: upsert: %w", err)
		}
	}
	return tx.Commit(ctx)
}

// Query implements vector.Store using pgvector's cosine distance operator.
// Score is cosine similarity (1 - cosine distance).
func (s *Store) Query(ctx context.Context, q vector.Query) ([]vector.Match, error) {
	if q.Model == "" {
		return nil, fmt.Errorf("postgres: query model is required")
	}
	if len(q.Embedding) == 0 {
		return nil, fmt.Errorf("postgres: query embedding is required")
	}
	if len(q.Embedding) != s.dim {
		return nil, fmt.Errorf("%w: query has %d dims, store expects %d",
			vector.ErrDimensionMismatch, len(q.Embedding), s.dim)
	}
	topK := q.TopK
	if topK <= 0 {
		topK = 5
	}

	// Refuse if the store holds vectors only from other models.
	var withModel int
	if err := s.pool.QueryRow(ctx,
		`SELECT COUNT(*) FROM documents WHERE model = $1`, q.Model).Scan(&withModel); err != nil {
		return nil, fmt.Errorf("postgres: query: %w", err)
	}
	if withModel == 0 {
		var total int
		if err := s.pool.QueryRow(ctx, `SELECT COUNT(*) FROM documents`).Scan(&total); err != nil {
			return nil, fmt.Errorf("postgres: query: %w", err)
		}
		if total > 0 {
			return nil, fmt.Errorf("%w: query model %q", vector.ErrModelMismatch, q.Model)
		}
	}

	// Build the WHERE clause: model plus optional metadata equality filters.
	args := []any{vectorLiteral(q.Embedding), q.Model}
	where := []string{"model = $2"}
	for _, f := range q.Filters {
		args = append(args, f.Value)
		where = append(where, fmt.Sprintf("metadata->>%s = $%d", quoteLiteral(f.Key), len(args)))
	}
	args = append(args, topK)
	sql := fmt.Sprintf(`
SELECT id, text, metadata, 1 - (embedding <=> $1) AS score
FROM documents
WHERE %s
ORDER BY embedding <=> $1
LIMIT $%d`, strings.Join(where, " AND "), len(args))

	rows, err := s.pool.Query(ctx, sql, args...)
	if err != nil {
		return nil, fmt.Errorf("postgres: query: %w", err)
	}
	defer rows.Close()

	var matches []vector.Match
	for rows.Next() {
		var (
			id, text string
			metaRaw  []byte
			score    float64
		)
		if err := rows.Scan(&id, &text, &metaRaw, &score); err != nil {
			return nil, fmt.Errorf("postgres: query: %w", err)
		}
		var meta map[string]string
		if err := json.Unmarshal(metaRaw, &meta); err != nil {
			return nil, fmt.Errorf("postgres: unmarshaling metadata: %w", err)
		}
		matches = append(matches, vector.Match{
			Document: vector.Document{ID: id, Text: text, Metadata: meta},
			Score:    float32(score),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("postgres: query: %w", err)
	}
	return matches, nil
}

// Delete implements vector.Store.
func (s *Store) Delete(ctx context.Context, ids ...string) error {
	if len(ids) == 0 {
		return nil
	}
	if _, err := s.pool.Exec(ctx, `DELETE FROM documents WHERE id = ANY($1)`, ids); err != nil {
		return fmt.Errorf("postgres: delete: %w", err)
	}
	return nil
}

var _ vector.Store = (*Store)(nil)

// vectorLiteral formats a float32 slice as a pgvector text literal: "[1,2,3]".
func vectorLiteral(v []float32) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(f), 'g', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}

// quoteLiteral single-quotes a SQL string literal for use as a JSON key in the
// metadata->>'key' operator. Keys come from application code, but quoting keeps
// it safe regardless.
func quoteLiteral(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}
