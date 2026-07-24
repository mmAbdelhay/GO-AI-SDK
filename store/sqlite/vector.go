package sqlite

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"github.com/mmabdelhay/go-ai-sdk/vector"
)

// Upsert implements vector.Store. It records the embedding model alongside each
// vector so queries can refuse cross-model comparisons.
func (s *Store) Upsert(ctx context.Context, model string, docs ...vector.Document) error {
	if model == "" {
		return fmt.Errorf("sqlite: embedding model is required on upsert")
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: upsert: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	stmt, err := tx.PrepareContext(ctx, `
INSERT INTO documents (id, model, text, metadata_json, embedding) VALUES (?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET model=excluded.model, text=excluded.text,
	metadata_json=excluded.metadata_json, embedding=excluded.embedding`)
	if err != nil {
		return fmt.Errorf("sqlite: upsert: %w", err)
	}
	defer stmt.Close()

	for _, d := range docs {
		if d.ID == "" {
			return fmt.Errorf("sqlite: document ID is required")
		}
		if len(d.Embedding) == 0 {
			return fmt.Errorf("sqlite: document %q has no embedding", d.ID)
		}
		meta, err := json.Marshal(d.Metadata)
		if err != nil {
			return fmt.Errorf("sqlite: marshaling metadata: %w", err)
		}
		if _, err := stmt.ExecContext(ctx, d.ID, model, d.Text, string(meta), encodeVector(d.Embedding)); err != nil {
			return fmt.Errorf("sqlite: upsert: %w", err)
		}
	}
	return tx.Commit()
}

// Query implements vector.Store. Candidate rows are filtered by model (and
// metadata) in SQL, then scored by exact cosine similarity in Go.
func (s *Store) Query(ctx context.Context, q vector.Query) ([]vector.Match, error) {
	if q.Model == "" {
		return nil, fmt.Errorf("sqlite: query model is required")
	}
	if len(q.Embedding) == 0 {
		return nil, fmt.Errorf("sqlite: query embedding is required")
	}
	topK := q.TopK
	if topK <= 0 {
		topK = 5
	}

	// Refuse the query if the store holds vectors only from other models.
	var withModel int
	if err := s.db.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM documents WHERE model = ?`, q.Model).Scan(&withModel); err != nil {
		return nil, fmt.Errorf("sqlite: query: %w", err)
	}
	if withModel == 0 {
		var total int
		if err := s.db.QueryRowContext(ctx, `SELECT COUNT(*) FROM documents`).Scan(&total); err != nil {
			return nil, fmt.Errorf("sqlite: query: %w", err)
		}
		if total > 0 {
			return nil, fmt.Errorf("%w: query model %q", vector.ErrModelMismatch, q.Model)
		}
	}

	rows, err := s.db.QueryContext(ctx,
		`SELECT id, text, metadata_json, embedding FROM documents WHERE model = ?`, q.Model)
	if err != nil {
		return nil, fmt.Errorf("sqlite: query: %w", err)
	}
	defer rows.Close()

	var matches []vector.Match
	for rows.Next() {
		var (
			id, text, metaJSON string
			blob               []byte
		)
		if err := rows.Scan(&id, &text, &metaJSON, &blob); err != nil {
			return nil, fmt.Errorf("sqlite: query: %w", err)
		}
		var meta map[string]string
		if err := json.Unmarshal([]byte(metaJSON), &meta); err != nil {
			return nil, fmt.Errorf("sqlite: unmarshaling metadata: %w", err)
		}
		if !matchesFilters(meta, q.Filters) {
			continue
		}
		emb := decodeVector(blob)
		if len(emb) != len(q.Embedding) {
			return nil, fmt.Errorf("%w: query %d vs stored %d (doc %q)",
				vector.ErrDimensionMismatch, len(q.Embedding), len(emb), id)
		}
		matches = append(matches, vector.Match{
			Document: vector.Document{ID: id, Text: text, Metadata: meta, Embedding: emb},
			Score:    cosine(q.Embedding, emb),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("sqlite: query: %w", err)
	}

	sort.Slice(matches, func(i, j int) bool { return matches[i].Score > matches[j].Score })
	if len(matches) > topK {
		matches = matches[:topK]
	}
	return matches, nil
}

// Delete implements vector.Store.
func (s *Store) Delete(ctx context.Context, ids ...string) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("sqlite: delete: %w", err)
	}
	defer func() { _ = tx.Rollback() }()
	stmt, err := tx.PrepareContext(ctx, `DELETE FROM documents WHERE id = ?`)
	if err != nil {
		return fmt.Errorf("sqlite: delete: %w", err)
	}
	defer stmt.Close()
	for _, id := range ids {
		if _, err := stmt.ExecContext(ctx, id); err != nil {
			return fmt.Errorf("sqlite: delete: %w", err)
		}
	}
	return tx.Commit()
}

var _ vector.Store = (*Store)(nil)

// encodeVector serializes a float32 slice to little-endian bytes for BLOB
// storage.
func encodeVector(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func decodeVector(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func matchesFilters(meta map[string]string, filters []vector.Filter) bool {
	for _, f := range filters {
		if meta[f.Key] != f.Value {
			return false
		}
	}
	return true
}

func cosine(a, b []float32) float32 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return float32(dot / (math.Sqrt(na) * math.Sqrt(nb)))
}
