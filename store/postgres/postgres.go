// Package postgres provides PostgreSQL-backed implementations of the go-ai-sdk
// memory.Store (conversation persistence) and vector.Store (similarity search
// via the pgvector extension) interfaces.
//
// The vector store uses pgvector's native cosine distance operator (<=>) and an
// index, so it scales to large corpora — unlike the exact-search sqlite store.
// It records the embedding model and dimension per document and refuses queries
// whose model does not match, mirroring the in-memory and sqlite stores.
//
// Migrate requires the pgvector extension to be available
// (CREATE EXTENSION vector). Integration tests run only when TEST_POSTGRES_DSN
// is set; everything else is exercised at compile and vet time.
package postgres

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5/pgxpool"
)

// Store is a PostgreSQL-backed store implementing both memory.Store and
// vector.Store. Construct it with Open, then call Migrate once.
type Store struct {
	pool *pgxpool.Pool
	dim  int
}

// Option configures a Store.
type Option func(*Store)

// WithDimension sets the embedding dimension used when creating the documents
// table's pgvector column. It must match the embedding model's output width.
// Defaults to 1536 (OpenAI text-embedding-3-small).
func WithDimension(d int) Option { return func(s *Store) { s.dim = d } }

// Open connects to PostgreSQL using a pgx pool.
func Open(ctx context.Context, dsn string, opts ...Option) (*Store, error) {
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("postgres: connect: %w", err)
	}
	s := &Store{pool: pool, dim: 1536}
	for _, opt := range opts {
		opt(s)
	}
	return s, nil
}

// FromPool wraps an existing pgx pool. The caller retains ownership.
func FromPool(pool *pgxpool.Pool, opts ...Option) *Store {
	s := &Store{pool: pool, dim: 1536}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// Close closes the underlying pool.
func (s *Store) Close() { s.pool.Close() }

// Migrate creates the conversations and documents tables (and the pgvector
// extension) if they do not exist. The documents.embedding column is
// vector(dimension); call it once with the dimension matching your embeddings.
func (s *Store) Migrate(ctx context.Context) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE TABLE IF NOT EXISTS conversations (
			conversation_id TEXT NOT NULL,
			seq             BIGINT NOT NULL,
			message_json    JSONB NOT NULL,
			PRIMARY KEY (conversation_id, seq)
		)`,
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS documents (
			id            TEXT PRIMARY KEY,
			model         TEXT NOT NULL,
			text          TEXT NOT NULL,
			metadata      JSONB NOT NULL DEFAULT '{}',
			embedding     vector(%d) NOT NULL
		)`, s.dim),
		`CREATE INDEX IF NOT EXISTS idx_documents_model ON documents(model)`,
	}
	for _, stmt := range stmts {
		if _, err := s.pool.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("postgres: migrate: %w", err)
		}
	}
	return nil
}
