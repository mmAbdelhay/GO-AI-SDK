// Package sqlite provides SQLite-backed implementations of the go-ai-sdk
// memory.Store (conversation persistence) and vector.Store (embedding storage
// and similarity search) interfaces. It uses the pure-Go modernc.org/sqlite
// driver, so it needs no cgo and runs anywhere, including with an in-memory
// database (path ":memory:") for tests.
//
// SQLite has no native vector index, so similarity search is exact
// (brute-force) cosine over the candidate rows, scored in Go. This is well
// suited to small and medium corpora; for large-scale search use the pgvector
// store in store/postgres.
package sqlite

import (
	"database/sql"
	"fmt"

	_ "modernc.org/sqlite"
)

// Store is a SQLite-backed store. It implements both memory.Store and
// vector.Store. Construct it with Open.
type Store struct {
	db *sql.DB
}

// Open opens (or creates) a SQLite database at path and migrates the schema.
// Use ":memory:" for an ephemeral in-process database.
func Open(path string) (*Store, error) {
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, fmt.Errorf("sqlite: open: %w", err)
	}
	// A single connection avoids "database is locked" for :memory: databases,
	// whose data is per-connection.
	db.SetMaxOpenConns(1)
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		_ = db.Close()
		return nil, err
	}
	return s, nil
}

// OpenDB wraps an existing *sql.DB (already opened against a SQLite driver) and
// migrates the schema. The caller retains ownership of db.
func OpenDB(db *sql.DB) (*Store, error) {
	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, err
	}
	return s, nil
}

// Close closes the underlying database.
func (s *Store) Close() error { return s.db.Close() }

func (s *Store) migrate() error {
	const schema = `
CREATE TABLE IF NOT EXISTS conversations (
	conversation_id TEXT NOT NULL,
	seq             INTEGER NOT NULL,
	message_json    TEXT NOT NULL,
	PRIMARY KEY (conversation_id, seq)
);
CREATE TABLE IF NOT EXISTS documents (
	id            TEXT PRIMARY KEY,
	model         TEXT NOT NULL,
	text          TEXT NOT NULL,
	metadata_json TEXT NOT NULL,
	embedding     BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_model ON documents(model);
`
	if _, err := s.db.Exec(schema); err != nil {
		return fmt.Errorf("sqlite: migrate: %w", err)
	}
	return nil
}
