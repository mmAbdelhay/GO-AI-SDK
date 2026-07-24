// Command rag demonstrates end-to-end retrieval-augmented generation using only
// this library: embed documents (OpenAI), store and search them (in-memory
// vector store), and answer with the retrieved context (Anthropic).
//
//	ANTHROPIC_API_KEY=sk-... OPENAI_API_KEY=sk-... go run ./examples/rag
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
	"github.com/mmabdelhay/go-ai-sdk/provider/openai"
	"github.com/mmabdelhay/go-ai-sdk/vector"
)

var docs = []string{
	"The warranty on the X200 vacuum covers motor failures for 5 years.",
	"The X200 vacuum's battery lasts 45 minutes on turbo mode.",
	"Our office in Cairo is open Sunday through Thursday, 9am to 5pm.",
	"Returns are accepted within 30 days with the original receipt.",
}

func main() {
	ctx := context.Background()
	anthropicKey, openaiKey := os.Getenv("ANTHROPIC_API_KEY"), os.Getenv("OPENAI_API_KEY")
	if anthropicKey == "" || openaiKey == "" {
		log.Fatal("set ANTHROPIC_API_KEY and OPENAI_API_KEY")
	}

	embedder := openai.NewEmbedder(openaiKey, "text-embedding-3-small")
	store := vector.NewInMemory()

	// Index.
	emb, err := embedder.Embed(ctx, docs)
	if err != nil {
		log.Fatalf("embed: %v", err)
	}
	for i, d := range docs {
		err = store.Upsert(ctx, emb.Model, vector.Document{
			ID: fmt.Sprintf("doc-%d", i), Text: d, Embedding: emb.Vectors[i],
		})
		if err != nil {
			log.Fatalf("upsert: %v", err)
		}
	}

	// Retrieve.
	question := "How long does the X200 battery last?"
	q, err := embedder.Embed(ctx, []string{question})
	if err != nil {
		log.Fatalf("embed query: %v", err)
	}
	matches, err := store.Query(ctx, vector.Query{
		Embedding: q.Vectors[0], Model: q.Model, TopK: 2,
	})
	if err != nil {
		log.Fatalf("query: %v", err)
	}

	// Generate with the retrieved context.
	contextText := ""
	for _, m := range matches {
		contextText += "- " + m.Document.Text + "\n"
	}
	model := anthropic.New(anthropicKey, anthropic.WithModel("claude-sonnet-4-20250514"))
	resp, err := ai.Generate(ctx, model,
		"Answer using only this context:\n"+contextText+"\nQuestion: "+question,
		ai.WithMaxTokens(200))
	if err != nil {
		log.Fatalf("generate: %v", err)
	}
	fmt.Println(resp.Text())
}
