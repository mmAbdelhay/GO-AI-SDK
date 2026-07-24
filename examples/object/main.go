// Command object demonstrates structured output: extracting a typed value from
// free text via GenerateObject.
//
//	ANTHROPIC_API_KEY=sk-... go run ./examples/object
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	ai "github.com/mmabdelhay/go-ai-sdk"
	"github.com/mmabdelhay/go-ai-sdk/provider/anthropic"
)

// Invoice is the shape we want back. Field tags drive the generated JSON
// Schema: json names, desc descriptions, enum constraints; pointers and
// omitempty mark optional fields.
type Invoice struct {
	Number   string  `json:"number" desc:"The invoice number"`
	Customer string  `json:"customer" desc:"Billed customer name"`
	Total    float64 `json:"total" desc:"Grand total in the invoice currency"`
	Currency string  `json:"currency" enum:"USD,EUR,EGP"`
	DueDate  *string `json:"due_date,omitempty" desc:"Due date, YYYY-MM-DD"`
}

const email = `Hi! Please find attached invoice INV-2024-0042 for Nile Software Ltd.
The total comes to 1,250.00 EGP, due by 2024-08-15. Thanks!`

func main() {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		log.Fatal("set ANTHROPIC_API_KEY")
	}
	model := anthropic.New(key, anthropic.WithModel("claude-sonnet-4-20250514"))

	inv, err := ai.GenerateObject[Invoice](context.Background(), model,
		"Extract the invoice from this email:\n\n"+email,
		ai.WithMaxTokens(500))
	if err != nil {
		log.Fatalf("extract: %v", err)
	}

	fmt.Printf("number:   %s\ncustomer: %s\ntotal:    %.2f %s\n",
		inv.Number, inv.Customer, inv.Total, inv.Currency)
	if inv.DueDate != nil {
		fmt.Printf("due:      %s\n", *inv.DueDate)
	}
}
