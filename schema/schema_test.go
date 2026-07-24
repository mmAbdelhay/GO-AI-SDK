package schema_test

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/mmabdelhay/go-ai-sdk/schema"
)

type address struct {
	City string `json:"city"`
	Zip  string `json:"zip,omitempty"`
}

type invoice struct {
	Number   string          `json:"number" desc:"The invoice number"`
	Status   string          `json:"status" enum:"draft,sent,paid"`
	Total    float64         `json:"total"`
	Lines    []line          `json:"lines"`
	Note     *string         `json:"note,omitempty"`
	Issued   time.Time       `json:"issued"`
	Tags     map[string]int  `json:"tags,omitempty"`
	Extra    json.RawMessage `json:"extra,omitempty"`
	Address  address         `json:"address"`
	internal string          //nolint:unused // must be skipped
	Skipped  string          `json:"-"`
}

type line struct {
	Desc string `json:"desc"`
	Qty  int    `json:"qty"`
}

func TestForInvoice(t *testing.T) {
	s, err := schema.For[invoice]()
	if err != nil {
		t.Fatal(err)
	}
	if s.Type != "object" {
		t.Fatalf("type = %q", s.Type)
	}
	if s.Properties["number"].Description != "The invoice number" {
		t.Errorf("desc tag not applied")
	}
	if got := s.Properties["status"].Enum; len(got) != 3 || got[2] != "paid" {
		t.Errorf("enum = %v", got)
	}
	if s.Properties["total"].Type != "number" || s.Properties["lines"].Type != "array" {
		t.Errorf("basic kinds wrong")
	}
	if s.Properties["lines"].Items.Properties["qty"].Type != "integer" {
		t.Errorf("nested item schema wrong")
	}
	if s.Properties["issued"].Format != "date-time" {
		t.Errorf("time.Time not date-time")
	}
	if s.Properties["tags"].AdditionalProperties.(*schema.Schema).Type != "integer" {
		t.Errorf("map value schema wrong")
	}
	if _, ok := s.Properties["Skipped"]; ok {
		t.Errorf("json:\"-\" field not skipped")
	}
	if _, ok := s.Properties["internal"]; ok {
		t.Errorf("unexported field not skipped")
	}
	// Required: everything except pointer/omitempty fields.
	req := strings.Join(s.Required, ",")
	for _, want := range []string{"number", "status", "total", "lines", "issued", "address"} {
		if !strings.Contains(req, want) {
			t.Errorf("required missing %q (got %s)", want, req)
		}
	}
	for _, notWant := range []string{"note", "tags", "extra"} {
		if strings.Contains(req, notWant) {
			t.Errorf("optional field %q marked required", notWant)
		}
	}
}

type cyclic struct {
	Next *cyclic `json:"next"`
}

func TestCyclicTypeErrors(t *testing.T) {
	if _, err := schema.For[cyclic](); err == nil {
		t.Fatal("expected error for cyclic type")
	}
}

func TestUnsupportedKindErrors(t *testing.T) {
	type bad struct {
		F func() `json:"f"`
	}
	if _, err := schema.For[bad](); err == nil {
		t.Fatal("expected error for func field")
	}
}

func TestValidate(t *testing.T) {
	s := schema.MustFor[line]()
	tests := []struct {
		name    string
		data    string
		wantErr bool
	}{
		{"valid", `{"desc":"x","qty":2}`, false},
		{"missing required", `{"desc":"x"}`, true},
		{"wrong type", `{"desc":"x","qty":"two"}`, true},
		{"non-integer", `{"desc":"x","qty":2.5}`, true},
		{"unknown key", `{"desc":"x","qty":2,"bogus":1}`, true},
		{"not an object", `[1,2]`, true},
		{"invalid json", `{`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := schema.Validate(s, []byte(tt.data))
			if (err != nil) != tt.wantErr {
				t.Errorf("Validate(%s) err = %v, wantErr %v", tt.data, err, tt.wantErr)
			}
		})
	}
}

func TestValidateEnum(t *testing.T) {
	type doc struct {
		Status string `json:"status" enum:"open,closed"`
	}
	s := schema.MustFor[doc]()
	if err := schema.Validate(s, []byte(`{"status":"open"}`)); err != nil {
		t.Errorf("valid enum rejected: %v", err)
	}
	if err := schema.Validate(s, []byte(`{"status":"other"}`)); err == nil {
		t.Errorf("invalid enum accepted")
	}
}

// FuzzValidate ensures validation never panics on arbitrary input, since the
// JSON being validated comes from a model and is untrusted.
func FuzzValidate(f *testing.F) {
	s := schema.MustFor[invoice]()
	f.Add(`{"number":"1"}`)
	f.Add(`[]`)
	f.Add(`null`)
	f.Add(`{"lines":[{"qty":true}]}`)
	f.Add(`{{{`)
	f.Fuzz(func(t *testing.T, data string) {
		_ = schema.Validate(s, []byte(data))
	})
}
