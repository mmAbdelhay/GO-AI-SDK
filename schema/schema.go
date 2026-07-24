// Package schema derives JSON Schema documents from Go types via reflection and
// struct tags, and validates JSON values against them. It powers structured
// output (ai.GenerateObject) and tool input schemas (package tool).
//
// Field names come from `json` tags. A field is required unless it is a pointer
// or its json tag carries omitempty. Two additional tags are recognized:
//
//	type Invoice struct {
//		Number string  `json:"number" desc:"The invoice number"`
//		Status string  `json:"status" enum:"draft,sent,paid"`
//		Note   *string `json:"note,omitempty"` // optional
//	}
package schema

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"time"
)

// Schema is a JSON Schema document (a pragmatic subset of draft 2020-12
// sufficient for describing model inputs and outputs).
type Schema struct {
	Type        string             `json:"type,omitempty"`
	Format      string             `json:"format,omitempty"`
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	Items       *Schema            `json:"items,omitempty"`
	Enum        []string           `json:"enum,omitempty"`
	// AdditionalProperties is false for derived structs (unknown keys rejected),
	// a *Schema for maps, and nil (unconstrained) otherwise.
	AdditionalProperties any `json:"additionalProperties,omitempty"`
}

// JSON returns the schema marshaled as JSON.
func (s *Schema) JSON() json.RawMessage {
	b, err := json.Marshal(s)
	if err != nil {
		// Schema contains only marshalable types; this is unreachable in
		// practice but kept honest.
		panic(fmt.Sprintf("schema: marshal: %v", err))
	}
	return b
}

// For derives the JSON Schema for type T.
func For[T any]() (*Schema, error) {
	return Of(reflect.TypeOf((*T)(nil)).Elem())
}

// MustFor is like [For] but panics on error. Intended for package-level
// variables and tests where the type is known valid.
func MustFor[T any]() *Schema {
	s, err := For[T]()
	if err != nil {
		panic(err)
	}
	return s
}

// Of derives the JSON Schema for the given reflect.Type.
func Of(t reflect.Type) (*Schema, error) {
	return of(t, make(map[reflect.Type]bool))
}

var (
	timeType    = reflect.TypeOf(time.Time{})
	rawMsgType  = reflect.TypeOf(json.RawMessage{})
	anySchema   = func() *Schema { return &Schema{} }
	errCyclical = fmt.Errorf("schema: cyclic type not supported")
)

func of(t reflect.Type, seen map[reflect.Type]bool) (*Schema, error) {
	switch t.Kind() {
	case reflect.Pointer:
		return of(t.Elem(), seen)
	case reflect.Bool:
		return &Schema{Type: "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return &Schema{Type: "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return &Schema{Type: "number"}, nil
	case reflect.String:
		return &Schema{Type: "string"}, nil
	case reflect.Interface:
		return anySchema(), nil
	case reflect.Slice, reflect.Array:
		if t == rawMsgType {
			return anySchema(), nil
		}
		if t.Elem().Kind() == reflect.Uint8 {
			// encoding/json represents byte slices as base64 strings.
			return &Schema{Type: "string"}, nil
		}
		items, err := of(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return &Schema{Type: "array", Items: items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("schema: map key must be string, got %s", t.Key())
		}
		vs, err := of(t.Elem(), seen)
		if err != nil {
			return nil, err
		}
		return &Schema{Type: "object", AdditionalProperties: vs}, nil
	case reflect.Struct:
		if t == timeType {
			return &Schema{Type: "string", Format: "date-time"}, nil
		}
		if seen[t] {
			return nil, fmt.Errorf("%w: %s", errCyclical, t)
		}
		seen[t] = true
		defer delete(seen, t)
		return ofStruct(t, seen)
	default:
		return nil, fmt.Errorf("schema: unsupported kind %s (%s)", t.Kind(), t)
	}
}

func ofStruct(t reflect.Type, seen map[reflect.Type]bool) (*Schema, error) {
	s := &Schema{
		Type:                 "object",
		Properties:           map[string]*Schema{},
		AdditionalProperties: false,
	}
	for _, f := range reflect.VisibleFields(t) {
		if f.PkgPath != "" || f.Anonymous {
			continue // unexported, or the embedded container itself
		}
		name, omitempty, skip := jsonName(f)
		if skip {
			continue
		}
		fs, err := of(f.Type, seen)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t.Name(), f.Name, err)
		}
		if d := f.Tag.Get("desc"); d != "" {
			fs.Description = d
		}
		if e := f.Tag.Get("enum"); e != "" {
			fs.Enum = strings.Split(e, ",")
		}
		s.Properties[name] = fs
		if f.Type.Kind() != reflect.Pointer && !omitempty {
			s.Required = append(s.Required, name)
		}
	}
	return s, nil
}

func jsonName(f reflect.StructField) (name string, omitempty, skip bool) {
	tag := f.Tag.Get("json")
	if tag == "-" {
		return "", false, true
	}
	name = f.Name
	parts := strings.Split(tag, ",")
	if parts[0] != "" {
		name = parts[0]
	}
	for _, p := range parts[1:] {
		if p == "omitempty" {
			omitempty = true
		}
	}
	return name, omitempty, false
}
