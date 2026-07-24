package schema

import (
	"encoding/json"
	"fmt"
	"math"
	"slices"
)

// Validate checks that data (a JSON document) conforms to s. Errors describe the
// failing path so they can be fed back to a model as repair instructions.
func Validate(s *Schema, data []byte) error {
	var v any
	if err := json.Unmarshal(data, &v); err != nil {
		return fmt.Errorf("invalid JSON: %w", err)
	}
	return validate(s, v, "$")
}

func validate(s *Schema, v any, path string) error {
	if s == nil || (s.Type == "" && len(s.Enum) == 0) {
		return nil // unconstrained
	}
	if v == nil {
		// null is tolerated anywhere; requiredness is enforced by the parent
		// object's required list, which a null-valued key satisfies poorly but
		// unambiguously — reject it for typed schemas.
		if s.Type != "" {
			return fmt.Errorf("%s: expected %s, got null", path, s.Type)
		}
		return nil
	}
	switch s.Type {
	case "object":
		m, ok := v.(map[string]any)
		if !ok {
			return fmt.Errorf("%s: expected object, got %T", path, v)
		}
		for _, req := range s.Required {
			if _, ok := m[req]; !ok {
				return fmt.Errorf("%s: missing required property %q", path, req)
			}
		}
		for k, val := range m {
			if ps, ok := s.Properties[k]; ok {
				if err := validate(ps, val, path+"."+k); err != nil {
					return err
				}
				continue
			}
			switch ap := s.AdditionalProperties.(type) {
			case bool:
				if !ap {
					return fmt.Errorf("%s: unexpected property %q", path, k)
				}
			case *Schema:
				if err := validate(ap, val, path+"."+k); err != nil {
					return err
				}
			}
		}
	case "array":
		arr, ok := v.([]any)
		if !ok {
			return fmt.Errorf("%s: expected array, got %T", path, v)
		}
		for i, item := range arr {
			if err := validate(s.Items, item, fmt.Sprintf("%s[%d]", path, i)); err != nil {
				return err
			}
		}
	case "string":
		str, ok := v.(string)
		if !ok {
			return fmt.Errorf("%s: expected string, got %T", path, v)
		}
		if len(s.Enum) > 0 && !slices.Contains(s.Enum, str) {
			return fmt.Errorf("%s: %q is not one of %v", path, str, s.Enum)
		}
	case "integer":
		f, ok := v.(float64)
		if !ok || f != math.Trunc(f) || math.IsInf(f, 0) {
			return fmt.Errorf("%s: expected integer, got %v", path, v)
		}
	case "number":
		if _, ok := v.(float64); !ok {
			return fmt.Errorf("%s: expected number, got %T", path, v)
		}
	case "boolean":
		if _, ok := v.(bool); !ok {
			return fmt.Errorf("%s: expected boolean, got %T", path, v)
		}
	}
	return nil
}
