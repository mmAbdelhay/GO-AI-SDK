package anthropic

import (
	"flag"
	"os"
	"path/filepath"
	"testing"
)

// update regenerates golden files: go test ./provider/anthropic -update
var update = flag.Bool("update", false, "update golden files")

// goldenBytes compares got against the golden file at testdata/name, updating it
// when -update is set.
func goldenBytes(t *testing.T, name string, got []byte) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if *update {
		if err := os.WriteFile(path, got, 0o644); err != nil {
			t.Fatalf("writing golden %s: %v", path, err)
		}
		return
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading golden %s: %v (run with -update to create)", path, err)
	}
	if string(got) != string(want) {
		t.Errorf("golden mismatch for %s\n--- got ---\n%s\n--- want ---\n%s", name, got, want)
	}
}

func readTestdata(t *testing.T, name string) []byte {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatalf("reading testdata %s: %v", name, err)
	}
	return b
}
