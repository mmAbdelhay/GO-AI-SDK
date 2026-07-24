package aitest

import (
	"bytes"
	"io"
	"net/http"
	"sync"
)

// RoundTripFunc adapts a function to an http.RoundTripper.
type RoundTripFunc func(*http.Request) (*http.Response, error)

// RoundTrip implements http.RoundTripper.
func (f RoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// Transport is a fake http.RoundTripper that returns queued responses and records
// the requests it receives, letting real provider clients be tested entirely
// offline. It is safe for concurrent use.
type Transport struct {
	// Responses is a queue of canned responses returned in order. When it is
	// exhausted, RoundTrip returns the last response repeatedly.
	Responses []*http.Response
	// Err, when non-nil, is returned by every RoundTrip call.
	Err error

	mu       sync.Mutex
	requests []*recordedRequest
	next     int
}

type recordedRequest struct {
	Method string
	URL    string
	Header http.Header
	Body   []byte
}

// RoundTrip implements http.RoundTripper. Like a real transport, it fails when
// the request's context is already done.
func (t *Transport) RoundTrip(r *http.Request) (*http.Response, error) {
	if err := r.Context().Err(); err != nil {
		return nil, err
	}
	var body []byte
	if r.Body != nil {
		body, _ = io.ReadAll(r.Body)
		_ = r.Body.Close()
	}
	t.mu.Lock()
	t.requests = append(t.requests, &recordedRequest{
		Method: r.Method,
		URL:    r.URL.String(),
		Header: r.Header.Clone(),
		Body:   body,
	})
	if t.Err != nil {
		t.mu.Unlock()
		return nil, t.Err
	}
	if len(t.Responses) == 0 {
		t.mu.Unlock()
		return NewResponse(http.StatusOK, nil, ""), nil
	}
	i := t.next
	if i >= len(t.Responses) {
		i = len(t.Responses) - 1
	} else {
		t.next++
	}
	resp := t.Responses[i]
	t.mu.Unlock()
	return resp, nil
}

// RequestBodies returns the raw request bodies received, in order.
func (t *Transport) RequestBodies() [][]byte {
	t.mu.Lock()
	defer t.mu.Unlock()
	out := make([][]byte, len(t.requests))
	for i, r := range t.requests {
		out[i] = r.Body
	}
	return out
}

// LastRequestBody returns the most recent request body and whether one exists.
func (t *Transport) LastRequestBody() ([]byte, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.requests) == 0 {
		return nil, false
	}
	return t.requests[len(t.requests)-1].Body, true
}

// LastRequestURL returns the URL of the most recent request.
func (t *Transport) LastRequestURL() (string, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.requests) == 0 {
		return "", false
	}
	return t.requests[len(t.requests)-1].URL, true
}

// LastRequestHeader returns the header of the most recent request.
func (t *Transport) LastRequestHeader() (http.Header, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if len(t.requests) == 0 {
		return nil, false
	}
	return t.requests[len(t.requests)-1].Header, true
}

// Client returns an *http.Client backed by this transport.
func (t *Transport) Client() *http.Client { return &http.Client{Transport: t} }

// NewResponse builds an *http.Response with the given status, headers, and string
// body, suitable for a Transport queue.
func NewResponse(status int, header http.Header, body string) *http.Response {
	if header == nil {
		header = http.Header{}
	}
	return &http.Response{
		StatusCode: status,
		Status:     http.StatusText(status),
		Header:     header,
		Body:       io.NopCloser(bytes.NewReader([]byte(body))),
	}
}

// NewSSEResponse builds a 200 response whose body is the given server-sent-events
// text with the appropriate content type.
func NewSSEResponse(body string) *http.Response {
	h := http.Header{}
	h.Set("Content-Type", "text/event-stream")
	return NewResponse(http.StatusOK, h, body)
}
