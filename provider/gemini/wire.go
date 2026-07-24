package gemini

import "encoding/json"

// Wire types for the Gemini generateContent API. Unexported: they exist only at
// the translation boundary.

type gRequest struct {
	SystemInstruction *gContent    `json:"systemInstruction,omitempty"`
	Contents          []gContent   `json:"contents"`
	GenerationConfig  *gGenConfig  `json:"generationConfig,omitempty"`
	Tools             []gTools     `json:"tools,omitempty"`
	ToolConfig        *gToolConfig `json:"toolConfig,omitempty"`
}

type gGenConfig struct {
	Temperature     *float64 `json:"temperature,omitempty"`
	TopP            *float64 `json:"topP,omitempty"`
	TopK            *int     `json:"topK,omitempty"`
	MaxOutputTokens int      `json:"maxOutputTokens,omitempty"`
	StopSequences   []string `json:"stopSequences,omitempty"`
}

type gContent struct {
	Role  string  `json:"role,omitempty"` // "user" or "model"
	Parts []gPart `json:"parts"`
}

type gPart struct {
	Text             string           `json:"text,omitempty"`
	InlineData       *gInlineData     `json:"inlineData,omitempty"`
	FileData         *gFileData       `json:"fileData,omitempty"`
	FunctionCall     *gFunctionCall   `json:"functionCall,omitempty"`
	FunctionResponse *gFunctionResult `json:"functionResponse,omitempty"`
}

type gInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64
}

type gFileData struct {
	FileURI string `json:"fileUri"`
}

type gFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type gFunctionResult struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type gTools struct {
	FunctionDeclarations []gFunctionDecl `json:"functionDeclarations"`
}

type gFunctionDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type gToolConfig struct {
	FunctionCallingConfig gFunctionCallingConfig `json:"functionCallingConfig"`
}

type gFunctionCallingConfig struct {
	Mode                 string   `json:"mode"` // AUTO, ANY, NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

type gResponse struct {
	Candidates []struct {
		Content      gContent `json:"content"`
		FinishReason string   `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata *gUsage `json:"usageMetadata"`
	ModelVersion  string  `json:"modelVersion"`
}

type gUsage struct {
	PromptTokenCount        int `json:"promptTokenCount"`
	CandidatesTokenCount    int `json:"candidatesTokenCount"`
	CachedContentTokenCount int `json:"cachedContentTokenCount"`
}

type gErrorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

type gEmbedRequest struct {
	Requests []gEmbedOne `json:"requests"`
}

type gEmbedOne struct {
	Model   string   `json:"model"`
	Content gContent `json:"content"`
}

type gEmbedResponse struct {
	Embeddings []struct {
		Values []float32 `json:"values"`
	} `json:"embeddings"`
}
