package loop

import (
	"cmp"
	"context"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"sketch.dev/httprr"
	"sketch.dev/llm"
	"sketch.dev/llm/ant"
	"sketch.dev/llm/conversation"
)

// TestAgentLoop tests that the Agent loop functionality works correctly.
// It uses the httprr package to record HTTP interactions for replay in tests.
// When failing, rebuild with "go test ./sketch/loop -run TestAgentLoop -httprecord .*agent_loop.*"
// as necessary.
func TestAgentLoop(t *testing.T) {
	ctx := context.Background()

	// Setup httprr recorder
	rrPath := "testdata/agent_loop.httprr"
	rr, err := httprr.Open(rrPath, http.DefaultTransport)
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}

	if rr.Recording() {
		// Skip the test if API key is not available
		if os.Getenv("ANTHROPIC_API_KEY") == "" {
			t.Fatal("ANTHROPIC_API_KEY not set, required for HTTP recording")
		}
	}

	// Create HTTP client
	var client *http.Client
	if rr != nil {
		// Scrub API keys from requests for security
		rr.ScrubReq(func(req *http.Request) error {
			req.Header.Del("x-api-key")
			req.Header.Del("anthropic-api-key")
			req.Header.Del("Anthropic-Beta") // beta features change over time
			return nil
		})
		client = rr.Client()
	} else {
		client = &http.Client{Transport: http.DefaultTransport}
	}

	// Create a new agent with the httprr client
	origWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir("/"); err != nil {
		t.Fatal(err)
	}
	budget := conversation.Budget{MaxDollars: 10.0}
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	apiKey := cmp.Or(os.Getenv("OUTER_SKETCH_MODEL_API_KEY"), os.Getenv("ANTHROPIC_API_KEY"))
	cfg := AgentConfig{
		Context:    ctx,
		WorkingDir: wd,
		Service: &ant.Service{
			APIKey: apiKey,
			HTTPC:  client,
		},
		Budget:       budget,
		GitUsername:  "Test Agent",
		GitEmail:     "totallyhuman@sketch.dev",
		SessionID:    "test-session-id",
		ClientGOOS:   "linux",
		ClientGOARCH: "amd64",
	}
	agent := NewAgent(cfg)

	// Use fixed time for deterministic tests
	fixedTime := time.Date(2025, 7, 25, 19, 37, 57, 0, time.UTC)
	agent.now = func() time.Time { return fixedTime }

	if err := os.Chdir(origWD); err != nil {
		t.Fatal(err)
	}
	err = agent.Init(AgentInit{NoGit: true})
	if err != nil {
		t.Fatal(err)
	}

	// Setup a test message that will trigger a simple, predictable response
	userMessage := "What tools are available to you? Please just list them briefly."

	// Set a slug so that the agent doesn't have to.
	agent.SetSlug("list-available-tools")

	// Send the message to the agent
	agent.UserMessage(ctx, userMessage)

	// Process a single loop iteration to avoid long-running tests
	agent.processTurn(ctx)

	// Collect responses with a timeout
	var responses []AgentMessage
	ctx2, cancel := context.WithDeadline(ctx, time.Now().Add(10*time.Second))
	defer cancel()
	done := false
	it := agent.NewIterator(ctx2, 0)

	for !done {
		msg := it.Next()
		t.Logf("Received message: Type=%s, EndOfTurn=%v, Content=%q", msg.Type, msg.EndOfTurn, msg.Content)
		responses = append(responses, *msg)
		if msg.EndOfTurn {
			done = true
		}
	}

	// Verify we got at least one response
	if len(responses) == 0 {
		t.Fatal("No responses received from agent")
	}

	// Log the received responses for debugging
	t.Logf("Received %d responses", len(responses))

	// Find the final agent response (with EndOfTurn=true)
	var finalResponse *AgentMessage
	for i := range responses {
		if responses[i].Type == AgentMessageType && responses[i].EndOfTurn {
			finalResponse = &responses[i]
			break
		}
	}

	// Verify we got a final agent response
	if finalResponse == nil {
		t.Fatal("No final agent response received")
	}

	// Check that the response contains tools information
	if !strings.Contains(strings.ToLower(finalResponse.Content), "tool") {
		t.Error("Expected response to mention tools")
	}

	// Count how many tool use messages we received
	toolUseCount := 0
	for _, msg := range responses {
		if msg.Type == ToolUseMessageType {
			toolUseCount++
		}
	}

	t.Logf("Agent used %d tools in its response", toolUseCount)
}

func TestAgentTracksOutstandingCalls(t *testing.T) {
	agent := &Agent{
		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
		stateMachine:         NewStateMachine(),
	}

	// Check initial state
	if count := agent.OutstandingLLMCallCount(); count != 0 {
		t.Errorf("Expected 0 outstanding LLM calls, got %d", count)
	}

	if tools := agent.OutstandingToolCalls(); len(tools) != 0 {
		t.Errorf("Expected 0 outstanding tool calls, got %d", len(tools))
	}

	// Add some calls
	agent.mu.Lock()
	agent.outstandingLLMCalls["llm1"] = struct{}{}
	agent.outstandingToolCalls["tool1"] = "bash"
	agent.outstandingToolCalls["tool2"] = "think"
	agent.mu.Unlock()

	// Check tracking works
	if count := agent.OutstandingLLMCallCount(); count != 1 {
		t.Errorf("Expected 1 outstanding LLM call, got %d", count)
	}

	tools := agent.OutstandingToolCalls()
	if len(tools) != 2 {
		t.Errorf("Expected 2 outstanding tool calls, got %d", len(tools))
	}

	// Check removal
	agent.mu.Lock()
	delete(agent.outstandingLLMCalls, "llm1")
	delete(agent.outstandingToolCalls, "tool1")
	agent.mu.Unlock()

	if count := agent.OutstandingLLMCallCount(); count != 0 {
		t.Errorf("Expected 0 outstanding LLM calls after removal, got %d", count)
	}

	tools = agent.OutstandingToolCalls()
	if len(tools) != 1 {
		t.Errorf("Expected 1 outstanding tool call after removal, got %d", len(tools))
	}

	if tools[0] != "think" {
		t.Errorf("Expected 'think' tool remaining, got %s", tools[0])
	}
}

// TestAgentProcessTurnWithNilResponse tests the scenario where Agent.processTurn receives
// a nil value for initialResp from processUserMessage.
func TestAgentProcessTurnWithNilResponse(t *testing.T) {
	// Create a mock conversation that will return nil and error
	mockConvo := &MockConvoInterface{
		sendMessageFunc: func(message llm.Message) (*llm.Response, error) {
			return nil, fmt.Errorf("test error: simulating nil response")
		},
	}

	// Create a minimal Agent instance for testing
	agent := &Agent{
		convo:                mockConvo,
		inbox:                make(chan string, 10),
		subscribers:          []chan *AgentMessage{},
		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
	}

	// Create a test context
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Push a test message to the inbox so that processUserMessage will try to process it
	agent.inbox <- "Test message"

	// Call processTurn - it should exit early without panic when initialResp is nil
	agent.processTurn(ctx)

	// Verify error message was added to history
	agent.mu.Lock()
	defer agent.mu.Unlock()

	// There should be exactly two messages: slug + error
	if len(agent.history) != 2 {
		t.Errorf("Expected exactly two messages (slug + error), got %d", len(agent.history))
	} else {
		slugMsg := agent.history[0]
		if slugMsg.Type != SlugMessageType {
			t.Errorf("Expected first message to be slug, got message type: %s", slugMsg.Type)
		}
		errorMsg := agent.history[1]
		if errorMsg.Type != ErrorMessageType {
			t.Errorf("Expected second message to be error, got message type: %s", errorMsg.Type)
		}
		if !strings.Contains(errorMsg.Content, "simulating nil response") {
			t.Errorf("Expected error message to contain 'simulating nil response', got: %s", errorMsg.Content)
		}
	}
}

// MockConvoInterface implements the ConvoInterface for testing
type MockConvoInterface struct {
	sendMessageFunc              func(message llm.Message) (*llm.Response, error)
	sendUserTextMessageFunc      func(s string, otherContents ...llm.Content) (*llm.Response, error)
	toolResultContentsFunc       func(ctx context.Context, resp *llm.Response) ([]llm.Content, bool, error)
	toolResultCancelContentsFunc func(resp *llm.Response) ([]llm.Content, error)
	cancelToolUseFunc            func(toolUseID string, cause error) error
	cumulativeUsageFunc          func() conversation.CumulativeUsage
	lastUsageFunc                func() llm.Usage
	resetBudgetFunc              func(conversation.Budget)
	overBudgetFunc               func() error
	getIDFunc                    func() string
	subConvoWithHistoryFunc      func() *conversation.Convo
	debugJSONFunc                func() ([]byte, error)
}

func (m *MockConvoInterface) SendMessage(message llm.Message) (*llm.Response, error) {
	if m.sendMessageFunc != nil {
		return m.sendMessageFunc(message)
	}
	return nil, nil
}

func (m *MockConvoInterface) SendUserTextMessage(s string, otherContents ...llm.Content) (*llm.Response, error) {
	if m.sendUserTextMessageFunc != nil {
		return m.sendUserTextMessageFunc(s, otherContents...)
	}
	return nil, nil
}

func (m *MockConvoInterface) ToolResultContents(ctx context.Context, resp *llm.Response) ([]llm.Content, bool, error) {
	if m.toolResultContentsFunc != nil {
		return m.toolResultContentsFunc(ctx, resp)
	}
	return nil, false, nil
}

func (m *MockConvoInterface) ToolResultCancelContents(resp *llm.Response) ([]llm.Content, error) {
	if m.toolResultCancelContentsFunc != nil {
		return m.toolResultCancelContentsFunc(resp)
	}
	return nil, nil
}

func (m *MockConvoInterface) CancelToolUse(toolUseID string, cause error) error {
	if m.cancelToolUseFunc != nil {
		return m.cancelToolUseFunc(toolUseID, cause)
	}
	return nil
}

func (m *MockConvoInterface) CumulativeUsage() conversation.CumulativeUsage {
	if m.cumulativeUsageFunc != nil {
		return m.cumulativeUsageFunc()
	}
	return conversation.CumulativeUsage{}
}

func (m *MockConvoInterface) LastUsage() llm.Usage {
	if m.lastUsageFunc != nil {
		return m.lastUsageFunc()
	}
	return llm.Usage{}
}

func (m *MockConvoInterface) ResetBudget(budget conversation.Budget) {
	if m.resetBudgetFunc != nil {
		m.resetBudgetFunc(budget)
	}
}

func (m *MockConvoInterface) OverBudget() error {
	if m.overBudgetFunc != nil {
		return m.overBudgetFunc()
	}
	return nil
}

func (m *MockConvoInterface) GetID() string {
	if m.getIDFunc != nil {
		return m.getIDFunc()
	}
	return "mock-convo-id"
}

func (m *MockConvoInterface) SubConvoWithHistory() *conversation.Convo {
	if m.subConvoWithHistoryFunc != nil {
		return m.subConvoWithHistoryFunc()
	}
	return nil
}

func (m *MockConvoInterface) DebugJSON() ([]byte, error) {
	if m.debugJSONFunc != nil {
		return m.debugJSONFunc()
	}
	return []byte(`[{"role": "user", "content": [{"type": "text", "text": "mock conversation"}]}]`), nil
}

// TestAgentProcessTurnWithNilResponseNilError tests the scenario where Agent.processTurn receives
// a nil value for initialResp and nil error from processUserMessage.
// This test verifies that the implementation properly handles this edge case.
func TestAgentProcessTurnWithNilResponseNilError(t *testing.T) {
	// Create a mock conversation that will return nil response and nil error
	mockConvo := &MockConvoInterface{
		sendMessageFunc: func(message llm.Message) (*llm.Response, error) {
			return nil, nil // This is unusual but now handled gracefully
		},
	}

	// Create a minimal Agent instance for testing
	agent := &Agent{
		convo:                mockConvo,
		inbox:                make(chan string, 10),
		subscribers:          []chan *AgentMessage{},
		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
	}

	// Create a test context
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Push a test message to the inbox so that processUserMessage will try to process it
	agent.inbox <- "Test message"

	// Call processTurn - it should handle nil initialResp with a descriptive error
	err := agent.processTurn(ctx)

	// Verify we get the expected error
	if err == nil {
		t.Error("Expected processTurn to return an error for nil initialResp, but got nil")
	} else if !strings.Contains(err.Error(), "unexpected nil response") {
		t.Errorf("Expected error about nil response, got: %v", err)
	} else {
		t.Logf("As expected, processTurn returned error: %v", err)
	}

	// Verify error message was added to history
	agent.mu.Lock()
	defer agent.mu.Unlock()

	// There should be exactly two messages: slug + error
	if len(agent.history) != 2 {
		t.Errorf("Expected exactly two messages (slug + error), got %d", len(agent.history))
	} else {
		slugMsg := agent.history[0]
		if slugMsg.Type != SlugMessageType {
			t.Errorf("Expected first message to be slug, got message type: %s", slugMsg.Type)
		}
		errorMsg := agent.history[1]
		if errorMsg.Type != ErrorMessageType {
			t.Errorf("Expected second message to be error, got message type: %s", errorMsg.Type)
		}
		if !strings.Contains(errorMsg.Content, "unexpected nil response") {
			t.Errorf("Expected error about nil response, got: %s", errorMsg.Content)
		}
	}
}

func TestAgentStateMachine(t *testing.T) {
	// Create a simplified test for the state machine functionality
	agent := &Agent{
		stateMachine: NewStateMachine(),
	}

	// Initially the state should be Ready
	if state := agent.CurrentState(); state != StateReady {
		t.Errorf("Expected initial state to be StateReady, got %s", state)
	}

	// Test manual transitions to verify state tracking
	ctx := context.Background()

	// Track transitions
	var transitions []State
	agent.stateMachine.SetTransitionCallback(func(ctx context.Context, from, to State, event TransitionEvent) {
		transitions = append(transitions, to)
		t.Logf("State transition: %s -> %s (%s)", from, to, event.Description)
	})

	// Perform a valid sequence of transitions (based on the state machine rules)
	expectedStates := []State{
		StateWaitingForUserInput,
		StateSendingToLLM,
		StateProcessingLLMResponse,
		StateToolUseRequested,
		StateCheckingForCancellation,
		StateRunningTool,
		StateCheckingGitCommits,
		StateRunningAutoformatters,
		StateCheckingBudget,
		StateGatheringAdditionalMessages,
		StateSendingToolResults,
		StateProcessingLLMResponse,
		StateEndOfTurn,
	}

	// Manually perform each transition
	for _, state := range expectedStates {
		err := agent.stateMachine.Transition(ctx, state, "Test transition to "+state.String())
		if err != nil {
			t.Errorf("Failed to transition to %s: %v", state, err)
		}
	}

	// Check if we recorded the right number of transitions
	if len(transitions) != len(expectedStates) {
		t.Errorf("Expected %d state transitions, got %d", len(expectedStates), len(transitions))
	}

	// Check each transition matched what we expected
	for i, expected := range expectedStates {
		if i < len(transitions) {
			if transitions[i] != expected {
				t.Errorf("Transition %d: expected %s, got %s", i, expected, transitions[i])
			}
		}
	}

	// Verify the current state is the last one we transitioned to
	if state := agent.CurrentState(); state != expectedStates[len(expectedStates)-1] {
		t.Errorf("Expected current state to be %s, got %s", expectedStates[len(expectedStates)-1], state)
	}

	// Test force transition
	agent.stateMachine.ForceTransition(ctx, StateCancelled, "Testing force transition")

	// Verify current state was updated
	if state := agent.CurrentState(); state != StateCancelled {
		t.Errorf("Expected forced state to be StateCancelled, got %s", state)
	}
}

// mockConvoInterface is a mock implementation of ConvoInterface for testing
type mockConvoInterface struct {
	SendMessageFunc        func(message llm.Message) (*llm.Response, error)
	ToolResultContentsFunc func(ctx context.Context, resp *llm.Response) ([]llm.Content, bool, error)
}

func (c *mockConvoInterface) GetID() string {
	return "mockConvoInterface-id"
}

func (c *mockConvoInterface) SubConvoWithHistory() *conversation.Convo {
	return nil
}

func (m *mockConvoInterface) CumulativeUsage() conversation.CumulativeUsage {
	return conversation.CumulativeUsage{}
}

func (m *mockConvoInterface) LastUsage() llm.Usage {
	return llm.Usage{}
}

func (m *mockConvoInterface) ResetBudget(conversation.Budget) {}

func (m *mockConvoInterface) OverBudget() error {
	return nil
}

func (m *mockConvoInterface) SendMessage(message llm.Message) (*llm.Response, error) {
	if m.SendMessageFunc != nil {
		return m.SendMessageFunc(message)
	}
	return &llm.Response{StopReason: llm.StopReasonEndTurn}, nil
}

func (m *mockConvoInterface) SendUserTextMessage(s string, otherContents ...llm.Content) (*llm.Response, error) {
	return m.SendMessage(llm.UserStringMessage(s))
}

func (m *mockConvoInterface) ToolResultContents(ctx context.Context, resp *llm.Response) ([]llm.Content, bool, error) {
	if m.ToolResultContentsFunc != nil {
		return m.ToolResultContentsFunc(ctx, resp)
	}
	return []llm.Content{}, false, nil
}

func (m *mockConvoInterface) ToolResultCancelContents(resp *llm.Response) ([]llm.Content, error) {
	return []llm.Content{llm.StringContent("Tool use cancelled")}, nil
}

func (m *mockConvoInterface) CancelToolUse(toolUseID string, cause error) error {
	return nil
}

func (m *mockConvoInterface) DebugJSON() ([]byte, error) {
	return []byte(`[{"role": "user", "content": [{"type": "text", "text": "mock conversation"}]}]`), nil
}

func TestAgentProcessTurnStateTransitions(t *testing.T) {
	// Create a mock ConvoInterface for testing
	mockConvo := &mockConvoInterface{}

	// Use the testing context
	ctx := t.Context()

	// Create an agent with the state machine
	agent := &Agent{
		convo:  mockConvo,
		config: AgentConfig{Context: ctx},
		inbox:  make(chan string, 10),
		ready:  make(chan struct{}),

		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
		stateMachine:         NewStateMachine(),
		startOfTurn:          time.Now(),
		subscribers:          []chan *AgentMessage{},
	}

	// Verify initial state
	if state := agent.CurrentState(); state != StateReady {
		t.Errorf("Expected initial state to be StateReady, got %s", state)
	}

	// Add a message to the inbox so we don't block in GatherMessages
	agent.inbox <- "Test message"

	// Setup the mock to simulate a model response with end of turn
	mockConvo.SendMessageFunc = func(message llm.Message) (*llm.Response, error) {
		return &llm.Response{
			StopReason: llm.StopReasonEndTurn,
			Content: []llm.Content{
				llm.StringContent("This is a test response"),
			},
		}, nil
	}

	// Track state transitions
	var transitions []State
	agent.stateMachine.SetTransitionCallback(func(ctx context.Context, from, to State, event TransitionEvent) {
		transitions = append(transitions, to)
		t.Logf("State transition: %s -> %s (%s)", from, to, event.Description)
	})

	// Process a turn, which should trigger state transitions
	agent.processTurn(ctx)

	// The minimum expected states for a simple end-of-turn response
	minExpectedStates := []State{
		StateWaitingForUserInput,
		StateSendingToLLM,
		StateProcessingLLMResponse,
		StateEndOfTurn,
	}

	// Verify we have at least the minimum expected states
	if len(transitions) < len(minExpectedStates) {
		t.Errorf("Expected at least %d state transitions, got %d", len(minExpectedStates), len(transitions))
	}

	// Check that the transitions follow the expected sequence
	for i, expected := range minExpectedStates {
		if i < len(transitions) {
			if transitions[i] != expected {
				t.Errorf("Transition %d: expected %s, got %s", i, expected, transitions[i])
			}
		}
	}

	// Verify the final state is EndOfTurn
	if state := agent.CurrentState(); state != StateEndOfTurn {
		t.Errorf("Expected final state to be StateEndOfTurn, got %s", state)
	}
}

func TestAgentProcessTurnWithToolUse(t *testing.T) {
	// Create a mock ConvoInterface for testing
	mockConvo := &mockConvoInterface{}

	// Setup a test context
	ctx := context.Background()

	// Create an agent with the state machine
	agent := &Agent{
		convo:  mockConvo,
		config: AgentConfig{Context: ctx},
		inbox:  make(chan string, 10),
		ready:  make(chan struct{}),

		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
		stateMachine:         NewStateMachine(),
		startOfTurn:          time.Now(),
		subscribers:          []chan *AgentMessage{},
	}

	// Add a message to the inbox so we don't block in GatherMessages
	agent.inbox <- "Test message"

	// First response requests a tool
	firstResponseDone := false
	mockConvo.SendMessageFunc = func(message llm.Message) (*llm.Response, error) {
		if !firstResponseDone {
			firstResponseDone = true
			return &llm.Response{
				StopReason: llm.StopReasonToolUse,
				Content: []llm.Content{
					llm.StringContent("I'll use a tool"),
					{Type: llm.ContentTypeToolUse, ToolName: "test_tool", ToolInput: []byte("{}"), ID: "test_id"},
				},
			}, nil
		}
		// Second response ends the turn
		return &llm.Response{
			StopReason: llm.StopReasonEndTurn,
			Content: []llm.Content{
				llm.StringContent("Finished using the tool"),
			},
		}, nil
	}

	// Tool result content handler
	mockConvo.ToolResultContentsFunc = func(ctx context.Context, resp *llm.Response) ([]llm.Content, bool, error) {
		return []llm.Content{llm.StringContent("Tool executed successfully")}, false, nil
	}

	// Track state transitions
	var transitions []State
	agent.stateMachine.SetTransitionCallback(func(ctx context.Context, from, to State, event TransitionEvent) {
		transitions = append(transitions, to)
		t.Logf("State transition: %s -> %s (%s)", from, to, event.Description)
	})

	// Process a turn with tool use
	agent.processTurn(ctx)

	// Define expected states for a tool use flow
	expectedToolStates := []State{
		StateWaitingForUserInput,
		StateSendingToLLM,
		StateProcessingLLMResponse,
		StateToolUseRequested,
		StateCheckingForCancellation,
		StateRunningTool,
	}

	// Verify that these states are present in order
	for i, expectedState := range expectedToolStates {
		if i >= len(transitions) {
			t.Errorf("Missing expected transition to %s; only got %d transitions", expectedState, len(transitions))
			continue
		}
		if transitions[i] != expectedState {
			t.Errorf("Expected transition %d to be %s, got %s", i, expectedState, transitions[i])
		}
	}

	// Also verify we eventually reached EndOfTurn
	if !slices.Contains(transitions, StateEndOfTurn) {
		t.Errorf("Expected to eventually reach StateEndOfTurn, but never did")
	}
}

func TestPushToOutbox(t *testing.T) {
	// Create a new agent
	a := &Agent{
		outstandingLLMCalls:  make(map[string]struct{}),
		outstandingToolCalls: make(map[string]string),
		stateMachine:         NewStateMachine(),
		subscribers:          make([]chan *AgentMessage, 0),
	}

	// Create a channel to receive messages
	messageCh := make(chan *AgentMessage, 1)

	// Add the channel to the subscribers list
	a.mu.Lock()
	a.subscribers = append(a.subscribers, messageCh)
	a.mu.Unlock()

	// We need to set the text that would be produced by our modified contentToString function
	resultText := "test resultnested result" // Directly set the expected output

	// In a real-world scenario, this would be coming from a toolResult that contained nested content

	m := AgentMessage{
		Type:       ToolUseMessageType,
		ToolResult: resultText,
	}

	// Push the message to the outbox
	a.pushToOutbox(context.Background(), m)

	// Receive the message from the subscriber
	received := <-messageCh

	// Check that the Content field contains the concatenated text from ToolResult
	expected := "test resultnested result"
	if received.Content != expected {
		t.Errorf("Expected Content to be %q, got %q", expected, received.Content)
	}
}

// TestCleanSlugName tests the slug cleaning function
func TestCleanSlugName(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{"simple lowercase", "fix-bug", "fix-bug"},
		{"uppercase to lowercase", "FIX-BUG", "fix-bug"},
		{"spaces to hyphens", "fix login bug", "fix-login-bug"},
		{"mixed case and spaces", "Fix Login Bug", "fix-login-bug"},
		{"special characters removed", "fix_bug@home!", "fixbughome"},
		{"multiple hyphens truncated", "fix--bug---here", "fix--bug-"},
		{"leading/trailing hyphens preserved", "-fix-bug-", "-fix-bug-"},
		{"truncate to 4 words", "fix-login-bug-here-now", "fix-login-bug-here"},
		{"truncate at 64 bytes", "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123", "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz01"},
		{"64 bytes then 4 words", "a-b-c-d-e-fghijklmnopqrstuvwxyz0123456789abcdefghijklm", "a-b-c-d"},
		{"4 words under 64 bytes", "short-slug-here-but-more-text", "short-slug-here-but"},
		{"numbers preserved", "fix-bug-v2", "fix-bug-v2"},
		{"empty string", "", ""},
		{"only special chars", "@#$%", ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := cleanSlugName(tt.input)
			if got != tt.want {
				t.Errorf("cleanSlugName(%q) = %q, want %q", tt.input, got, tt.want)
			}
		})
	}
}

// TestAutoGenerateSlugInputValidation tests input validation for auto slug generation
func TestAutoGenerateSlugInputValidation(t *testing.T) {
	// Test soleText with empty input
	emptyContents := []llm.Content{}
	_, err := soleText(emptyContents)
	if err == nil {
		t.Errorf("Expected error for empty contents, got nil")
	}

	// Test with non-text content only
	nonTextContents := []llm.Content{
		{Type: llm.ContentTypeToolUse, ToolName: "bash"},
	}
	_, err = soleText(nonTextContents)
	if err == nil {
		t.Errorf("Expected error for non-text contents, got nil")
	}

	// Test slug formatting
	testInputs := []string{
		"Fix the login bug",
		"Add user authentication system",
		"Refactor API endpoints",
		"Update documentation",
	}

	for _, input := range testInputs {
		slug := cleanSlugName(strings.ToLower(strings.ReplaceAll(input, " ", "-")))
		if slug == "" {
			t.Errorf("cleanSlugName produced empty result for input %q", input)
		}
		if !strings.Contains(slug, "-") {
			// We expect most multi-word inputs to contain hyphens after processing
			t.Logf("Input %q produced slug %q (no hyphen found, might be single word)", input, slug)
		}
	}
}

// TestSoleText tests the soleText helper function
func TestSoleText(t *testing.T) {
	tests := []struct {
		name     string
		contents []llm.Content
		wantText string
		wantErr  bool
	}{
		{
			name: "single text content",
			contents: []llm.Content{
				{Type: llm.ContentTypeText, Text: "  Hello world  "},
			},
			wantText: "Hello world",
			wantErr:  false,
		},
		{
			name:     "empty slice",
			contents: []llm.Content{},
			wantText: "",
			wantErr:  true,
		},
		{
			name: "multiple contents",
			contents: []llm.Content{
				{Type: llm.ContentTypeText, Text: "First"},
				{Type: llm.ContentTypeText, Text: "Second"},
			},
			wantText: "",
			wantErr:  true,
		},
		{
			name: "non-text content",
			contents: []llm.Content{
				{Type: llm.ContentTypeToolUse, ToolName: "bash"},
			},
			wantText: "",
			wantErr:  true,
		},
		{
			name: "empty text content",
			contents: []llm.Content{
				{Type: llm.ContentTypeText, Text: ""},
			},
			wantText: "",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotText, err := soleText(tt.contents)
			if (err != nil) != tt.wantErr {
				t.Errorf("soleText() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotText != tt.wantText {
				t.Errorf("soleText() gotText = %v, want %v", gotText, tt.wantText)
			}
		})
	}
}

// TestSystemPromptIncludesDateTime tests that the system prompt includes current date/time
func TestSystemPromptIncludesDateTime(t *testing.T) {
	ctx := context.Background()

	// Create a minimal agent config for testing
	config := AgentConfig{
		Context:      ctx,
		ClientGOOS:   "linux",
		ClientGOARCH: "amd64",
	}

	// Create agent
	agent := NewAgent(config)

	// Use fixed time for deterministic tests
	fixedTime := time.Date(2025, 7, 25, 19, 37, 57, 0, time.UTC)
	agent.now = func() time.Time { return fixedTime }

	// Set minimal required fields for rendering
	agent.workingDir = "/tmp"
	agent.repoRoot = "/tmp"

	// Mock SketchGitBase to return a valid commit hash
	// We'll override this by setting a method that returns a fixed value
	// Since we can't easily mock the git calls, we'll work around it

	// Render the system prompt
	systemPrompt := agent.renderSystemPrompt()

	// Check that the system prompt contains a current_date section
	if !strings.Contains(systemPrompt, "<current_date>") {
		t.Error("System prompt should contain <current_date> section")
	}

	// Check that it contains what looks like a date
	// The format is "2006-01-02" (time.DateOnly)
	if !strings.Contains(systemPrompt, "-") {
		t.Error("System prompt should contain a formatted date")
	}

	// Verify the expected fixed date (2025-07-25)
	expectedDate := "2025-07-25"
	if !strings.Contains(systemPrompt, expectedDate) {
		t.Errorf("System prompt should contain expected fixed date %s", expectedDate)
	}

	// Print part of the system prompt for manual verification in test output
	// Find the current_date section
	start := strings.Index(systemPrompt, "<current_date>")
	if start != -1 {
		end := strings.Index(systemPrompt[start:], "</current_date>") + start
		if end > start {
			dateSection := systemPrompt[start : end+len("</current_date>")]
			t.Logf("Date section in system prompt: %s", dateSection)
		}
	}
}
