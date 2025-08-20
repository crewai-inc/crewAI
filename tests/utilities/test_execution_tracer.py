#!/usr/bin/env python3
"""
Tests for CrewAI Execution Tracer
=================================

Unit tests validating the execution tracer functionality without requiring
external LLM API calls. Tests focus on the core tracing logic and data structures.
"""

from unittest.mock import Mock

import pytest

from crewai.utilities.execution_tracer import (
    ExecutionStep,
    ExecutionTrace,
    ExecutionTracer,
    StepType,
)


def test_step_type_enum():
    """Test StepType enum values"""
    assert StepType.HUMAN_MESSAGE.value == "HumanMessage"
    assert StepType.AI_MESSAGE.value == "AIMessage"
    assert StepType.TOOL_CALL.value == "ToolCall"
    assert StepType.TOOL_MESSAGE.value == "ToolMessage"
    assert StepType.TASK_COMPLETE.value == "TaskComplete"
    assert StepType.CREW_START.value == "CrewStart"
    assert StepType.CREW_COMPLETE.value == "CrewComplete"


def test_execution_step_creation():
    """Test ExecutionStep creation and serialization"""
    metadata = {"tool": "calculator", "input": "2+2"}
    step = ExecutionStep(StepType.TOOL_CALL, "Agent1", "Calculate 2+2", metadata)
    
    assert step.step_type.value == StepType.TOOL_CALL.value
    assert step.agent_name == "Agent1"
    assert step.content == "Calculate 2+2"
    assert step.metadata == metadata
    assert isinstance(step.timestamp, float)
    
    # Test serialization
    step_dict = step.to_dict()
    assert step_dict["step_type"] == "ToolCall"
    assert step_dict["agent"] == "Agent1"
    assert step_dict["content"] == "Calculate 2+2"
    assert step_dict["metadata"] == metadata
    assert "time" in step_dict
    assert "timestamp" in step_dict


def test_execution_trace_basic():
    """Test ExecutionTrace basic functionality"""
    trace = ExecutionTrace()
    
    # Initially empty
    assert len(trace.steps) == 0
    assert trace.actions() == []
    
    # Add a step
    step = trace.add_step(StepType.AI_MESSAGE, "Assistant", "Hello world")
    assert len(trace.steps) == 1
    assert step.step_type.value == StepType.AI_MESSAGE.value
    
    # Test actions method
    actions = trace.actions()
    assert len(actions) == 1
    assert actions[0]["step_type"] == "AIMessage"
    assert actions[0]["agent"] == "Assistant"


def test_execution_trace_logs():
    """Test ExecutionTrace logs categorization"""
    trace = ExecutionTrace()
    
    # Add various step types
    trace.add_step(StepType.CREW_START, "System", "Starting crew")
    trace.add_step(StepType.AI_MESSAGE, "Agent1", "I'll help you")
    trace.add_step(StepType.TOOL_CALL, "Agent1", "Using calculator")
    trace.add_step(StepType.TOOL_MESSAGE, "System", "Result: 42")
    trace.add_step(StepType.TASK_COMPLETE, "Agent1", "Task done")
    trace.add_step(StepType.CREW_COMPLETE, "System", "Crew finished")
    
    logs = trace.logs()
    
    # Check structure
    assert "total_steps" in logs
    assert "execution_time" in logs
    assert "step_types" in logs
    assert "logs_by_type" in logs
    assert "chronological_sequence" in logs
    
    # Check values
    assert logs["total_steps"] == 6
    assert isinstance(logs["execution_time"], float)
    
    # Check categorization
    step_types = logs["step_types"]
    assert "CrewStart" in step_types
    assert "AIMessage" in step_types
    assert "ToolCall" in step_types
    assert "ToolMessage" in step_types
    assert "TaskComplete" in step_types
    assert "CrewComplete" in step_types
    
    # Check logs by type
    logs_by_type = logs["logs_by_type"]
    assert len(logs_by_type["AIMessage"]) == 1
    assert len(logs_by_type["ToolCall"]) == 1
    assert len(logs_by_type["ToolMessage"]) == 1


def test_execution_tracer_step_categorization():
    """Test ExecutionTracer step categorization logic"""
    tracer = ExecutionTracer()
    
    # Test ToolMessage categorization
    tool_result = Mock()
    tool_result.result = "42"
    tool_result.result_as_answer = True
    step_type = tracer._categorize_step("Tool returned: 42", tool_result)
    assert step_type.value == StepType.TOOL_MESSAGE.value
    
    # Test ToolCall categorization
    agent_action = Mock(spec=[])  # No result/result_as_answer attributes
    agent_action.thought = "I need to calculate"
    agent_action.tool = "calculator"
    agent_action.tool_input = "2+2"
    step_type = tracer._categorize_step("Using calculator", agent_action)
    assert step_type.value == StepType.TOOL_CALL.value
    
    # Test AIMessage categorization (AgentAction without tool)
    agent_action_no_tool = Mock(spec=[])  # No result/result_as_answer attributes
    agent_action_no_tool.thought = "Let me think"
    agent_action_no_tool.tool = ""
    agent_action_no_tool.tool_input = {}
    step_type = tracer._categorize_step("Thinking...", agent_action_no_tool)
    assert step_type.value == StepType.AI_MESSAGE.value
    
    # Test HumanMessage categorization
    human_input = Mock(spec=[])  # No result/result_as_answer attributes
    step_type = tracer._categorize_step("Task: Calculate 2+2", human_input)
    assert step_type.value == StepType.HUMAN_MESSAGE.value
    
    # Test default AIMessage categorization
    generic_output = Mock(spec=[])  # No result/result_as_answer/thought attributes
    step_type = tracer._categorize_step("Some response", generic_output)
    assert step_type.value == StepType.AI_MESSAGE.value


def test_execution_tracer_content_extraction():
    """Test ExecutionTracer content extraction"""
    tracer = ExecutionTracer()
    
    # Test ToolResult content extraction
    tool_result = Mock()
    tool_result.result = "The answer is 42"
    tool_result.result_as_answer = True
    content = tracer._extract_content(tool_result)
    assert "Tool returned: The answer is 42" in content
    
    # Test AgentAction text extraction
    agent_action = Mock(spec=[])  # No result/result_as_answer attributes
    agent_action.text = "I'll solve this problem"
    agent_action.thought = "Let me think about this"
    content = tracer._extract_content(agent_action)
    assert content == "I'll solve this problem"
    
    # Test AgentAction thought extraction (when no text)
    agent_action_no_text = Mock(spec=[])  # Mock with no text attribute
    agent_action_no_text.thought = "Just thinking"
    content = tracer._extract_content(agent_action_no_text)
    assert content == "Just thinking"
    
    # Test generic string conversion
    generic_output = Mock()
    content = tracer._extract_content(generic_output)
    assert isinstance(content, str)


def test_execution_tracer_metadata_extraction():
    """Test ExecutionTracer metadata extraction"""
    tracer = ExecutionTracer()
    
    # Test tool-related metadata
    agent_output = Mock()
    agent_output.tool = "calculator"
    agent_output.tool_input = {"operation": "2+2"}
    agent_output.result_as_answer = True
    
    metadata = tracer._extract_metadata(agent_output)
    assert metadata["tool"] == "calculator"
    assert metadata["tool_input"] == {"operation": "2+2"}
    assert metadata["result_as_answer"]
    
    # Test object without metadata
    simple_output = Mock(spec=[])
    metadata = tracer._extract_metadata(simple_output)
    assert metadata == {}


def test_execution_tracer_agent_name_extraction():
    """Test ExecutionTracer agent name extraction"""
    tracer = ExecutionTracer()
    
    # Test agent extraction from output
    agent_output = Mock()
    agent_output.agent = Mock()
    agent_output.agent.role = "Calculator"
    agent_name = tracer._extract_agent_name(agent_output)
    assert agent_name == "Calculator"
    
    # Test fallback to current_agent
    tracer.current_agent = Mock()
    tracer.current_agent.role = "DefaultAgent"
    simple_output = Mock(spec=[])
    agent_name = tracer._extract_agent_name(simple_output)
    assert agent_name == "DefaultAgent"
    
    # Test fallback to Unknown
    tracer.current_agent = None
    agent_name = tracer._extract_agent_name(simple_output)
    assert agent_name == "Unknown"


def test_execution_tracer_callbacks():
    """Test ExecutionTracer callback methods"""
    tracer = ExecutionTracer()
    
    # Test on_crew_start
    inputs = {"input": "test"}
    result = tracer.on_crew_start(inputs)
    assert result == inputs
    assert len(tracer.trace.steps) == 1
    assert tracer.trace.steps[0].step_type.value == StepType.CREW_START.value
    
    # Test on_step_complete
    agent_output = Mock()
    agent_output.text = "I'm working on it"
    tracer.current_agent = Mock()
    tracer.current_agent.role = "Worker"
    
    result = tracer.on_step_complete(agent_output)
    assert result == agent_output  # Should return the input unchanged
    assert len(tracer.trace.steps) == 2  # Added one more step
    
    # Check captured metadata
    step = tracer.trace.steps[1]
    assert step.metadata["raw_output_type"] == "Mock"
    assert "task" in step.metadata
    
    # Test on_task_complete
    task_output = Mock()
    task_output.agent = "Worker"
    task_output.description = "Calculate something"
    task_output.raw = "Result: 42"
    
    result = tracer.on_task_complete(task_output)
    assert result == task_output
    assert len(tracer.trace.steps) == 3
    
    # Test on_crew_complete
    crew_output = Mock()
    crew_output.__dict__ = {}
    
    result = tracer.on_crew_complete(crew_output)
    assert result == crew_output
    assert len(tracer.trace.steps) == 4  # Added crew complete step
    
    # Check that trace data was added to crew_output
    assert hasattr(crew_output, 'execution_steps')
    assert hasattr(crew_output, 'interaction_logs')
    assert hasattr(crew_output, 'tracing_enabled')
    assert crew_output.tracing_enabled
    assert len(crew_output.execution_steps) == 4
    assert crew_output.interaction_logs["total_steps"] == 4


def test_execution_tracer_content_length_limiting():
    """Test that long content is properly limited"""
    tracer = ExecutionTracer()
    tracer.current_agent = Mock()
    tracer.current_agent.role = "TestAgent"
    
    # Create output with very long content
    long_content = "x" * 1000  # 1000 characters
    agent_output = Mock(spec=[])  # No result/result_as_answer attributes
    agent_output.text = long_content
    
    tracer.on_step_complete(agent_output)
    
    # Check that content was limited to 500 characters
    step = tracer.trace.steps[0]
    assert len(step.content) == 500
    assert step.content == "x" * 500


def test_execution_tracer_with_none_inputs():
    """Test ExecutionTracer handles None inputs gracefully"""
    tracer = ExecutionTracer()
    
    # Test on_crew_start with None
    result = tracer.on_crew_start(None)
    assert result == {}
    
    # Test on_crew_complete with object without __dict__
    crew_output = "string_output"  # No __dict__ attribute
    result = tracer.on_crew_complete(crew_output)
    assert result == crew_output
    # Should still add the completion step
    assert len(tracer.trace.steps) == 2  # start + complete


if __name__ == "__main__":
    pytest.main([__file__, "-v"])