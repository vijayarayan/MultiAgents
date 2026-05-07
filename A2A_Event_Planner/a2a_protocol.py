"""
Minimal A2A (Agent-to-Agent) protocol types and helpers.

Implements the core subset of the A2A spec:
  - Agent Cards   (discovery)
  - Tasks         (unit of work)
  - Messages      (communication within tasks)
  - Parts         (text + structured data within messages)
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Agent Card — describes an agent's identity and capabilities
# ---------------------------------------------------------------------------

class AgentSkill(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str] = []
    examples: list[str] = []


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str
    url: str
    version: str = "1.0"
    capabilities: AgentCapabilities = AgentCapabilities()
    skills: list[AgentSkill] = []


# ---------------------------------------------------------------------------
# Message Parts — content blocks within a message
# ---------------------------------------------------------------------------

class TextPart(BaseModel):
    type: str = "text"
    text: str


class DataPart(BaseModel):
    type: str = "data"
    data: dict[str, Any]


Part = TextPart | DataPart


# ---------------------------------------------------------------------------
# Message — a single message exchanged within a task
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str  # "user" or "agent"
    parts: list[Part]


# ---------------------------------------------------------------------------
# Task — a unit of work sent between agents
# ---------------------------------------------------------------------------

class TaskState(str, Enum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(BaseModel):
    state: TaskState


class Task(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus(state=TaskState.SUBMITTED)
    messages: list[Message] = []


# ---------------------------------------------------------------------------
# Request / Response wrappers for the A2A HTTP API
# ---------------------------------------------------------------------------

class TaskSendRequest(BaseModel):
    """POST /a2a/tasks/send — send a new task or continue an existing one."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: Message


class TaskSendResponse(BaseModel):
    """Response from /a2a/tasks/send."""
    id: str
    status: TaskStatus
    messages: list[Message] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_user_message(text: str = "", data: dict | None = None) -> Message:
    """Build a user message with optional text and structured data parts."""
    parts: list[Part] = []
    if text:
        parts.append(TextPart(text=text))
    if data:
        parts.append(DataPart(data=data))
    return Message(role="user", parts=parts)


def make_agent_message(text: str = "", data: dict | None = None) -> Message:
    """Build an agent response message with optional text and data."""
    parts: list[Part] = []
    if text:
        parts.append(TextPart(text=text))
    if data:
        parts.append(DataPart(data=data))
    return Message(role="agent", parts=parts)


def extract_text(message: Message) -> str:
    """Extract concatenated text from all TextParts in a message."""
    return "\n".join(p.text for p in message.parts if isinstance(p, TextPart))


def extract_data(message: Message) -> dict:
    """Extract merged data from all DataParts in a message."""
    merged: dict[str, Any] = {}
    for p in message.parts:
        if isinstance(p, DataPart):
            merged.update(p.data)
    return merged
