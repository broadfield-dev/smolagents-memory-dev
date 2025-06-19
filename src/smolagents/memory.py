import imemory
import json
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, TypedDict

# Load environment variables for imemory
from dotenv import load_dotenv
load_dotenv()

from smolagents.models import ChatMessage, MessageRole
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


logger = getLogger(__name__)


class Message(TypedDict):
    role: MessageRole
    content: str | list[dict[str, Any]]


@dataclass
class ToolCall:
    name: str
    arguments: Any
    id: str

    def dict(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    def dict(self):
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    step_number: int
    timing: Timing
    model_input_messages: list[Message] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: ChatMessage | None = None
    model_output: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self):
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": self.model_input_messages,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": self.model_output_message.dict() if self.model_output_message else None,
            "model_output": self.model_output,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                Message(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                Message(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                Message(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    model_input_messages: list[Message]
    model_output_message: ChatMessage
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        if summary_mode:
            return []
        return [
            Message(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            Message(role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]),
            # This second message creates a role change to prevent models models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            for image in self.task_images:
                content.append({"type": "image", "image": image})

        return [Message(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list[Message]:
        if summary_mode:
            return []
        return [Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    output: Any


class AgentMemory:
    def __init__(self, system_prompt: str):
        self.system_prompt = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []
        
        # --- iMemory Integration: Initialization ---
        logger.info("Initializing persistent memory system...")
        imemory.initialize_memory_system()
        logger.info("Persistent memory system initialized.")

    def add_step(self, step: MemoryStep):
        """
        Adds a new step to the agent's short-term memory and saves
        completed interaction turns to long-term persistent memory.
        """
        self.steps.append(step)
        
        # --- iMemory Integration: Save Hook ---
        # If the last step was a user task and this step is the bot's action/plan,
        # it forms a complete "turn" that we can save.
        if len(self.steps) > 1:
            previous_step = self.steps[-2]
            current_step = self.steps[-1]
            
            # We define a "turn" as a TaskStep (user input) followed by the agent's response.
            if isinstance(previous_step, TaskStep) and isinstance(current_step, (ActionStep, PlanningStep)):
                user_input_str = json.dumps(previous_step.dict())
                bot_response_str = json.dumps(current_step.dict())
                
                logger.info("Saving interaction turn to persistent memory.")
                imemory.add_memory_entry(user_input=user_input_str, bot_response=bot_response_str)

    def get_retrieved_context_messages(self, current_query: str, k: int = 2) -> list[Message]:
        """
        --- iMemory Integration: Retrieve Hook ---
        Performs a semantic search on the persistent memory to find relevant
        past interactions and returns them as a list of Message objects.
        
        Args:
            current_query (str): The current user task or query to search for.
            k (int): The number of past interactions to retrieve.
            
        Returns:
            list[Message]: A list of messages to be injected into the prompt context.
        """
        logger.info(f"Retrieving {k} relevant memories for query: '{current_query[:50]}...'")
        retrieved_memories = imemory.retrieve_memories_semantic(current_query, k=k)
        
        if not retrieved_memories:
            return []
            
        context_messages: list[Message] = []
        # Prepend a header to clearly separate context for the LLM
        context_header = "Here are some relevant excerpts from our past conversations to provide you with long-term memory. Use them to inform your response:\n"
        
        # Format retrieved memories into a single context block
        full_context_text = context_header
        for i, mem in enumerate(retrieved_memories):
            try:
                # The memory entries were saved as JSON strings of step dicts
                user_step = json.loads(mem['user_input'])
                bot_step = json.loads(mem['bot_response'])
                
                # Extract the core information for a concise context
                user_task = user_step.get('task', 'User provided input.')
                bot_plan = bot_step.get('plan')
                bot_output = bot_step.get('model_output') or bot_step.get('observations')
                
                bot_action = bot_plan or bot_output or 'Agent took action.'

                full_context_text += f"\n--- Past Interaction {i+1} ---\n"
                full_context_text += f"User's Task: {user_task}\n"
                full_context_text += f"Your Response/Action: {bot_action}\n"

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not parse a retrieved memory entry: {e}")
                continue
        
        # Add the entire context block as a single system-level or user-level message
        context_messages.append(Message(role=MessageRole.SYSTEM, content=[{"type": "text", "text": full_context_text}]))
        
        return context_messages
        
    def reset(self):
        """Resets the in-session (short-term) memory. Does not affect persistent memory."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False):
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (AgentLogger): The logger to print replay logs to.
            detailed (bool, optional): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)


__all__ = ["AgentMemory"]
