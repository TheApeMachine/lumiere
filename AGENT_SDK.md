Agents
Agents are the core building block in your apps. An agent is a large language model (LLM), configured with instructions and tools.

Basic configuration
The most common properties of an agent you'll configure are:

name: A required string that identifies your agent.
instructions: also known as a developer message or system prompt.
model: which LLM to use, and optional model_settings to configure model tuning parameters like temperature, top_p, etc.
tools: Tools that the agent can use to achieve its tasks.

from agents import Agent, ModelSettings, function_tool

@function_tool
def get_weather(city: str) -> str:
    """returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Haiku agent",
    instructions="Always respond in haiku form",
    model="gpt-5-nano",
    tools=[get_weather],
)
Context
Agents are generic on their context type. Context is a dependency-injection tool: it's an object you create and pass to Runner.run(), that is passed to every agent, tool, handoff etc, and it serves as a grab bag of dependencies and state for the agent run. You can provide any Python object as the context.


@dataclass
class UserContext:
    name: str
    uid: str
    is_pro_user: bool

    async def fetch_purchases() -> list[Purchase]:
        return ...

agent = Agent[UserContext](
    ...,
)
Output types
By default, agents produce plain text (i.e. str) outputs. If you want the agent to produce a particular type of output, you can use the output_type parameter. A common choice is to use Pydantic objects, but we support any type that can be wrapped in a Pydantic TypeAdapter - dataclasses, lists, TypedDict, etc.


from pydantic import BaseModel
from agents import Agent


class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

agent = Agent(
    name="Calendar extractor",
    instructions="Extract calendar events from text",
    output_type=CalendarEvent,
)
Note

When you pass an output_type, that tells the model to use structured outputs instead of regular plain text responses.

Multi-agent system design patterns
There are many ways to design multi‑agent systems, but we commonly see two broadly applicable patterns:

Manager (agents as tools): A central manager/orchestrator invokes specialized sub‑agents as tools and retains control of the conversation.
Handoffs: Peer agents hand off control to a specialized agent that takes over the conversation. This is decentralized.
See our practical guide to building agents for more details.

Manager (agents as tools)
The customer_facing_agent handles all user interaction and invokes specialized sub‑agents exposed as tools. Read more in the tools documentation.


from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

customer_facing_agent = Agent(
    name="Customer-facing agent",
    instructions=(
        "Handle all direct user communication. "
        "Call the relevant tools when specialized expertise is needed."
    ),
    tools=[
        booking_agent.as_tool(
            tool_name="booking_expert",
            tool_description="Handles booking questions and requests.",
        ),
        refund_agent.as_tool(
            tool_name="refund_expert",
            tool_description="Handles refund questions and requests.",
        )
    ],
)
Handoffs
Handoffs are sub‑agents the agent can delegate to. When a handoff occurs, the delegated agent receives the conversation history and takes over the conversation. This pattern enables modular, specialized agents that excel at a single task. Read more in the handoffs documentation.


from agents import Agent

booking_agent = Agent(...)
refund_agent = Agent(...)

triage_agent = Agent(
    name="Triage agent",
    instructions=(
        "Help the user with their questions. "
        "If they ask about booking, hand off to the booking agent. "
        "If they ask about refunds, hand off to the refund agent."
    ),
    handoffs=[booking_agent, refund_agent],
)
Dynamic instructions
In most cases, you can provide instructions when you create the agent. However, you can also provide dynamic instructions via a function. The function will receive the agent and context, and must return the prompt. Both regular and async functions are accepted.


def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."


agent = Agent[UserContext](
    name="Triage agent",
    instructions=dynamic_instructions,
)
Lifecycle events (hooks)
Sometimes, you want to observe the lifecycle of an agent. For example, you may want to log events, or pre-fetch data when certain events occur. You can hook into the agent lifecycle with the hooks property. Subclass the AgentHooks class, and override the methods you're interested in.

Guardrails
Guardrails allow you to run checks/validations on user input in parallel to the agent running, and on the agent's output once it is produced. For example, you could screen the user's input and agent's output for relevance. Read more in the guardrails documentation.

Cloning/copying agents
By using the clone() method on an agent, you can duplicate an Agent, and optionally change any properties you like.


pirate_agent = Agent(
    name="Pirate",
    instructions="Write like a pirate",
    model="gpt-4.1",
)

robot_agent = pirate_agent.clone(
    name="Robot",
    instructions="Write like a robot",
)
Forcing tool use
Supplying a list of tools doesn't always mean the LLM will use a tool. You can force tool use by setting ModelSettings.tool_choice. Valid values are:

auto, which allows the LLM to decide whether or not to use a tool.
required, which requires the LLM to use a tool (but it can intelligently decide which tool).
none, which requires the LLM to not use a tool.
Setting a specific string e.g. my_tool, which requires the LLM to use that specific tool.

from agents import Agent, Runner, function_tool, ModelSettings

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="get_weather")
)
Tool Use Behavior
The tool_use_behavior parameter in the Agent configuration controls how tool outputs are handled:

"run_llm_again": The default. Tools are run, and the LLM processes the results to produce a final response.
"stop_on_first_tool": The output of the first tool call is used as the final response, without further LLM processing.

from agents import Agent, Runner, function_tool, ModelSettings

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    tool_use_behavior="stop_on_first_tool"
)
StopAtTools(stop_at_tool_names=[...]): Stops if any specified tool is called, using its output as the final response.

from agents import Agent, Runner, function_tool
from agents.agent import StopAtTools

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

@function_tool
def sum_numbers(a: int, b: int) -> int:
    """Adds two numbers."""
    return a + b

agent = Agent(
    name="Stop At Stock Agent",
    instructions="Get weather or sum numbers.",
    tools=[get_weather, sum_numbers],
    tool_use_behavior=StopAtTools(stop_at_tool_names=["get_weather"])
)
ToolsToFinalOutputFunction: A custom function that processes tool results and decides whether to stop or continue with the LLM.

from agents import Agent, Runner, function_tool, FunctionToolResult, RunContextWrapper
from agents.agent import ToolsToFinalOutputResult
from typing import List, Any

@function_tool
def get_weather(city: str) -> str:
    """Returns weather info for the specified city."""
    return f"The weather in {city} is sunny"

def custom_tool_handler(
    context: RunContextWrapper[Any],
    tool_results: List[FunctionToolResult]
) -> ToolsToFinalOutputResult:
    """Processes tool results to decide final output."""
    for result in tool_results:
        if result.output and "sunny" in result.output:
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output=f"Final weather: {result.output}"
            )
    return ToolsToFinalOutputResult(
        is_final_output=False,
        final_output=None
    )

agent = Agent(
    name="Weather Agent",
    instructions="Retrieve weather details.",
    tools=[get_weather],
    tool_use_behavior=custom_tool_handler
)
Note

To prevent infinite loops, the framework automatically resets tool_choice to "auto" after a tool call. This behavior is configurable via agent.reset_tool_choice. The infinite loop is because tool results are sent to the LLM, which then generates another tool call because of tool_choice, ad infinitum.

Running agents
You can run agents via the Runner class. You have 3 options:

Runner.run(), which runs async and returns a RunResult.
Runner.run_sync(), which is a sync method and just runs .run() under the hood.
Runner.run_streamed(), which runs async and returns a RunResultStreaming. It calls the LLM in streaming mode, and streams those events to you as they are received.

from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = await Runner.run(agent, "Write a haiku about recursion in programming.")
    print(result.final_output)
    # Code within the code,
    # Functions calling themselves,
    # Infinite loop's dance
Read more in the results guide.

The agent loop
When you use the run method in Runner, you pass in a starting agent and input. The input can either be a string (which is considered a user message), or a list of input items, which are the items in the OpenAI Responses API.

The runner then runs a loop:

We call the LLM for the current agent, with the current input.
The LLM produces its output.
If the LLM returns a final_output, the loop ends and we return the result.
If the LLM does a handoff, we update the current agent and input, and re-run the loop.
If the LLM produces tool calls, we run those tool calls, append the results, and re-run the loop.
If we exceed the max_turns passed, we raise a MaxTurnsExceeded exception.
Note

The rule for whether the LLM output is considered as a "final output" is that it produces text output with the desired type, and there are no tool calls.

Streaming
Streaming allows you to additionally receive streaming events as the LLM runs. Once the stream is done, the RunResultStreaming will contain the complete information about the run, including all the new outputs produced. You can call .stream_events() for the streaming events. Read more in the streaming guide.

Run config
The run_config parameter lets you configure some global settings for the agent run:

model: Allows setting a global LLM model to use, irrespective of what model each Agent has.
model_provider: A model provider for looking up model names, which defaults to OpenAI.
model_settings: Overrides agent-specific settings. For example, you can set a global temperature or top_p.
input_guardrails, output_guardrails: A list of input or output guardrails to include on all runs.
handoff_input_filter: A global input filter to apply to all handoffs, if the handoff doesn't already have one. The input filter allows you to edit the inputs that are sent to the new agent. See the documentation in Handoff.input_filter for more details.
tracing_disabled: Allows you to disable tracing for the entire run.
trace_include_sensitive_data: Configures whether traces will include potentially sensitive data, such as LLM and tool call inputs/outputs.
workflow_name, trace_id, group_id: Sets the tracing workflow name, trace ID and trace group ID for the run. We recommend at least setting workflow_name. The group ID is an optional field that lets you link traces across multiple runs.
trace_metadata: Metadata to include on all traces.
Conversations/chat threads
Calling any of the run methods can result in one or more agents running (and hence one or more LLM calls), but it represents a single logical turn in a chat conversation. For example:

User turn: user enter text
Runner run: first agent calls LLM, runs tools, does a handoff to a second agent, second agent runs more tools, and then produces an output.
At the end of the agent run, you can choose what to show to the user. For example, you might show the user every new item generated by the agents, or just the final output. Either way, the user might then ask a followup question, in which case you can call the run method again.

Manual conversation management
You can manually manage conversation history using the RunResultBase.to_input_list() method to get the inputs for the next turn:


async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
        print(result.final_output)
        # San Francisco

        # Second turn
        new_input = result.to_input_list() + [{"role": "user", "content": "What state is it in?"}]
        result = await Runner.run(agent, new_input)
        print(result.final_output)
        # California
Automatic conversation management with Sessions
For a simpler approach, you can use Sessions to automatically handle conversation history without manually calling .to_input_list():


from agents import Agent, Runner, SQLiteSession

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # Create session instance
    session = SQLiteSession("conversation_123")

    thread_id = "thread_123"  # Example thread ID
    with trace(workflow_name="Conversation", group_id=thread_id):
        # First turn
        result = await Runner.run(agent, "What city is the Golden Gate Bridge in?", session=session)
        print(result.final_output)
        # San Francisco

        # Second turn - agent automatically remembers previous context
        result = await Runner.run(agent, "What state is it in?", session=session)
        print(result.final_output)
        # California
Sessions automatically:

Retrieves conversation history before each run
Stores new messages after each run
Maintains separate conversations for different session IDs
See the Sessions documentation for more details.

Server-managed conversations
You can also let the OpenAI conversation state feature manage conversation state on the server side, instead of handling it locally with to_input_list() or Sessions. This allows you to preserve conversation history without manually resending all past messages. See the OpenAI Conversation state guide for more details.

OpenAI provides two ways to track state across turns:

1. Using conversation_id
You first create a conversation using the OpenAI Conversations API and then reuse its ID for every subsequent call:


from agents import Agent, Runner
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def main():
    # Create a server-managed conversation
    conversation = await client.conversations.create()
    conv_id = conversation.id    

    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # First turn
    result1 = await Runner.run(agent, "What city is the Golden Gate Bridge in?", conversation_id=conv_id)
    print(result1.final_output)
    # San Francisco

    # Second turn reuses the same conversation_id
    result2 = await Runner.run(
        agent,
        "What state is it in?",
        conversation_id=conv_id,
    )
    print(result2.final_output)
    # California
2. Using previous_response_id
Another option is response chaining, where each turn links explicitly to the response ID from the previous turn.


from agents import Agent, Runner

async def main():
    agent = Agent(name="Assistant", instructions="Reply very concisely.")

    # First turn
    result1 = await Runner.run(agent, "What city is the Golden Gate Bridge in?")
    print(result1.final_output)
    # San Francisco

    # Second turn, chained to the previous response
    result2 = await Runner.run(
        agent,
        "What state is it in?",
        previous_response_id=result1.last_response_id,
    )
    print(result2.final_output)
    # California
Long running agents & human-in-the-loop
You can use the Agents SDK Temporal integration to run durable, long-running workflows, including human-in-the-loop tasks. View a demo of Temporal and the Agents SDK working in action to complete long-running tasks in this video, and view docs here.

Exceptions
The SDK raises exceptions in certain cases. The full list is in agents.exceptions. As an overview:

AgentsException: This is the base class for all exceptions raised within the SDK. It serves as a generic type from which all other specific exceptions are derived.
MaxTurnsExceeded: This exception is raised when the agent's run exceeds the max_turns limit passed to the Runner.run, Runner.run_sync, or Runner.run_streamed methods. It indicates that the agent could not complete its task within the specified number of interaction turns.
ModelBehaviorError: This exception occurs when the underlying model (LLM) produces unexpected or invalid outputs. This can include:
Malformed JSON: When the model provides a malformed JSON structure for tool calls or in its direct output, especially if a specific output_type is defined.
Unexpected tool-related failures: When the model fails to use tools in an expected manner
UserError: This exception is raised when you (the person writing code using the SDK) make an error while using the SDK. This typically results from incorrect code implementation, invalid configuration, or misuse of the SDK's API.
InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered: This exception is raised when the conditions of an input guardrail or output guardrail are met, respectively. Input guardrails check incoming messages before processing, while output guardrails check the agent's final response before delivery.

Handoffs
Handoffs allow an agent to delegate tasks to another agent. This is particularly useful in scenarios where different agents specialize in distinct areas. For example, a customer support app might have agents that each specifically handle tasks like order status, refunds, FAQs, etc.

Handoffs are represented as tools to the LLM. So if there's a handoff to an agent named Refund Agent, the tool would be called transfer_to_refund_agent.

Creating a handoff
All agents have a handoffs param, which can either take an Agent directly, or a Handoff object that customizes the Handoff.

You can create a handoff using the handoff() function provided by the Agents SDK. This function allows you to specify the agent to hand off to, along with optional overrides and input filters.

Basic Usage
Here's how you can create a simple handoff:


from agents import Agent, handoff

billing_agent = Agent(name="Billing agent")
refund_agent = Agent(name="Refund agent")


triage_agent = Agent(name="Triage agent", handoffs=[billing_agent, handoff(refund_agent)])
Customizing handoffs via the handoff() function
The handoff() function lets you customize things.

agent: This is the agent to which things will be handed off.
tool_name_override: By default, the Handoff.default_tool_name() function is used, which resolves to transfer_to_<agent_name>. You can override this.
tool_description_override: Override the default tool description from Handoff.default_tool_description()
on_handoff: A callback function executed when the handoff is invoked. This is useful for things like kicking off some data fetching as soon as you know a handoff is being invoked. This function receives the agent context, and can optionally also receive LLM generated input. The input data is controlled by the input_type param.
input_type: The type of input expected by the handoff (optional).
input_filter: This lets you filter the input received by the next agent. See below for more.
is_enabled: Whether the handoff is enabled. This can be a boolean or a function that returns a boolean, allowing you to dynamically enable or disable the handoff at runtime.

from agents import Agent, handoff, RunContextWrapper

def on_handoff(ctx: RunContextWrapper[None]):
    print("Handoff called")

agent = Agent(name="My agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    tool_name_override="custom_handoff_tool",
    tool_description_override="Custom description",
)
Handoff inputs
In certain situations, you want the LLM to provide some data when it calls a handoff. For example, imagine a handoff to an "Escalation agent". You might want a reason to be provided, so you can log it.


from pydantic import BaseModel

from agents import Agent, handoff, RunContextWrapper

class EscalationData(BaseModel):
    reason: str

async def on_handoff(ctx: RunContextWrapper[None], input_data: EscalationData):
    print(f"Escalation agent called with reason: {input_data.reason}")

agent = Agent(name="Escalation agent")

handoff_obj = handoff(
    agent=agent,
    on_handoff=on_handoff,
    input_type=EscalationData,
)
Input filters
When a handoff occurs, it's as though the new agent takes over the conversation, and gets to see the entire previous conversation history. If you want to change this, you can set an input_filter. An input filter is a function that receives the existing input via a HandoffInputData, and must return a new HandoffInputData.

There are some common patterns (for example removing all tool calls from the history), which are implemented for you in agents.extensions.handoff_filters


from agents import Agent, handoff
from agents.extensions import handoff_filters

agent = Agent(name="FAQ agent")

handoff_obj = handoff(
    agent=agent,
    input_filter=handoff_filters.remove_all_tools, 
)
Recommended prompts
To make sure that LLMs understand handoffs properly, we recommend including information about handoffs in your agents. We have a suggested prefix in agents.extensions.handoff_prompt.RECOMMENDED_PROMPT_PREFIX, or you can call agents.extensions.handoff_prompt.prompt_with_handoff_instructions to automatically add recommended data to your prompts.


from agents import Agent
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

billing_agent = Agent(
    name="Billing agent",
    instructions=f"""{RECOMMENDED_PROMPT_PREFIX}
    <Fill in the rest of your prompt here>.""",
)