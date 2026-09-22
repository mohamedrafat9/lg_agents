from typing import TypedDict, Optional
from typing import List, Sequence
from langgraph.graph import StateGraph,END,MessageGraph
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import get_settings
settings=get_settings()
llm=ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional LinkedIn content assistant.

Generate a LinkedIn post under 160 characters.

If previous attempts and critique are provided:
- You must produce a meaningfully different version.
- Apply at least one concrete suggestion from the critique.
- Do not copy the previous post.
- Return only the final LinkedIn post.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Critique the latest LinkedIn post.

Identify exactly:
1. What should be removed
2. What should be added
3. What wording should be changed

The next version must be different from the current version.
"""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect_chain = reflection_prompt | llm

graph = MessageGraph()

def generation_node(state: Sequence[BaseMessage]) -> List[BaseMessage]:
    generated_post = generate_chain.invoke({"messages": state})
    return [AIMessage(content=generated_post.content)]

def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    res = reflect_chain.invoke({"messages": messages})  # Passes messages as input to reflect_chain
    return [HumanMessage(content=res.content)] 

graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)
graph.add_edge("reflect", "generate")
graph.set_entry_point("generate")
def should_continue(state: List[BaseMessage]):
    # print(state)
    # print(len(state))
    print("----------------------------------------------------------------------")
    if len(state) > 6:
        return END
    return "reflect"

graph.add_conditional_edges("generate", should_continue)

workflow = graph.compile()

inputs = HumanMessage(content="""Write a linkedin post on getting a software developer job at IBM under 160 characters""")

response = workflow.invoke(inputs)

first_post = response[1].content
final_post = response[-1].content

print("First:", first_post)
print("Final:", final_post)
print("Changed:", first_post.strip() != final_post.strip())

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

print(response[2].content)#first critique

print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

