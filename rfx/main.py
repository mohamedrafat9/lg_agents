from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from typing import TypedDict, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph,END
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import get_settings
settings=get_settings()

tavilyTool=TavilySearchResults(
    max_results=1,
    api_wrapper=TavilySearchAPIWrapper(
        tavily_api_key=settings.tavily_api_key
    ),
)
# sample_query = "healthy breakfast recipes"
# search_results = tavilyTool.invoke(sample_query)
# print(search_results)
llm=ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are Dr. Paul Saladino, "Carnivore MD," advocating for animal-based nutrition and challenging plant-centric dietary dogma. Focus on the superior bioavailability of animal foods and the potential toxicity of plant compounds such as oxalates, lectins, and phytates.

        Your response must follow these steps:
        1. {first_instruction}
        2. Present the evolutionary and biochemical rationale for animal-based nutrition, emphasizing nutrient density, bioavailability, and the absence of plant antinutrients.
        3. Challenge conventional "plants are healthy" narratives with mechanistic evidence about plant defense compounds and their impact on human physiology.
        4. Reflect and critique your answer. Be rigorous about evolutionary consistency and nutrient science.
        5. After the reflection, **list 1-3 search queries separately** for researching antinutrients, bioavailability studies, or evolutionary nutrition. Do not include them inside the reflection.

        Focus on: organ meats as superfoods, the carnivore elimination protocol, and why "fiber is unnecessary" based on human evolutionary biology.
        """
    ),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system", 
        "Answer the user's question above using the required format, emphasizing the superiority of animal-based nutrition."
    ),
])
question="Any ideas for a healthy breakfast"
first_responder_prompt = prompt_template.partial(first_instruction="Provide a detailed ~250 word answer")

class Reflection(BaseModel):
	missing: str = Field(description="What information is missing")
	superfluous: str = Field(description="What information is unnecessary")

class AnswerQuestion(BaseModel):
	answer: str = Field(description="Main response to the question")
	reflection: Reflection = Field(description="Self-critique of the answer")
	search_queries: List[str] = Field(description="Queries for additional research")


class ResearchState(TypedDict, total=False):
    question: str
    answer: str
    reflection: Reflection
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    final_answer: str


first_responder_chain = first_responder_prompt | llm.with_structured_output(AnswerQuestion)

reflection_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Review the answer below as a rigorous nutrition editor.
Identify what information is missing and what information is unnecessary.
Do not rewrite the answer. Return only the requested structured critique.""",
    ),
    ("human", "Question: {question}\n\nAnswer:\n{answer}"),
])
reflection_chain = reflection_prompt | llm.with_structured_output(Reflection)

revision_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Rewrite the answer to the user's question using the editor's critique and
the research notes. Keep the answer practical, evidence-aware, and about 250 words.
Do not mention the editing process, research notes, or these instructions.
Return only the revised answer.""",
    ),
    (
        "human",
        "Question: {question}\n\nOriginal answer:\n{answer}\n\n"
        "Critique:\n{reflection}\n\nResearch notes:\n{research_results}",
    ),
])
revision_chain = revision_prompt | llm


def first_responder_node(state: ResearchState) -> ResearchState:
    result = first_responder_chain.invoke(
        {"messages": [HumanMessage(content=state["question"])]}
    )
    return {
        "answer": result.answer,
        "reflection": result.reflection,
        "search_queries": result.search_queries[:3],
    }


def reflection_node(state: ResearchState) -> ResearchState:
    reflection = reflection_chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
    })
    return {"reflection": reflection}


def research_node(state: ResearchState) -> ResearchState:
    search_results: List[Dict[str, Any]] = []
    for query in state.get("search_queries", []):
        result = tavilyTool.invoke(query)
        if isinstance(result, list):
            search_results.extend(result)
        else:
            search_results.append({"query": query, "result": result})
    return {"search_results": search_results}


def revision_node(state: ResearchState) -> ResearchState:
    result = revision_chain.invoke({
        "question": state["question"],
        "answer": state["answer"],
        "reflection": state["reflection"].model_dump(),
        "research_results": state.get("search_results", []),
    })
    return {"final_answer": result.content}


workflow = StateGraph(ResearchState)
workflow.add_node("first_responder", first_responder_node)
workflow.add_node("reflection", reflection_node)
workflow.add_node("research", research_node)
workflow.add_node("revision", revision_node)
workflow.set_entry_point("first_responder")
workflow.add_edge("first_responder", "reflection")
workflow.add_edge("reflection", "research")
workflow.add_edge("research", "revision")
workflow.add_edge("revision", END)

app = workflow.compile()
result = app.invoke({"question": question})

print("\nFIRST ANSWER:\n")
print(result["answer"])
print("\nREFLECTION:\n")
print(result["reflection"].model_dump_json(indent=2))
print("\nFINAL ANSWER:\n")
print(result["final_answer"])
