from typing import TypedDict, Optional
from langgraph.graph import StateGraph,END
from langchain_groq import ChatGroq
from config import get_settings
settings=get_settings()
class QAState(TypedDict):
    question: Optional[str]
    context: Optional[str]
    answer: Optional[str]
    valid: Optional[bool]
    error: Optional[str]

qa_state_example = QAState(
    question="What is the purpose of this guided project?",
    context="This project focuses on building a chatbot using Python.",
    answer=None
)
def input_validation_node(state):
    question = state.get("question", "").strip()
    if not question:
        return {"valid": False, "error": "Question cannot be empty."}
    
    return {"valid": True}

def context_provider_node(state):
    question = state.get("question", "").lower()
    if "langgraph" in question or "guided project" in question:
        context = (
            "This guided project is about using LangGraph, a Python library to design state-based workflows. "
            "LangGraph simplifies building complex applications by connecting modular nodes with conditional edges."
        )
        return {"context": context}
    return {"context": None}

llm=ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

def llm_qa_node(state: QAState) -> QAState:
    question = state.get("question") or ""
    context = state.get("context")

    if not context:
        return {"answer": "I don't have enough context to answer your question."}

    try:
        response = llm.invoke(
            [
                {
                    "role": "user",
                    "content": (
                        f"Context: {context}\n"
                        f"Question: {question}\n"
                        "Answer the question based on the provided context."
                    ),
                }
            ]
        )
        return {"answer": response.content.strip()}
    except Exception as e:
        return {"answer": f"An error occurred: {str(e)}"}

qa_workflow = StateGraph(QAState)
qa_workflow.add_node("InputNode", input_validation_node)
qa_workflow.add_node("ContextNode", context_provider_node)
qa_workflow.add_node("QANode", llm_qa_node)
qa_workflow.set_entry_point("InputNode")
qa_workflow.add_edge("InputNode", "ContextNode")
qa_workflow.add_edge("ContextNode", "QANode")
qa_workflow.add_edge("QANode", END)
qa_app = qa_workflow.compile()
res=qa_app.invoke({"question": "What is LangGraph?"})
print(res)