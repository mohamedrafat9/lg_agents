from langgraph.graph import StateGraph
from typing import TypedDict, Optional
from langgraph.graph import END

class AuthState(TypedDict):
    username: Optional[str] 
    password: Optional[str]
    is_authenticated: Optional[bool]
    output: Optional[str]

auth_state_1: AuthState = {
    "username": "alice123",
    "password": "123",
    "is_authenticated": True,
    "output": "Login successful."
}
# print(f"auth_state_1: {auth_state_1}")

def input_node(state):
    print(state)
    if state.get('username', "") =="":
        username = input("What is your username?")

    password = input("Enter your password: ")
    
    if state.get('username', "") =="":
        return {"username":username, "password":password}
    else:
        return {"password": password}

def validate_credentials_node(state):
    username = state.get("username", "")
    password = state.get("password", "")

    print("Username :", username, "Password :", password)
    if username == "test_user" and password == "secure_password":
        is_authenticated = True
    else:
        is_authenticated = False

    return {"is_authenticated": is_authenticated}

def success_node(state):
    return {"output": "Authentication successful! Welcome."}

def failure_node(state):
    return {"output": "Not Successfull, please try again!"}

def router(state):# acts like a  decision making point
    if state['is_authenticated']:
        return "success_node"
    else:
        return "failure_node"


workflow = StateGraph(AuthState)

workflow.add_node("InputNode",input_node)
workflow.add_node("ValidateCredential", validate_credentials_node)
workflow.add_node("Success", success_node)
workflow.add_node("Failure", failure_node)
workflow.add_edge("InputNode", "ValidateCredential")
workflow.add_edge("Success", END)
workflow.add_edge("Failure", "InputNode")
workflow.add_conditional_edges(
        "ValidateCredential",
         router, 
         {
            "success_node": "Success",
            "failure_node": "Failure"
         }
)

workflow.set_entry_point("InputNode")

app = workflow.compile()
inputs = {"username": "test_user"}
result = app.invoke(inputs)
print(result)
print(result['output'])
print(app.get_graph().draw_mermaid())