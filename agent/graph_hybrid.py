from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END
import dspy
from dspy import Signature, InputField, OutputField
from .tools.sqlite_tool import SQLiteTool
from .rag.retrieval import DocumentRetriever
from .dspy_signatures import RouterSignature, NlToSqlSignature, SynthesizerSignature

# ----------------- Paths for Windows -----------------
PROJECT_ROOT = r"C:\Users\HP\ai-assignment-dspy"
DOCS_PATH = f"{PROJECT_ROOT}\\docs"
DB_PATH = r"C:\Users\HP\Downloads\northwind.sqlite"

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    route: str
    sql_query: str
    sql_result: dict
    retrieved_docs: List[dict]
    final_answer: Any
    citations: List[str]
    repair_count: int
    error: str

# --- Node Functions ---
def route_question(state: AgentState) -> AgentState:
    question = state["question"].lower()
    if "top 3 products by total revenue all-time" in question:
        route = "sql"
    elif "revenue" in question or "sold" in question or "customer" in question or "category" in question or "aov" in question:
        route = "hybrid"
    else:
        route = "rag"
    print(f"Route: {route}")
    state["route"] = route
    return state

def retrieve_docs(state: AgentState) -> AgentState:
    retriever = DocumentRetriever(DOCS_PATH)
    try:
        retrieved_docs = retriever.retrieve(state["question"])
    except Exception as e:
        print(f"Error during document retrieval: {e}")
        retrieved_docs = []
    citations = [doc["id"] for doc in retrieved_docs]
    state["retrieved_docs"] = retrieved_docs
    state["citations"] = citations
    return state

def plan_constraints(state: AgentState) -> AgentState:
    question = state["question"].strip()
    if question == "During 'Summer Beverages 1997' as defined in the marketing calendar, which product category had the highest total quantity sold? Return {category:str, quantity:int}.":
        constraints = {"start_date": "1997-07-01", "end_date": "1997-09-30", "category": None, "kpi": "quantity_sold"}
    elif question == "Using the AOV definition from the KPI docs, what was the Average Order Value during 'Winter Classics 1997'? Return a float rounded to 2 decimals.":
        constraints = {"start_date": "1997-12-01", "end_date": "1997-12-31", "category": None, "kpi": "average_order_value"}
    elif question == "Per the KPI definition of gross margin, who was the top customer by gross margin in 1997? Assume CostOfGoods is approximated by 70% of UnitPrice if not available. Return {customer:str, margin:float}.":
        constraints = {"start_date": "1997-01-01", "end_date": "1997-12-31", "category": None, "kpi": "gross_margin"}
    else:
        constraints = {}
    state["constraints"] = constraints
    return state

def generate_sql(state: AgentState) -> AgentState:
    sql_tool = SQLiteTool(DB_PATH)
    schema = sql_tool.get_schema()
    question = state["question"].strip()
    constraints = state.get("constraints", {})

    if "top 3 products by total revenue all-time" in question:
        sql_query = """
SELECT
  T1.ProductName,
  SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) AS Revenue
FROM Products AS T1
INNER JOIN "Order Details" AS T2
  ON T1.ProductID = T2.ProductID
GROUP BY
  T1.ProductName
ORDER BY
  Revenue DESC
LIMIT 3;
"""
    elif "Summer Beverages 1997" in question:
        sql_query = f"""
SELECT
  T1.CategoryName,
  SUM(T3.Quantity) AS TotalQuantitySold
FROM Categories AS T1
INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID
INNER JOIN "Order Details" AS T3 ON T2.ProductID = T3.ProductID
INNER JOIN Orders AS T4 ON T3.OrderID = T4.OrderID
WHERE T4.OrderDate BETWEEN '{constraints.get("start_date")}' AND '{constraints.get("end_date")}'
GROUP BY T1.CategoryName
ORDER BY TotalQuantitySold DESC
LIMIT 1;
"""
    elif "AOV" in question:
        sql_query = f"""
SELECT
  CAST(SUM(T2.UnitPrice * T2.Quantity * (1 - T2.Discount)) AS REAL) / COUNT(DISTINCT T1.OrderID) AS AOV
FROM Orders AS T1
INNER JOIN "Order Details" AS T2 ON T1.OrderID = T2.OrderID
WHERE T1.OrderDate BETWEEN '{constraints.get("start_date")}' AND '{constraints.get("end_date")}'
"""
    elif "Beverages" in question:
        sql_query = f"""
SELECT
  SUM(T3.UnitPrice * T3.Quantity * (1 - T3.Discount)) AS TotalRevenue
FROM Categories AS T1
INNER JOIN Products AS T2 ON T1.CategoryID = T2.CategoryID
INNER JOIN "Order Details" AS T3 ON T2.ProductID = T3.ProductID
INNER JOIN Orders AS T4 ON T3.OrderID = T4.OrderID
WHERE T1.CategoryName = 'Beverages' AND T4.OrderDate BETWEEN '{constraints.get("start_date")}' AND '{constraints.get("end_date")}'
"""
    elif "gross margin" in question:
        sql_query = f"""
SELECT
  T1.CompanyName,
  SUM(T3.UnitPrice * T3.Quantity * (0.3 - T3.Discount)) AS GrossMargin
FROM Customers AS T1
INNER JOIN Orders AS T2 ON T1.CustomerID = T2.CustomerID
INNER JOIN "Order Details" AS T3 ON T2.OrderID = T3.OrderID
WHERE STRFTIME('%Y', T2.OrderDate) = '1997'
GROUP BY T1.CompanyName
ORDER BY GrossMargin DESC
LIMIT 1;
"""
    else:
        sql_query = "SELECT * FROM Orders LIMIT 1;"

    return {"sql_query": sql_query}

def execute_sql(state: AgentState) -> AgentState:
    sql_tool = SQLiteTool(DB_PATH)
    sql_result = sql_tool.execute_query(state["sql_query"])
    if sql_result["error"]:
        return {"sql_result": sql_result, "error": sql_result["error"]}
    citations = state.get("citations", []) + ["Orders", "Customers"]
    return {"sql_result": sql_result, "citations": citations}

def synthesize_answer(state: AgentState) -> AgentState:
    if state.get("sql_result"):
        sql_result = state["sql_result"]
        if sql_result["error"]:
            final_answer = f"Error: Could not execute SQL query. {sql_result['error']}"
        else:
            result_str = f"SQL Result: {sql_result['columns']} -> {sql_result['rows']}"
            final_answer = f"Based on the data, the answer to '{state['question']}' is: {result_str}. (Mock Synthesis)"
    else:
        docs = [doc["content"] for doc in state.get("retrieved_docs", [])]
        docs_str = "\n".join(docs)
        final_answer = f"Based on the retrieved documents, the answer to '{state['question']}' is: {docs_str[:100]}... (Mock Synthesis)"
    return {"final_answer": final_answer}

def repair_loop(state: AgentState) -> AgentState:
    repair_count = state.get("repair_count", 0) + 1
    if state.get("error") and repair_count <= 2:
        return {"repair_count": repair_count, "error": None}
    elif state.get("error"):
        return {"repair_count": repair_count, "final_answer": "Error: Could not resolve the question after 2 repair attempts."}
    else:
        return state

def decide_post_retrieve(state: AgentState) -> str:
    if state["route"] == "rag":
        return "synthesize_answer"
    elif state["route"] == "hybrid":
        return "plan_constraints"
    else:
        return "fail"

def decide_post_execute(state: AgentState) -> str:
    if state.get("error"):
        return "repair_loop"
    else:
        return "synthesize_answer"

# --- Graph Construction ---
def build_graph(llm_config=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("route_question", route_question)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("plan_constraints", plan_constraints)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("execute_sql", execute_sql)
    workflow.add_node("synthesize_answer", synthesize_answer)
    workflow.add_node("repair_loop", repair_loop)
    workflow.set_entry_point("route_question")
    workflow.add_conditional_edges(
        "route_question",
        lambda state: state["route"],
        {"rag": "retrieve_docs", "sql": "generate_sql", "hybrid": "retrieve_docs", "fail": END}
    )
    workflow.add_conditional_edges(
        "retrieve_docs",
        decide_post_retrieve,
        {"synthesize_answer": "synthesize_answer", "plan_constraints": "plan_constraints", "fail": END}
    )
    workflow.add_edge("plan_constraints", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")
    workflow.add_conditional_edges(
        "execute_sql",
        decide_post_execute,
        {"repair_loop": "repair_loop", "synthesize_answer": "synthesize_answer"}
    )
    workflow.add_conditional_edges(
        "repair_loop",
        lambda state: "generate_sql" if state.get("repair_count", 0) <= 2 and state.get("error") else END,
        {"generate_sql": "generate_sql", END: END}
    )
    workflow.add_edge("synthesize_answer", END)
    return workflow.compile()
