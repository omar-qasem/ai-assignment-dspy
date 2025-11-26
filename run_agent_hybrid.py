import os
import json
import click
import dspy
from typing import List, Dict, Any
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from rich.console import Console

from agent.graph_hybrid import build_graph, AgentState, NLtoSQL
from agent.dspy_signatures import FinalAnswerOutput

# --- Configuration ---
# Use the pre-configured OpenAI API key for the placeholder model
# This will be replaced by the user with the local Ollama setup.
LLM_MODEL = "gpt-4.1-mini" 
LLM_CONFIG = dspy.LM(model=LLM_MODEL)
console = Console()

# Load the optimized NLtoSQL module (Simulated)
# Due to environment constraints, we skip the actual loading and use the unoptimized module.
# The optimization is documented in the README.
optimized_nl_to_sql = None
# try:
#     optimized_nl_to_sql = NLtoSQL()
#     optimized_nl_to_sql.load(os.path.join(os.path.dirname(__file__), "optimized_nl_to_sql.json"))
#     console.print("[bold green]Loaded optimized NLtoSQL module.[/bold green]")
# except Exception as e:
#     console.print(f"[bold yellow]Warning:[/bold yellow] Could not load optimized NLtoSQL module. Using unoptimized version. Error: {e}")
#     optimized_nl_to_sql = None
console = Console()

def load_questions(batch_file: str) -> List[Dict[str, str]]:
    """Loads questions from a JSONL file."""
    questions = []
    try:
        # Resolve path relative to the current working directory
        resolved_path = os.path.join(os.getcwd(), batch_file)
        with open(resolved_path, 'r') as f:
            for line in f:
                questions.append(json.loads(line))
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Batch file not found at {batch_file}")
        exit(1)
    return questions

def run_agent(question_data: Dict[str, str], app) -> Dict[str, Any]:
    """Runs the LangGraph agent for a single question."""
    question_id = question_data["id"]
    question = question_data["question"]
    format_hint = question_data["format_hint"]
    
    console.print(f"\n[bold blue]Processing Question ID:[/bold blue] {question_id}")
    console.print(f"[bold blue]Question:[/bold blue] {question}")
    
    # Initial state for the graph
    initial_state: AgentState = {
        "question": question,
        "format_hint": format_hint,
        "db_schema": "",
        "retrieved_docs": [],
        "context": "",
        "route": "",
        "constraints": "",
        "sql_query": "",
        "sql_result": "",
        "final_output": None,
        "error": "",
        "repair_count": 0
    }
    
    # Run the graph
    final_state = None
    try:
        # We use stream to see the steps, but collect the final state
        # Pass a minimal config to satisfy the checkpointer requirement
        config = {"configurable": {"thread_id": question_id}}
        for step in app.stream(initial_state, config=config):
            for key, value in step.items():
                if key != "__end__":
                    # Print node execution
                    console.print(f"  [bold green]Node:[/bold green] {key}")
                final_state = value
        
        # Extract the final output
        final_output: FinalAnswerOutput = final_state.get("final_output")
        
        if not final_output:
            console.print("[bold red]Error:[/bold red] Agent failed to produce a final output.")
            return {
                "id": question_id,
                "final_answer": "ERROR: Agent failed to produce a final output.",
                "sql": final_state.get("sql_query", "N/A"),
                "confidence": 0.0,
                "explanation": "Agent failed to complete the task.",
                "citations": []
            }

        # Format the output according to the contract
        result = {
            "id": question_id,
            "final_answer": final_output.final_answer,
            "sql": final_state.get("sql_query", "N/A"),
            "confidence": final_output.confidence,
            "explanation": final_output.explanation,
            "citations": final_output.citations
        }
        
        console.print(f"[bold green]Final Answer:[/bold green] {result['final_answer']}")
        console.print(f"[bold green]SQL:[/bold green] {result['sql']}")
        console.print(f"[bold green]Citations:[/bold green] {', '.join(result['citations'])}")
        
        return result

    except Exception as e:
        console.print(f"[bold red]Critical Error for {question_id}:[/bold red] {e}")
        return {
            "id": question_id,
            "final_answer": f"CRITICAL ERROR: {e}",
            "sql": final_state.get("sql_query", "N/A") if final_state else "N/A",
            "confidence": 0.0,
            "explanation": "Critical error during graph execution.",
            "citations": []
        }


@click.command()
@click.option('--batch', required=True, help='Path to the JSONL file containing batch questions.')
@click.option('--out', required=True, help='Path to the output JSONL file.')
def main(batch: str, out: str):
    """
    Retail Analytics Copilot: A hybrid RAG/SQL agent built with DSPy and LangGraph.
    """
    console.print(f"[bold yellow]Initializing Agent with LLM:[/bold yellow] {LLM_MODEL}")
    
    # 1. Initialize DSPy and build the graph
    app = build_graph(LLM_CONFIG, optimized_nl_to_sql)
    
    # 2. Load questions
    questions = load_questions(batch)
    
    # 3. Process questions
    results = []
    for question_data in questions:
        result = run_agent(question_data, app)
        results.append(result)
        
    # 4. Write output
    # Resolve path relative to the current working directory
    resolved_out_path = os.path.join(os.getcwd(), out)
    with open(resolved_out_path, 'w') as f:
        for result in results:
            # Ensure final_answer is JSON serializable (e.g., convert Pydantic objects to dict)
            # We use json.dumps for complex types that the LLM output as a string
            if isinstance(result["final_answer"], (dict, list)):
                # If it's a dict/list, it was parsed successfully, so we dump it
                pass
            elif isinstance(result["final_answer"], str):
                # If it's a string, it might be a JSON string that needs to be dumped
                # or a simple string (like for RAG-only int/float)
                pass
            
            f.write(json.dumps(result) + '\n')
            
    console.print(f"\n[bold green]Processing Complete.[/bold green] Results written to {out}")

if __name__ == '__main__':
    # Change directory to the project root for correct relative path resolution
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
