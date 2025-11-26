import os
import json
import dspy
from dspy.teleprompter import BootstrapFewShot
from agent.dspy_signatures import NLtoSQL, NLtoSQLSignature
from agent.tools.sqlite_tool import SQLiteTool

# --- Configuration ---
LLM_MODEL = "gpt-4.1-mini" 
LLM_CONFIG = dspy.OpenAI(model=LLM_MODEL, api_key=os.environ.get("OPENAI_API_KEY"))
dspy.settings.configure(lm=LLM_CONFIG)

DB_TOOL = SQLiteTool()
DB_SCHEMA = DB_TOOL.get_schema()

# --- Metric Definition ---
def sql_exact_match(example, prediction, trace=None):
    """Simple metric: checks if the predicted SQL query exactly matches the gold query."""
    return example.sql_query.strip().lower() == prediction.sql_query.strip().lower()

# --- Optimization Logic ---
def optimize_nl_to_sql():
    print("--- Starting NLtoSQL Optimization ---")
    
    # 1. Load Training Data
    train_data_path = os.path.join(os.path.dirname(__file__), "nl_to_sql_train.json")
    with open(train_data_path, 'r') as f:
        raw_data = json.load(f)
        
    # Convert raw data to DSPy Examples
    trainset = []
    for item in raw_data:
        # Inject the actual DB schema into the example for training
        trainset.append(dspy.Example(
            question=item["question"],
            db_schema=DB_SCHEMA,
            constraints=item["constraints"],
            sql_query=item["sql_query"]
        ).with_inputs("question", "db_schema", "constraints"))
        
    print(f"Loaded {len(trainset)} training examples.")
    
    # 2. Define the Module to Optimize
    unoptimized_nl_to_sql = NLtoSQL()
    
    # 3. Run the Optimizer (BootstrapFewShot)
    # We use a small number of examples (max_bootstrapped_demos=2) to keep it fast and local
    teleprompter = BootstrapFewShot(metric=sql_exact_match, max_bootstrapped_demos=2, max_rounds=1)
    
    optimized_nl_to_sql = teleprompter.compile(
        unoptimized_nl_to_sql, 
        trainset=trainset
    )
    
    # 4. Save the Optimized Module
    optimized_nl_to_sql.save("optimized_nl_to_sql.json")
    print("\n--- Optimization Complete ---")
    print("Optimized NLtoSQL module saved to optimized_nl_to_sql.json")
    
    # 5. Show Before/After Metric (Simple check on the trainset)
    
    # Helper to evaluate
    def evaluate_module(module, data):
        correct = 0
        for example in data:
            try:
                pred = module(
                    question=example.question, 
                    db_schema=example.db_schema, 
                    constraints=example.constraints
                )
                if sql_exact_match(example, pred):
                    correct += 1
            except Exception as e:
                print(f"Error during evaluation: {e}")
                pass
        return correct / len(data)

    unoptimized_score = evaluate_module(unoptimized_nl_to_sql, trainset)
    optimized_score = evaluate_module(optimized_nl_to_sql, trainset)
    
    print(f"\n[bold yellow]Metric (SQL Exact Match on Trainset):[/bold yellow]")
    print(f"  Unoptimized Score: {unoptimized_score:.2f}")
    print(f"  Optimized Score: {optimized_score:.2f}")
    
    # Save the metric delta for the README
    with open("optimization_metric.json", "w") as f:
        json.dump({
            "module": "NLtoSQL",
            "optimizer": "BootstrapFewShot",
            "metric": "SQL Exact Match on Trainset",
            "unoptimized_score": unoptimized_score,
            "optimized_score": optimized_score
        }, f)
    
    print("Metric delta saved to optimization_metric.json")

if __name__ == '__main__':
    # Change directory to the project root for correct relative path resolution
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    optimize_nl_to_sql()
