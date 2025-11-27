import dspy

class RouterSignature(dspy.Signature):
    """Router signature"""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="rag | sql | hybrid")

class NlToSqlSignature(dspy.Signature):
    """NL to SQL signature"""

    question = dspy.InputField()
    schema = dspy.InputField()
    sql_query = dspy.OutputField(desc="SQL query")

class SynthesizerSignature(dspy.Signature):
    """Synthesizer signature"""

    question = dspy.InputField()
    retrieved_docs = dspy.InputField()
    sql_results = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer")
