import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.rag_compliance import query_compliance

# Test queries
queries = [
    "What is the tax rate on crypto?",
    "Do I need to pay TDS?",
    "What is FIU-IND?",
    "Can I offset my crypto losses?",
    "What are the risk limits for automated trading?"
]

print("=== RAG Evaluation ===")
print(f"Testing {len(queries)} queries against the hybrid RAG system...\n")

for q in queries:
    print(f"Q: {q}")
    result = query_compliance.invoke(q)
    print(f"A:\n{result}")
    print("-" * 50)

print("\nEvaluation complete.")
