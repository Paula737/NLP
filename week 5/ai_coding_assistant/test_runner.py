# test_runner.py
from ai_coding_assistant.modules.code_router import route_request
from modules.code_runner import run_code, extract_code_block

r = route_request("Write a function that checks if a list has close elements, then test it with a sample list and print the result")
print(r["answer"])

code = extract_code_block(r["answer"])
result = run_code(code)

print("\n--- Execution Result ---")
print("Success:", result["success"])
print("STDOUT:", result["stdout"])
print("STDERR:", result["stderr"])