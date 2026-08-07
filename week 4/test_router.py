# test_router.py
from modules.router import route_request

# Should classify as "explain"
r1 = route_request(
    "Explain this code",
    code_snippet="def add(a, b):\n    return a + b"
)
print("INTENT:", r1["intent"])
print(r1["answer"][:300], "\n")

# Should classify as "generate"
r2 = route_request("Write a function that checks if a list has close elements")
print("INTENT:", r2["intent"])
print(r2["answer"][:300])