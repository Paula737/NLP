# test_explainer.py
from modules.explainer import explain_results

result = explain_results(
    question="what is the average salary in the engineering department",
    sql="SELECT AVG(salary) FROM dataset WHERE LOWER(department) = LOWER('engineering')",
    rows=[{"AVG(salary)": 8550}]
)
print(result)