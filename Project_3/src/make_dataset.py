import os

print("=" * 60)
print("Resume - Job Description Matching Explanation")
print("=" * 60)

file_path = "outputs/prediction_result.md"

if not os.path.exists(file_path):
    print("Prediction result file not found!")
    exit()

# Read prediction file
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

print("\nPrediction Result\n")
print(content)

print("\nExplanation")
print("-" * 60)

if "Strong Match" in content:
    print("""
The model predicts STRONG MATCH.

Reason:
✓ Resume skills match the Job Description.
✓ Required technologies are present.
✓ Resume is suitable for the job.
""")

elif "Weak Match" in content:
    print("""
The model predicts WEAK MATCH.

Reason:
✗ Resume does not match the required skills.
✗ Important keywords are missing.
✗ Candidate is not a suitable fit.
""")

else:
    print("Unable to determine prediction.")

print("-" * 60)
print("Explanation Completed Successfully.")