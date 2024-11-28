import os
import json
import subprocess

# Input JSON files
json_files = ["./output.json", "./output_nithin.json"]

# Output results file
output_file = "./result_pipeline.txt"

# Prepare output file header
with open(output_file, "w") as f:
    f.write("image_path,caption,score_by_pipeline\n")

# Process each JSON file
for json_file in json_files:
    with open(json_file, "r") as f:
        for line in f:  # Process file line by line
            if line.strip():  # Ignore empty lines
                try:
                    data = json.loads(line)  # Parse individual JSON object
                    image_path = f".{data['image_path']}"
                    caption = data['caption']

                    # Run main.py with arguments
                    result = subprocess.run(
                        ["python", "main.py", image_path, caption],
                        capture_output=True,
                        text=True
                    )

                    # Extract the final score from the output
                    output_lines = result.stdout.splitlines()
                    final_score = None
                    for line in output_lines:
                        if line.startswith("Final Score:"):
                            final_score = line.split(":")[1].strip()
                            break

                    # Write to the result file
                    with open(output_file, "a") as f:
                        f.write(f"{image_path},{caption},{final_score}\n")

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line in {json_file}: {line}")
                    print(f"Details: {e}")
