import json
import os

checkpoints = "/nfs/brick/explainables/runs/euro_llm_explaind_scores/verify_blimp/train_annealing_megatronlm"

results = []
for ckpt in os.listdir(checkpoints):
    verf_path = os.path.join(checkpoints, ckpt, "verification_results.json")
    if os.path.exists(verf_path):
        with open(verf_path, "r") as f:
            data = json.load(f)
        results.append((ckpt, data))

deltas = [r[1][0]['de_sen_good']['delta'] for r in results]
change = [r[1][0]['de_sen_good']['actual_change'] for r in results]

percent_change = [abs(d/c)*100 if c != 0 else 0 for d, c in zip(deltas, change)]

mean_percent_change = sum(percent_change) / len(percent_change)
mean_deltas = sum(deltas) / len(deltas)
mean_change = sum(change) / len(change)

deltas_std = (sum((x - mean_deltas) ** 2 for x in deltas) / len(deltas)) ** 0.5

std_percent_change = (sum((x - mean_percent_change) ** 2 for x in percent_change) / len(percent_change)) ** 0.5

print(f"Mean Percent Change: {mean_percent_change}")
print(f"Mean Deltas: {mean_deltas}")
print(f"Std Deltas: {deltas_std}")
print(f"Mean Actual Change: {mean_change}")
print(f"Std Percent Change: {std_percent_change}")