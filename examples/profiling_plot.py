import json
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

def load_json_data(json_file):
    """
    Read and parse the JSON file, extract necessary fields such as prompt_len, prefill_cost, and decode_costs.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)

    prompt_len = data.get("case", {}).get("prompt_len", None)
    prefill_cost = data.get("prefill_cost", None)

    # Get all decode_costs (decode_cost_1, decode_cost_2, ..., decode_cost_n)
    decode_latencies = [data.get(f"decode_cost_{i+1}", None) for i in range(0, 9)]

    return prompt_len, prefill_cost, decode_latencies

def plot_latency_vs_prompt_len(directory):
    """
    Read and parse the JSON files in the specified directory, extract necessary fields such as prompt_len, prefill_cost, and decode_costs.
    """
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]

    plt.figure(figsize=(12, 8))

    all_prompt_lens = []
    all_prefill_latencies = []
    all_decode_latencies = []

    for json_file in json_files:
        json_path = os.path.join(directory, json_file)
        prompt_len, prefill_latency, decode_latencies = load_json_data(json_path)
        print(f"prompt_len: {prompt_len}, prefill_latency: {prefill_latency}, decode_latencies: {decode_latencies}")

        if prompt_len and prefill_latency is not None and decode_latencies:
            all_prompt_lens.append(prompt_len)
            all_prefill_latencies.append(prefill_latency)
            all_decode_latencies.append(np.mean(decode_latencies))

    unique_prompt_lens = sorted(set(all_prompt_lens))
    avg_prefill_latencies = []
    avg_decode_latencies = []

    for prompt_len in unique_prompt_lens:
        prefill_values = [latency for pl, latency in zip(all_prompt_lens, all_prefill_latencies) if pl == prompt_len]
        decode_values = [latency for pl, latency in zip(all_prompt_lens, all_decode_latencies) if pl == prompt_len]

        avg_prefill_latencies.append(np.mean(prefill_values))
        avg_decode_latencies.append(np.mean(decode_values))

    plt.plot(unique_prompt_lens, avg_prefill_latencies, label='Prefill Latency (Average)', marker='o', linestyle='-', color='b')
    plt.plot(unique_prompt_lens, avg_decode_latencies, label='Decode Latency (Average)', marker='x', linestyle='-', color='r')

    plt.title('Latency vs Prompt Length', fontsize=16)
    plt.xlabel('Prompt Length', fontsize=14)
    plt.ylabel('Latency (seconds)', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('latency_vs_prompt_len.png')

    # Save prefill latency and decode latency to a CSV file
    with open('latency_data.csv', 'w') as csv_file:
        csv_file.write("Prompt Length, Prefill Latency, Decode Latency\n")
        for i in range(len(unique_prompt_lens)):
            csv_file.write(f"{unique_prompt_lens[i]}, {avg_prefill_latencies[i]}, {avg_decode_latencies[i]}\n")

directory = "datenlord_profile_k100"

plot_latency_vs_prompt_len(directory)