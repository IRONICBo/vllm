import traceback

import torch
# torch.set_printoptions(threshold=torch.inf)

import time
from vllm import LLM, SamplingParams

def get_generation_time(llm, sampling_params, prompts):
    # time the generation
    start_time = time.time()
    # output = llm.generate(prompts, sampling_params=sampling_params)
    output = llm.generate(prompts, sampling_params=sampling_params)
    end_time = time.time()
    # print the output and generation time
    if len(output) >= 1:
        for i in range(len(output)):
            print(f"output[{i}] Output: [ {output[i].outputs[0].text} ]")
        # print(f"output >=1 Output: [ {output[0].outputs[0].text} ]")
    else:
        print(f"output < 1 Output: [ {output} ]")
    print(f"Generation time: {end_time - start_time} seconds.")


# set enable_prefix_caching=True to enable APC
llm = LLM(
    model='Qwen/Qwen2-0.5B',
    enable_prefix_caching=True,
    # device='cpu',
)

# max tokens is decode length
# sampling_params = SamplingParams(temperature=0, max_tokens=10)
sampling_params = SamplingParams(temperature=0, min_tokens=1800, max_tokens=1800, seed=42, top_k=1, top_p=0.001)
# sampling_params = SamplingParams(temperature=0, min_tokens=208, max_tokens=208, seed=42, top_k=1, top_p=0.001)


# # Querying the age of John Doe
# get_generation_time(
#     llm,
#     sampling_params,
#     LONG_PROMPT + "Question: what is the age of John Doe? Your answer: The age of John Doe is ",
# )

# # Querying the age of Zack Blue
# # This query will be faster since vllm avoids computing the KV cache of LONG_PROMPT again.
# get_generation_time(
#     llm,
#     sampling_params,
#     LONG_PROMPT + "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is ",
# )

# get_generation_time(
#     llm,
#     sampling_params,
#     "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is "+LONG_PROMPT,
# )

prompts = []
for i in range(1):
    # prompts.append(f"Hello, Hello, Hello, Hello,{i}")
    # prompts.append(f"Hello, What is your name in model {i}?")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello"*100)
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello")
    prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello"*200)
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello,{i}")


from datetime import datetime
now = datetime.now()
milliseconds = int(now.timestamp() * 1000)
print(f"[datenlord profiling]: start timestamps {milliseconds} ms")
get_generation_time(
    llm,
    sampling_params,
    prompts
)
now = datetime.now()
milliseconds = int(now.timestamp() * 1000)
print(f"[datenlord profiling]: end timestamps {milliseconds} ms")

print("===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================")

# get_generation_time(
#     llm,
#     sampling_params,
#     prompts
# )

# prompts = []
# for i in range(1):
    # prompts.append(f"Hello, Hello, Hello, Hello,{i}")
    # prompts.append(f"Hello, What is your name in model {i}?")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello You You")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello,{i}")

# get_generation_time(
#     llm,
#     sampling_params,
#     prompts
# )

# print("===============================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================")

# get_generation_time(
#     llm,
#     sampling_params,
#     prompts
# )