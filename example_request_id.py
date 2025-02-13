import traceback

import torch

torch.set_printoptions(threshold=torch.inf)

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
        print(f"output >=1 Output: [ {output[0].outputs[0].text} ]")
    else:
        print(f"output < 1 Output: [ {output} ]")
    # print(f"Output: {output[0].outputs[0].text}")
    print(f"Generation time: {end_time - start_time} seconds.")


# set enable_prefix_caching=True to enable APC
llm = LLM(
    # model='lmsys/longchat-13b-16k',
    model="Qwen/Qwen2-0.5B",
    enable_prefix_caching=True,
    # device='cpu',
)

# max tokens is decode length
# sampling_params = SamplingParams(temperature=0, max_tokens=10)
sampling_params = SamplingParams(
    temperature=0, max_tokens=10, seed=42, top_k=1, top_p=0.001
)


prompts = []
for i in range(10):
    # prompts.append(f"Hello, Hello, Hello, Hello,{i}")
    # prompts.append(f"Hello, What is your name in model {i}?")
    prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello Hello")
    # prompts.append(f"Hello,{i}")


get_generation_time(
    llm,
    sampling_params,
    # f"{i}"+LONG_PROMPT + "Question: what is the age of Zack Blue? Your answer: The age of Zack Blue is ",
    prompts,
)
