"""CacheEngine class for managing the KV cache."""
from typing import List

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)

def print_io(func):
    def wrapper(*args, **kwargs):
        print(f"INPUT: func={func}, args={args}, kwargs={kwargs}")
        result = func(*args, **kwargs)
        print(f"OUTPUT: {result}")
        return result
    return wrapper


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Get attention backend.
        self.attn_backend = get_attn_backend(self.head_size,
                                             model_config.dtype,
                                             cache_config.cache_dtype,
                                             self.block_size,
                                             model_config.is_attention_free)

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

        # import sys
        # import pympler
        # print(f"sys.getsizeof(self.gpu_cache): {sys.getsizeof(self.gpu_cache)} total memory size: {pympler.asizeof.asizeof(self.gpu_cache)} bytes")
        # print(f"sys.getsizeof(self.cpu_cache): {sys.getsizeof(self.cpu_cache)} total memory size: {pympler.asizeof.asizeof(self.cpu_cache)} bytes")

        # message
        import threading
        self.thread = threading.Thread(target=self.watch_and_swap_in, daemon=True)
        self.thread.start()

        self.thread2 = threading.Thread(target=self.watch_and_swap_out, daemon=True)
        self.thread2.start()


    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            block = torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device)
            torch.save
            import sys
            from pympler import asizeof
            print(f"attn_backend: {self.attn_backend} kvcache shape: {kv_cache_shape} device: {device} sys.getsizeof(block): {sys.getsizeof(block)} Total memory size: {asizeof.asizeof(block)} bytes")

            kv_cache.append(block)
        return kv_cache

    def watch_and_swap_in(self) -> None:
        """Watches the global queue and swaps in the cache."""
        print("watch_and_swap_in watch_and_swap_in started...")
        from vllm.coordinator_queue import swap_in_consume
        import time
        # Try to allocate a mocked
        # while True:
        while True:
            data = swap_in_consume()
            if data is None:
                time.sleep(1)
                continue

            print(f"watch_and_swap_in swap_in_consume Consumed: {data}")
            # read the cache from disk
            prefix_token_ids, current_token_ids, gpu_physical_id = data
            key = prefix_token_ids + current_token_ids
            # import hashlib
            # key = hashlib.md5(key.encode()).hexdigest()
            print(f"self.num_attention_layers: {self.num_attention_layers}")
            import threading
            lock = threading.Lock()

            from vllm.coordinator_queue import sdk
            from io import BytesIO
            (match_key, value) = sdk.try_load_sync(key)
            if match_key is None:
                print(f"watch_and_swap_in: key {key} not found in datenlord")
                continue

            memoryview_value = memoryview(value)
            memoryview_value = memoryview_value.tobytes()
            memoryview_value = BytesIO(memoryview_value)
            kv_cache = torch.load(memoryview_value)

            with lock:
                for i in range(self.num_attention_layers):
                    # filename = f"/home/lvbo/project/vllm/kvcache_dump/data/cache_{key}_{i}.bin"
                    # with open(filename, "rb") as f:
                    # print(f"swap_in key: {key} filename: {filename}")
                    # convert data to buffer
                    start = time.time()
                    # from head
                    # f.seek(0)
                    # buf = f.read()
                    # print(f"file read: {time.time() - start} shape of raw_data: {len(buf)}")
                    # mock buffer
                    # buf = bytes(4096)
                    # buf = bytes(4096)
                    # buf = bytearray([2] * 4096)
                    # print(f"Time to read from disk: {time.time() - start} shape of raw_data: {len(buf)}")

                    # kv_cache = torch.frombuffer(buf, dtype=self.dtype)
                    # kv_cache = torch.zeros((2, 1, 1024), dtype=self.dtype)
                    # kv_cache = kv_cache.reshape((2, 1024))
                    # print(f"watch and swap in: restore kv_cache shape: {kv_cache.shape} buffer max data: {kv_cache.max()} min data: {kv_cache.min()}")
                    # kv_cache = kv_cache.reshape(self.attn_backend.get_kv_cache_shape(
                    #     1, self.block_size, self.num_kv_heads, self.head_size))
                    # print(f"watch and swap in: restore kv_cache shape: {kv_cache.shape} buffer max data: {kv_cache.max()} min data: {kv_cache.min()}")

                    # copy data to cpu cache
                    # choose the last block as the target block
                    # cpu_physical_id = self.cpu_cache[i].shape[1] - 1
                    # # cpu cache shape? 2, 1, 1024
                    # self.cpu_cache[i][:, cpu_physical_id, :] = kv_cache[0, :, :]
                    # # print(f"swap_in key: {key} filename: {filename} cpu_physical_id {cpu_physical_id} in cache: {self.cpu_cache[i]}")
                    # print(f"swap_in key: {key} filename: {filename} buffer max data: {kv_cache.max()} min data: {kv_cache.min()}")
                    # print(f"swap_in Consumed: {data} buffer max data: {kv_cache.max()} min data: {kv_cache.min()}")

                    # Direct copy buffer to gpu kv cache
                    self.gpu_cache[i][:, gpu_physical_id, :] = kv_cache[i].cuda()
                    print(f"swap_in key: {key} kv_cache shape: {kv_cache[i].shape} gpu_physical_id {gpu_physical_id} in cache: {self.gpu_cache[i]}")
                    # kv_cache.squeeze(1).cuda() [2, 1, 1024] -> [2, 1024]
                    # self.gpu_cache[i][:, gpu_physical_id, :] = kv_cache.squeeze(1).cuda()
                    # print(f"swap_in key: {key} filename: {filename} gpu_physical_id {gpu_physical_id} gpu cache shape: {self.gpu_cache[i].shape} in cache: {self.gpu_cache[i]}")
                    # print(f"raw swap_in key: {key} filename: {filename} gpu_physical_id {gpu_physical_id} gpu cache shape: {self.gpu_cache[i].shape} in self.gpu_cache[i][:, gpu_physical_id, :]: {self.gpu_cache[i][:, gpu_physical_id, :]}")

                    # torch save for check
                    # torch.save(self.gpu_cache[i][:, gpu_physical_id, :], f"/home/lvbo/project/vllm/kvcache_dump/check/new_cache_{key}_{i}.pt")

                    # use check2 to check the data
                    # kv_cache = torch.load(f"/home/lvbo/project/vllm/kvcache_dump/check2/cache_{key}_{i}.pt")
                    # print(f"from tensor: watch and swap in: restore kv_cache shape: {kv_cache.shape} buffer max data: {kv_cache.max()} min data: {kv_cache.min()}")
                    # print(f"raw swap_in key: {key} filename: {filename} gpu_physical_id {gpu_physical_id} gpu cache shape: {self.gpu_cache[i].shape} in check2: {kv_cache}")
                    # self.gpu_cache[i][:, gpu_physical_id, :] = kv_cache

                    print(f"Time to convert to buffer: {time.time() - start} shape of raw_data: {len(kv_cache)}")

            # swap in from cpu to gpu
            # print(f"swap_in Consumed: {data} pointer: cpu: {cpu_physical_id} -> gpu:{gpu_physical_id}")
            # src_to_dst = torch.tensor([[cpu_physical_id, gpu_physical_id]], dtype=torch.int64)
            # self.swap_in(src_to_dst)
            # print(f"swap_in self.swap_in(src_to_dst): {data}: pointer: {src_to_dst}")

            from vllm.coordinator_queue import swap_in_done_produce
            swap_in_done_produce(data)

    def watch_and_swap_out(self) -> None:
        """Watches the global queue and swaps out the cache."""
        print("watch_and_swap_out started...")
        from vllm.coordinator_queue import swap_out_consume
        import time
        # Try to allocate a mocked
        while True:
            data = swap_out_consume()
            if data is None:
                time.sleep(1)
                continue

            print(f"swap_out_consume Consumed: {data}")
            # save the swapped out cache to disk with memoryview
            prefix_token_ids, current_token_ids, cpu_physical_id = data
            key = prefix_token_ids + current_token_ids
            print(f"Current key: {key}")
            # import hashlib
            # key = hashlib.md5(key.encode()).hexdigest()
            print(f"self.num_attention_layers: {self.num_attention_layers}")

            import io
            buffer = io.BytesIO()
            temp_cache = []
            import threading
            lock = threading.Lock()
            with lock:
                for i in range(self.num_attention_layers):
                    # num_blocks, self.block_size, self.num_kv_heads, self.head_size
                    # convert data to buffer
                    import time
                    start = time.time()
                    slice = self.gpu_cache[i][:, cpu_physical_id, :]
                    temp_cache.append(slice)
                    start = time.time()
                    print(f"Layer {i}: Saved to buffer, max: {slice.max()}, min: {slice.min()}")
                    print(f"Time to save layer {i} to buffer: {time.time() - start}")

                torch.save(temp_cache, buffer)
                print(f"Time to save all layers to buffer: {time.time() - start} shape of raw_data: {len(buffer.getvalue())}")

                from vllm.coordinator_queue import sdk
                sdk.insert_sync(key, buffer.getvalue())
                # write twice to check
                sdk.insert_sync(key, buffer.getvalue())


    @print_io
    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        print(f"src_to_dst: {src_to_dst}")
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    @print_io
    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        from vllm.coordinator_queue import swap_out_consume

        print(f"src_to_dst: {src_to_dst}")
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)
        # Try to allocate a mocked
        # used
        # from vllm.coordinator_queue import datenlord_flag
        # if not datenlord_flag:
        #     return
        # while (data := swap_out_consume()) is not None:
        #     print(f"swap_out Consumed: {data}")
        #     # save the swapped out cache to disk with memoryview
        #     prefix_token_ids, current_token_ids, cpu_physical_id = data
        #     key = str(prefix_token_ids + current_token_ids)
        #     # filename too long
        #     import hashlib
        #     key = hashlib.md5(key.encode()).hexdigest()
        #     print(f"self.num_attention_layers: {self.num_attention_layers}")
        #     for i in range(self.num_attention_layers):
        #         with open(f"/home/lvbo/project/vllm/kvcache_dump/data/cache_{key}_{i}.bin", "wb") as f:
        #             # num_blocks, self.block_size, self.num_kv_heads, self.head_size
        #             # convert data to buffer
        #             import ctypes
        #             import time
        #             start = time.time()
        #             slice = self.gpu_cache[i].cpu()[:, cpu_physical_id, :]
        #             data_ptr = slice.data_ptr()
        #             data_size = slice.numel() * slice.element_size()
        #             buffer = (ctypes.c_char * data_size).from_address(data_ptr)
        #             raw_data = bytes(buffer)
        #             print(f"/home/lvbo/project/vllm/kvcache_dump/data/cache_{key}_{i}.binbuffer max data: {self.cpu_cache[i][:, cpu_physical_id, :].max()} min data: {self.cpu_cache[i][:, cpu_physical_id, :].min()}")
        #             print(f"Time to convert to buffer: {time.time() - start} shape of raw_data: {len(raw_data)}")
        #             f.write(raw_data)

        # save the swapped out cache to disk with memoryview
        # for i in range(self.num_attention_layers):
        #     with open(f"/home/lvbo/project/vllm/kvcache_dump/data/cache_{i}.bin", "wb") as f:
        #         # convert data to buffer
        #         import ctypes
        #         import time
        #         start = time.time()
        #         data_ptr = self.cpu_cache[i].data_ptr()
        #         data_size = self.cpu_cache[i].numel() * self.cpu_cache[i].element_size()
        #         buffer = (ctypes.c_char * data_size).from_address(data_ptr)
        #         raw_data = bytes(buffer)
        #         print(f"Time to convert to buffer: {time.time() - start} shape of raw_data: {len(raw_data)}")
        #         f.write(raw_data)

    @print_io
    def copy(self, src_to_dsts: torch.Tensor) -> None:
        print(f"src_to_dst: {src_to_dsts}")
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total
