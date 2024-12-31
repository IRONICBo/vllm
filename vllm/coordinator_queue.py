from collections import deque
from typing import Dict, List, Optional, Tuple, Union

# Datenlord flag
datenlord_flag = False
is_prefill_process = True

PrefixHash = int

# PrefixTokenIds is a tuple of token ids that are used as a prefix for the
PrefixTokenIds = Tuple[int, ...]


# Global queue
swap_in_global_queue = deque()
swap_in_done_global_queue = deque()
swap_out_global_queue = deque()

# Global mapping of prefix hash to prefix token ids
global_prefix_hash_to_prefix_token_ids: Dict[PrefixHash, PrefixTokenIds] = {}

# prefix token ids, current token ids, cpu physical id
DataType = Tuple[List[int], List[int], int]

def put_global_prefix_hash_to_prefix_token_ids(prefix_hash: PrefixHash, prefix_token_ids: PrefixTokenIds):
    """
    Append the prefix hash to prefix token ids mapping to the global mapping
    :param prefix_hash: The hash of the prefix token ids
    :param prefix_token_ids: The prefix token ids
    """
    global_prefix_hash_to_prefix_token_ids[prefix_hash] = prefix_token_ids

def  get_global_prefix_hash_to_prefix_token_ids(prefix_hash: PrefixHash) -> Optional[PrefixTokenIds]:
    """
    Retrieve the prefix token ids from the global mapping
    :param prefix_hash: The hash of the prefix token ids
    :return: The prefix token ids if available, else None
    """
    return global_prefix_hash_to_prefix_token_ids.get(prefix_hash)

def swap_in_produce(data: DataType):
    """
    Produce data to the global queue
    :param data: A tuple consisting of two lists of integers and one integer
    """
    if isinstance(data, tuple) and len(data) == 3:
        if (isinstance(data[0], list) and isinstance(data[1], list) and
            isinstance(data[2], int)):
            swap_in_global_queue.append(data)
            print(f"swap_in_produce Produced: {data}")
        else:
            raise ValueError("Data must be in the format: (List[int], List[int], int)")
    else:
        raise TypeError("Input data must be a tuple of the form (List[int], List[int], int)")

def swap_in_consume() -> Optional[DataType]:
    """
    Consume data from the global queue.
    :return: A tuple of the format (List[int], List[int], int) if available, else None
    """
    if swap_in_global_queue:
        data = swap_in_global_queue.popleft()
        if isinstance(data, tuple) and len(data) == 3:
            if (isinstance(data[0], list) and isinstance(data[1], list) and
                isinstance(data[2], int)):
                print(f"swap_in_consume Consumed: {data}")
                return data
            else:
                raise ValueError("Data retrieved is not in the format: (List[int], List[int], int)")
        else:
            raise TypeError("Data retrieved is not a tuple with 3 elements")
    else:
        print("swap_in_consume Queue is empty!")
        return None

def swap_in_done_produce(data: DataType):
    """
    Produce data to the global queue
    :param data: A tuple consisting of two lists of integers and one integer
    """
    if isinstance(data, tuple) and len(data) == 3:
        if (isinstance(data[0], list) and isinstance(data[1], list) and
            isinstance(data[2], int)):
            swap_in_done_global_queue.append(data)
            print(f"swap_in_done_produce Produced: {data}")
        else:
            raise ValueError("Data must be in the format: (List[int], List[int], int)")
    else:
        raise TypeError("Input data must be a tuple of the form (List[int], List[int], int)")


def swap_in_done_consume() -> Optional[DataType]:
    """
    Consume data from the global queue.
    :return: A tuple of the format (List[int], List[int], int) if available, else None
    """
    if swap_in_done_global_queue:
        data = swap_in_done_global_queue.popleft()
        if isinstance(data, tuple) and len(data) == 3:
            if (isinstance(data[0], list) and isinstance(data[1], list) and
                isinstance(data[2], int)):
                print(f"swap_in_done_consume Consumed: {data}")
                return data
            else:
                raise ValueError("Data retrieved is not in the format: (List[int], List[int], int)")
        else:
            raise TypeError("Data retrieved is not a tuple with 3 elements")
    else:
        print("swap_in_done_consume Queue is empty!")
        return None

def swap_out_produce(data: DataType):
    """
    Produce data to the global queue
    :param data: A tuple consisting of two lists of integers and one integer
    """
    if isinstance(data, tuple) and len(data) == 3:
        if (isinstance(data[0], list) and isinstance(data[1], list) and
            isinstance(data[2], int)):
            swap_out_global_queue.append(data)
            print(f"swap_out_produce Produced: {data}")
        else:
            raise ValueError("Data must be in the format: (List[int], List[int], int)")
    else:
        raise TypeError("Input data must be a tuple of the form (List[int], List[int], int)")

def swap_out_consume() -> Optional[DataType]:
    """
    Consume data from the global queue.
    :return: A tuple of the format (List[int], List[int], int) if available, else None
    """
    if swap_out_global_queue:
        data = swap_out_global_queue.popleft()
        if isinstance(data, tuple) and len(data) == 3:
            if (isinstance(data[0], list) and isinstance(data[1], list) and
                isinstance(data[2], int)):
                print(f"swap_out_consume Consumed: {data}")
                return data
            else:
                raise ValueError("Data retrieved is not in the format: (List[int], List[int], int)")
        else:
            raise TypeError("Data retrieved is not a tuple with 3 elements")
    else:
        print("swap_out_consume Queue is empty!")
        return None