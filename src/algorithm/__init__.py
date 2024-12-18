import numpy as np
import torch


def to_array(input_array, array_type=0):
    if array_type == 0:
        if isinstance(input_array, np.ndarray):
            return input_array
        elif isinstance(input_array, torch.Tensor):
            return input_array.detach().numpy()
        return np.array(input_array)
    elif array_type == 1:
        if isinstance(input_array, np.ndarray):
            return torch.from_numpy(np.array(input_array))
        elif isinstance(input_array, torch.Tensor):
            return input_array
        elif isinstance(input_array, list):
            try:
                result = torch.from_numpy(np.array(input_array))
                return result
            except:
                try:
                    result = torch.tensor(input_array)
                    return result
                except:
                    return torch.from_numpy(np.array([item.cpu().detach().numpy() for item in input_array]))
        return torch.from_numpy(np.array(input_array))


def util(array_type=0):
    if array_type == 0:
        return np
    else:
        return torch


set_array_type = 1
array_function = lambda input_array: to_array(input_array, set_array_type)
array_util = util(set_array_type)
