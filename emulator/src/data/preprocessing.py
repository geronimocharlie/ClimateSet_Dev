""" Script containing custom preprocessing functions which are called variable-vise before loading data into memory"""

from torch import Tensor
import torch
from typing import Dict, List, Union, Tuple


def resolve_preprocessing(data: Tensor, mapping: Tuple(str, int))->Tensor:

    """
    Function to resolve mapping from config arguments to actual preprocessing function.
    Any new functions must be implemented within this script and added to this interface function for resolving link.
    """
    function_name, argument = mapping

    if function_name=="accum":
        return accum(data, argument)
    # NEW FUNCTIONS GO HERE
    # e.g. if function_name=='new_function':...
    else:
        print(f"Unknown preprocessing function {function_name}. Not applying any preprocessing")
        return data


def accum_time(data: Tensor, time_window: int =-1):
    # accumulated sum over time dimension only
    # window is given in years 
    # first axis is num_samples, second is sequence lenght, 3rd var, rest spatial
    # we need to resolve the window shape (multiply by frequency!), flatten out first two dimensions, than reshape back

    # get seq_len
    seq_len=data.size()[1]
    # recompute time window
    time_window = time_window * seq_len

    print("in accum data shape")
    print(data.size())

    # flatten out first two axis
    data_full_sequence=data.view(1,-1,*(data.size()[2:]))

    # make cumsum
    data_full_sequence_cumsum=torch.cumsum(data_full_sequence,1)

    # if window make window cumsum
    if window_size>0:
        data_full_sequence_cumsum[window_size:] = data_full_sequence_cumsum[window_size:] - data_full_sequence_cumsum[:-window_size]

    # reshape back
    data_final=data_full_sequence_cumsum.view(-1,seq_len,*data_full_sequence_cumsum.size()[2:])

    print("final data size")
    print(data_final.size())
    return data_final