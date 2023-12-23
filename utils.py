import torch


def format_target(target, pred_fault_loc=False):
    '''
    Converts int/float targets to multi-label targets. (i.e., 0 to [0,0,0,0], 1 to [1,1,0,0], 
    2 to [1,0,1,0], and [1,0,0,1]). The first item corresponds to whether there is an imbalance in
    the system or not, while the last 3 values correspond to the mass location of the imabalance
    '''
    # target = torch.tensor(target)

    # Get class output
    output = torch.where(target == 0, 0, 1).reshape(-1, 1)

    # Get fault location output if enabled
    if pred_fault_loc:
        output = torch.concat([output, torch.where(
            target == 1, 1, 0).reshape(-1, 1)], dim=-1)
        output = torch.concat([output, torch.where(
            target == 2, 1, 0).reshape(-1, 1)], dim=-1)
        output = torch.concat([output, torch.where(
            target == 3, 1, 0).reshape(-1, 1)], dim=-1)

    return output.type(torch.float32)
