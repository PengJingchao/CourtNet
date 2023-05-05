import numpy as np



def calculateF1Measure(output_image, gt_image, thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image > thre
    gt_bin = gt_image > thre
    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    precision = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * precision / np.maximum(0.001, recall + precision)
    return F1


def calculatePreRecF1Measure(output_image, gt_image, thre):
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image > thre
    gt_bin = gt_image
    recall = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(gt_bin))
    precision = np.sum(gt_bin * out_bin) / np.maximum(1, np.sum(out_bin))
    F1 = 2 * recall * precision / np.maximum(0.001, recall + precision)
    return precision, recall, F1
  
  
def save_checkpoint(state, filename='checkpoint'):
    filename = '{}.pth'.format(filename)
    torch.save(state, filename)


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.DataParallel):
            model_state = model.module.state_dict()
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state}
  
  
  
