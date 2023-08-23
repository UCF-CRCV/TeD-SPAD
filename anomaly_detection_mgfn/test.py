from torch.utils.data import DataLoader
import option
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
from tqdm import tqdm
args=option.parse_args()
from config import *
from models.mgfn import mgfn as Model
from datasets.dataset import Dataset


def test(dataloader, model, args, device):
    plt.clf()
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)
        featurelen = []
        for i, inputs in enumerate(dataloader):

            input = inputs[0].to(device)
            input = input.permute(0, 2, 1, 3)
            _, _, _, _, logits = model(input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            featurelen.append(len(sig))
            pred = torch.cat((pred, sig))

        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 32) # Skip rate of 2, 16 frames.
        ratio = float(len(list(gt))) / float(len(pred))
        # In case size mismatch btwn predictions and gt.
        if ratio == 1.0:
            final_pred = pred
        else:
            print(f'Ground truth not exact shape: {ratio}')
            final_pred = np.zeros_like(gt, dtype='float32')
            for i in range(len(pred)):
                b = int(i * ratio + 0.5)
                e = int((i + 1) * ratio + 0.5)
                final_pred[b:e] = pred[i]

        fpr, tpr, threshold = roc_curve(list(gt), list(final_pred), drop_intermediate=True)
        rec_auc = auc(fpr, tpr)
        precision, recall, th = precision_recall_curve(list(gt), list(final_pred))
        pr_auc = auc(recall, precision)
        print('pr_auc : ' + str(pr_auc))
        print('rec_auc : ' + str(rec_auc))
        return rec_auc, pr_auc


if __name__ == '__main__':
    args = option.parse_args()
    config = Config(args)
    device = torch.device("cuda")
    model = Model()
    model.eval()
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    model = model.to(device)
    model_dict = model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(args.pretrained_ckpt).items()})
    auc = test(test_loader, model, args, device)
