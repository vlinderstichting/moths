from typing import List

import wandb
from torch import Tensor


def log_wandb_confusion_matrix(phase_name: str, class_names: List[str], preds: Tensor, targets: Tensor) -> None:
    wandb.log(
        {
            f"{phase_name}-confusion-matrix": wandb.plot.confusion_matrix(
                probs=preds,
                y_true=targets,
                preds=None,
                class_names=class_names,
            )
        }
    )


#
#
# class Metric(ABC):
#     def __init__(self, **kwargs) -> None:
#         self.values = defaultdict(list)
#
#
#     def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
#         batch_values = []
#
#         classes = torch.unique(torch.stack([pred, target]))
#         for c in classes:
#             p = pred == c
#             t = target == c
#
#             tp = torch.logical_and(p, t).sum()
#             tn = torch.logical_and(~p, ~t).sum()
#             fp = torch.logical_and(p, ~t).sum()
#             fn = torch.logical_and(~p, t).sum()
#
#             support = p.sum()
#             value = self.calculate(tp, tn, fp, fn)
#
#             value_tensor = torch.stack([value, support])
#
#             self.values[c].append(value_tensor)
#             batch_values.append(value_tensor)
#
#         batch_value = torch.stack(batch_values)
#         batch_value = (batch_value[0, :] * batch_value[1, :]) / batch_value[:, 1].sum()
#
#         return batch_value
#
#     def compute(self) -> Tensor:
#
#         num_classes = max(self.values.keys())
#         # values = torch.zeros((num_classes, 2), )
#         # for c in self.values.keys():
#
#
#
#         values = torch.stack(self.values)
#         weights = torch.stack(self.weights)
#         return (values * weights) / weights.sum()
#
#     def reset(self) -> None:
#         self.values = []
#         self.weights = []
#
#     @abstractmethod
#     def calculate(self, tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
#         pass
#
#
# class Accuracy(Metric):
#     def calculate(self, tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
#         return (tp + tn) / (tp + tn + fn + fp)
#
#
# class Precision(Metric):
#     def calculate(self, tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
#         return tp / (tp + fp)
#
#
# class Recall(Metric):
#     def calculate(self, tp: Tensor, tn: Tensor, fp: Tensor, fn: Tensor) -> Tensor:
#         return tp / (tp + fn)
