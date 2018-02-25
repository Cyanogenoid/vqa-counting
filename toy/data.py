import torch
import torch.utils.data as data
import random


class ToyTask(data.Dataset):
    """
    This toy task is intended to test the robustness of the approach, not so much to be "fair" to other baselines.
    """
    def __init__(self, max_objects, coord, noise):
        super().__init__()
        self.max_objects = max_objects
        self.max_proposals = self.max_objects
        self.max_coord = max(coord, 1e-6)
        self.weight_noise = noise

    def __getitem__(self, item):
        # generate random object positions
        objects = torch.rand(self.max_proposals, 2) * (1 - self.max_coord)
        # generate object boxes, to make sure that all objects are covered
        boxes = torch.cat([objects, objects + self.max_coord], dim=1)
        # determine selected objects
        count = random.randint(0, self.max_objects)
        if count > 0:
            true_boxes = boxes[:count]
            # find the iou distance to the true objects
            iou = self.iou(boxes.t().contiguous(), true_boxes.t().contiguous())
        else:
            # no true objects, so no true overlaps to compute
            iou = torch.zeros(self.max_proposals, 1)
        # determine weighting by using each box' most overlapping true box
        weights = self.weight(iou.max(dim=1)[0])
        return weights, boxes, count

    def weight(self, x):
        noise = torch.rand(x.size())
        # linear interpolation between signal and noise
        x = (1 - self.weight_noise) * x + self.weight_noise * noise
        return x

    def iou(self, a, b):
        inter = self.intersection(a, b)
        area_a = self.area(a).unsqueeze(1).expand_as(inter)
        area_b = self.area(b).unsqueeze(0).expand_as(inter)
        return inter / (area_a + area_b - inter)

    def area(self, box):
        x = (box[2, :] - box[0, :]).clamp(min=0)
        y = (box[3, :] - box[1, :]).clamp(min=0)
        return x * y

    def intersection(self, a, b):
        size = (2, a.size(1), b.size(1))
        min_point = torch.max(
            a[:2, :].unsqueeze(dim=2).expand(*size),
            b[:2, :].unsqueeze(dim=1).expand(*size),
        )
        max_point = torch.min(
            a[2:, :].unsqueeze(dim=2).expand(*size),
            b[2:, :].unsqueeze(dim=1).expand(*size),
        )
        inter = (max_point - min_point).clamp(min=0)
        area = inter[0, :, :] * inter[1, :, :]
        return area

    def __len__(self):
        # "infinite" size dataset, so just return a big number
        return 2**32
