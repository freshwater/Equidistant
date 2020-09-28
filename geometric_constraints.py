
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeometricConstraints(nn.Module):
    def __init__(self, points, dimensions, constraints, iterations=2000, device=None):
        super().__init__()

        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self._points = torch.nn.Parameter(torch.randn(points, dimensions, device=self.device))
        self.constraints = constraints

        self.run(iterations=iterations)

    def step(self):
        distances = F.pdist(self._points, p=2)

        loss = 0
        for constraint in self.constraints:
            loss += constraint(self._points, distances)

        return loss

    def run(self, iterations):
        self.train()

        optimizer = torch.optim.AdamW(self.parameters())

        for iteration in range(iterations):
            optimizer.zero_grad()

            loss = self.step()
            loss.backward()

            optimizer.step()

    def distances(self):
        return F.pdist(self._points).detach().cpu().numpy()

    def points(self):
        return self._points.detach().cpu().numpy()
