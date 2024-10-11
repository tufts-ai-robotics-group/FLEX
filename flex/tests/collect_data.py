import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from agent.td3 import TD3

N = 10000
angles = np.arange(-np.pi, np.pi, step=(np.pi*2)/N)
# print(len(X))
X = np.zeros((N, 6), dtype=np.float32)
X[:, 0] = np.cos(angles)
X[:, 1] = np.sin(angles)

rnd_progress = np.random.rand(N)
# print(rnd_progress)
X[:, 3] = rnd_progress * X[:, 0]
X[:, 4] = rnd_progress * X[:, 1]
y = X[:, :3] * 5 
# print(y)


class DummyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

dataset = DummyDataset(X=X, y=y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

agent = TD3(lr=1e-3, state_dim=6, action_dim=3, max_action=5)

criterion = torch.nn.MSELoss()

for epoch in range(10):
    losses = []
    for feat, labels in dataloader:
        feat, labels = feat.to('cuda'), labels.to('cuda')
        a_pred = agent.actor(feat)
        loss = criterion(a_pred, labels)
        agent.actor_optimizer.zero_grad()
        loss.backward()
        agent.actor_optimizer.step()
        losses.append(loss.item())
    print('loss: ', np.mean(losses))


agent.save('checkpoints/il_policies', 'prismatic_pretrained')

agent = TD3(lr=1e-3, state_dim=6, action_dim=3, max_action=5)

X1 = np.zeros((N, 6), dtype=np.float32)
y1 = np.zeros((N, 3), dtype=np.float32)

import numpy as np

def find_y(x1, x2):
    # Normalize x1 to create an orthonormal basis
    x1 = x1 / np.linalg.norm(x1)
    
    # Find a vector orthogonal to x1
    v1 = np.cross(x1, np.array([1, 0, 0]))  # Cross product of x1 and x-axis
    if np.linalg.norm(v1) < 1e-10:  # Check if x1 is parallel to the x-axis
        v1 = np.cross(x1, np.array([0, 1, 0]))  # Use y-axis if needed
    v1 = v1 / np.linalg.norm(v1)
    
    # Another orthogonal vector in the plane orthogonal to x1
    v2 = np.cross(x1, v1)
    
    # Now y is a linear combination of v1 and v2: y = a * v1 + b * v2
    # Use the cross product condition: x2 cross y is parallel to x1
    # Therefore, (x2 cross (a * v1 + b * v2)) is parallel to x1
    
    A = np.cross(x2, v1)  # Cross x2 with v1
    B = np.cross(x2, v2)  # Cross x2 with v2
    
    # Solve for a and b such that a * A + b * B is parallel to x1
    # We want (a * A + b * B) to be parallel to x1 => a * A + b * B = lambda * x1
    # This can be solved using the projection of A and B onto x1
    
    A_proj = np.dot(A, x1)
    B_proj = np.dot(B, x1)
    
    # We want to solve the system: a * A_proj + b * B_proj = c (some constant)
    # Without loss of generality, let c = 1 to find a ratio for a and b
    
    a = B_proj
    b = -A_proj
    
    # Now construct y using the calculated a and b
    y = a * v1 + b * v2
    
    # Normalize y to have length 5
    y = 5 * y / np.linalg.norm(y)
    
    return y

X[:, 2] = -1 
X[:, ]

# rev_dataset = DummyDataset