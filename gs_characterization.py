import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F

N = 10
TEMP = 0.1

prob = torch.tensor([0.25, 0.25, 0.25, 0.25])
logits = torch.log(
    prob / (1 - prob)
    )

val = []
for i in range(N):
    val.append(F.gumbel_softmax(logits, tau=TEMP, dim=0).detach().numpy())

data = pd.DataFrame(val, columns=['0', '2', '4', '8'])
sns.lineplot(data=data, palette='tab10', linewidth=2.5)
plt.title(f'Gumbel Softmax Characterization - Temp={TEMP}')
plt.savefig(f'gs_characterization_T-{TEMP}.png')