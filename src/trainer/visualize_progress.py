import matplotlib.pyplot as plt
import pandas as pd

# Load the training progress CSV
df = pd.read_csv(
    './training_progress.csv',)

# Use training iteration as the x-axis
x = df['training_iteration']

# Identify subloss metrics (everything except total)
subloss_metrics = [col.split('/', 1)[1] for col in df.columns if col.startswith('train/') and col != 'train/total']

# Plot each subloss
for metric in subloss_metrics:
    plt.figure()
    plt.plot(x, df[f'train/{metric}'], label=f'Train {metric}')
    plt.plot(x, df[f'val/{metric}'], label=f'Val {metric}')
    plt.xlabel('Training Iteration')
    plt.ylabel(metric)
    plt.title(f'Training and Validation {metric}')
    plt.legend()
    plt.show()
