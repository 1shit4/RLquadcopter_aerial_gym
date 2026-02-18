import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trajectory_tracking_1770037861.csv')
plt.plot(df['target_x'], df['target_y'], label='Target')
plt.plot(df['actual_x'], df['actual_y'], label='Actual')
plt.legend()
plt.show()