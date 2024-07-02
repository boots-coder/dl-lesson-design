import matplotlib.pyplot as plt

# Data
mean_ap = 0.8898430900359435
ap_per_class = [0.9935897435897437, 0.9204081253713609, 0.9880952380952382, 0.8832621082621083, 0.7805419389978215, 0.773161385899389]
classes = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6']

# Plotting
plt.figure(figsize=(10, 6))
bars = plt.bar(classes, ap_per_class, color='skyblue')
plt.axhline(y=mean_ap, color='r', linestyle='--', label=f'Mean AP: {mean_ap:.2f}')
plt.xlabel('Classes')
plt.ylabel('AP')
plt.title('AP per Class and Mean AP')
plt.legend()

# Adding text labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom')

plt.show()