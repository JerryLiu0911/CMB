import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Model
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# Generate synthetic data
def generate_data(n_samples=5000):
    # Simulate 3 concepts: c0, c1, c2 (binary)
    c = np.random.randint(0, 2, size=(n_samples, 3))

    # Final label = 1 if (c0 and c1) or c2
    y = (np.logical_and(c[:, 0], c[:, 1]) | c[:, 2]).astype(int)

    # Simulate input features x (10D) = concept + noise
    x = c + 0.1 * np.random.randn(n_samples, 3)
    x = np.concatenate([x, np.random.randn(n_samples, 7)], axis=1)

    return x.astype('float32'), c.astype('float32'), y.astype('float32')

# Concept Encoder: x → ĉ
def build_concept_encoder(input_dim, concept_dim):
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.Dense(16, activation='relu')(x)
    c_hat = layers.Dense(concept_dim, activation='sigmoid')(x)  # output = concept predictions
    return Model(inputs, c_hat, name="ConceptEncoder")

# Label Predictor: ĉ → ŷ
def build_label_predictor(concept_dim):
    inputs = layers.Input(shape=(concept_dim,))
    x = layers.Dense(16, activation='relu')(inputs)
    y_hat = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, y_hat, name="LabelPredictor")

x, c, y = generate_data()
x_train, x_test, c_train, c_test, y_train, y_test = train_test_split(x, c, y, test_size=0.2)

# PCA to reduce to 2D
pca = PCA(n_components=2)
x_2d = pca.fit_transform(x)

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(x_2d[:, 0], x_2d[:, 1], c=y, cmap='coolwarm', alpha=0.6, edgecolor='k')
plt.title("PCA Projection of Input Features Colored by Label")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Label (0 = False, 1 = True)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Combine into CBM
input_dim = x.shape[1]
concept_dim = c.shape[1]

concept_encoder = build_concept_encoder(input_dim, concept_dim)
label_predictor = build_label_predictor(concept_dim)

x_input = layers.Input(shape=(input_dim,))
c_hat = concept_encoder(x_input)
y_hat = label_predictor(c_hat)

cbm = Model(x_input, y_hat, name="CBM")

# Combined loss: task + concept loss
cbm.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Optionally supervise concept encoder separately:
concept_encoder.compile(optimizer='adam', loss='binary_crossentropy')

# Supervised concept learning
concept_encoder.fit(x_train, c_train, epochs=10, batch_size=32, verbose=1)

# Freeze concept encoder if using independent bottleneck:
concept_encoder.trainable = False

# Train the label predictor on predicted concepts
cbm.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Predict concepts
c_pred = concept_encoder.predict(x_test[:5])

# Manual intervention (simulate correcting concept 0)
c_intervened = c_pred.copy()
c_intervened[:, 0] = c_test[:5, 0]  # use true value for concept 0

# Predict before and after intervention
y_original = label_predictor.predict(c_pred)
y_corrected = label_predictor.predict(c_intervened)

for i in range(5):
    print(f"Original: {y_original[i][0]:.2f} → Corrected: {y_corrected[i][0]:.2f}")

# 1. Visualize Predicted Concepts vs True Concepts (for test set)
c_pred_test = concept_encoder.predict(x_test)
c_pred_binary = (c_pred_test > 0.5).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
concept_names = [f"Concept {i}" for i in range(c.shape[1])]

for i in range(c.shape[1]):
    ax = axes[i]
    ax.scatter(c_test[:, i], c_pred_test[:, i], alpha=0.5)
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlabel(f"True {concept_names[i]}")
    ax.set_ylabel(f"Predicted {concept_names[i]}")
    ax.set_title(f"{concept_names[i]}: True vs Predicted")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
plt.tight_layout()
plt.show()

# 2. Label prediction changes after interventions (for all test samples)
c_intervened = c_pred_test.copy()

# Let's try intervening on each concept one by one and observe average label change
for concept_idx in range(c.shape[1]):
    c_intervened[:, concept_idx] = c_test[:, concept_idx]  # intervene on one concept at a time

    y_original = label_predictor.predict(c_pred_test).flatten()
    y_corrected = label_predictor.predict(c_intervened).flatten()

    change = y_corrected - y_original
    avg_change = np.mean(np.abs(change))
    print(f"Intervention on Concept {concept_idx} changes average label prediction by {avg_change:.4f}")

    # Reset for next concept
    c_intervened[:, concept_idx] = c_pred_test[:, concept_idx]

# For visualization, pick first 20 samples for concept 0 intervention:
c_intervened_0 = c_pred_test[:20].copy()
c_intervened_0[:, 0] = c_test[:20, 0]

y_original_20 = label_predictor.predict(c_pred_test[:20]).flatten()
y_corrected_20 = label_predictor.predict(c_intervened_0).flatten()

plt.figure(figsize=(8, 5))
plt.plot(range(20), y_original_20, 'bo-', label='Original prediction')
plt.plot(range(20), y_corrected_20, 'ro-', label='After intervention on Concept 0')
plt.xlabel("Sample index")
plt.ylabel("Predicted label probability")
plt.legend()
plt.title("Label Predictions Before and After Intervention on Concept 0")
plt.grid(True)
plt.show()

# 3. Confusion Matrices

# Concepts confusion matrices (multi-label): show per-concept confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i in range(c.shape[1]):
    cm = confusion_matrix(c_test[:, i], c_pred_binary[:, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
    axes[i].set_title(f"Confusion Matrix for {concept_names[i]}")
plt.tight_layout()
plt.show()

# Label confusion matrix
y_pred_label = (label_predictor.predict(c_pred_test) > 0.5).astype(int)
cm_label = confusion_matrix(y_test, y_pred_label)
disp_label = ConfusionMatrixDisplay(confusion_matrix=cm_label, display_labels=[0, 1])
disp_label.plot(cmap='Greens')
plt.title("Confusion Matrix for Label Prediction")
plt.show()


# 4. Decision Boundary plot in concept space (2D or 3D)

# Since concept space is 3D (concept_dim=3), let's try to visualize decision boundaries
# by fixing one concept and plotting over the other two

def plot_decision_boundary(label_predictor, fixed_concept_idx=2, fixed_value=0):
    # Create grid over concept 0 and 1
    resolution = 100
    c0_vals = np.linspace(0, 1, resolution)
    c1_vals = np.linspace(0, 1, resolution)
    c_grid = np.zeros((resolution * resolution, 3))

    c_grid[:, 0] = np.repeat(c0_vals, resolution)
    c_grid[:, 1] = np.tile(c1_vals, resolution)
    c_grid[:, fixed_concept_idx] = fixed_value

    y_grid_pred = label_predictor.predict(c_grid).reshape((resolution, resolution))

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(c0_vals, c1_vals, y_grid_pred, levels=50, cmap='coolwarm', alpha=0.8)
    plt.colorbar(contour, label='Predicted label probability')
    plt.xlabel('Concept 0')
    plt.ylabel('Concept 1')
    plt.title(f"Decision boundary fixing Concept {fixed_concept_idx} = {fixed_value}")

    # Overlay test points for concepts 0 and 1 where fixed concept equals fixed_value (with some tolerance)
    mask = np.isclose(c_test[:, fixed_concept_idx], fixed_value, atol=0.1)
    plt.scatter(c_test[mask, 0], c_test[mask, 1], c=y_test[mask], edgecolor='k', cmap='coolwarm', marker='o',
                label='Test points')
    plt.legend()
    plt.show()


plot_decision_boundary(label_predictor, fixed_concept_idx=2, fixed_value=0)
plot_decision_boundary(label_predictor, fixed_concept_idx=2, fixed_value=1)





