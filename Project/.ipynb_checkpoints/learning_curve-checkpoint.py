from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

#NOTE THAT THIS FUNCTION IS NOT USED IN THE FINAL IMPLEMENTATION - SMALL SUBSET OF DATA USED FOR TESTING
small_sample_index = np.random.choice(len(subset_train_data), 10, replace=False)  # Choose only 10 samples
small_train_data = subset_train_data[small_sample_index]
small_train_labels = subset_train_labels[small_sample_index]


quick_test_epochs = 1  # Instead of 5, just to test the code

#Split data into training sizes
data_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#Store accuracies for later plotting
accuracies = []
accuracies = []
precisions = []
recalls = []
f1_scores = []

for size in data_sizes:
    num_samples = int(size * len(testData))
    indices = np.random.choice(len(testData), num_samples, replace=False)
    subset_train_data = devData[indices]
    subset_train_labels = devLabels[indices]

    model = baselineModel(len(vocab), labels, DIM_EMBEDDING, LSTM_HIDDEN, CONSTRAINTS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(quick_test_epochs):
        optimizer.zero_grad()
        loss = model.forwardTrain(subset_train_data, subset_train_labels)
        loss.backward()
        optimizer.step()

    model.eval()
    predicts = []
    with torch.no_grad():
        predicts = model.forwardPred(subset_train_data)

    # Prediction to tensor
    predicts_tensor = torch.tensor(predicts, dtype=torch.long).flatten()
    labels_tensor = torch.flatten(subset_train_labels)

    # Calculating metrics
    accuracy = accuracy_score(labels_tensor, predicts_tensor)
    precision = precision_score(labels_tensor, predicts_tensor, average='macro', zero_division=0)
    recall = recall_score(labels_tensor, predicts_tensor, average='macro', zero_division=0)
    f1 = f1_score(labels_tensor, predicts_tensor, average='macro', zero_division=0)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

print("Accuracies:", accuracies)
print("Precisions:", precisions)
print("Recalls:", recalls)
print("F1-scores:", f1_scores)


# Plotting the metrics
plt.figure(figsize=(12, 8))
plt.plot(data_sizes, accuracies, label='Accuracy')
plt.plot(data_sizes, precisions, label='Precision')
plt.plot(data_sizes, recalls, label='Recall')
plt.plot(data_sizes, f1_scores, label='F1 Score')
plt.xlabel('Fraction of Training Data Used')
plt.ylabel('Metric Values')
plt.title('Performance Metrics vs. Data Size')
plt.legend()
plt.grid(True)
plt.show()

