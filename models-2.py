import os

# Create a directory to save confusion matrices if it doesn't exist
os.makedirs("confusion_matrices", exist_ok=True)

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} Precision: {precision:.4f}")
    print(f"{name} Recall: {recall:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"{name} ROC AUC Score: {roc_auc:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)
    
    # Store overall results
    results.append([name, accuracy, precision, recall, f1, roc_auc])
    
    # Store per-class metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    for class_label, metrics in report.items():
        if class_label != "accuracy":  # Ignore 'accuracy' key since it's a float
            detailed_results.append({
                "Model": name,
                "Class": class_label,
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1 Score": metrics.get("f1-score", 0),
                "Support": metrics.get("support", 0)
            })
    
    # Compute and store confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
    cm_df.insert(0, "Model", name)
    confusion_matrices.append(cm_df)

    # Save confusion matrix as an image using Matplotlib
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.colorbar()
    
    tick_marks = np.arange(len(label_encoder.classes_))
    plt.xticks(tick_marks, label_encoder.classes_, rotation=45)
    plt.yticks(tick_marks, label_encoder.classes_)

    # Label the axes
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # Add values inside the matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Save the image
    plt.tight_layout()
    cm_filename = f"confusion_matrices/confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved as {cm_filename}")

# Save overall metrics to CSV
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC Score"])
results_df.to_csv("model_metrics.csv", index=False)
print("Metrics saved to model_metrics.csv")

# Save per-class metrics to CSV
per_class_df = pd.DataFrame(detailed_results)
per_class_df.to_csv("d_
