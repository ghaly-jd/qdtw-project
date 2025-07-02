# src/main.py
from loader import load_all_sequences
from classifier import classify_knn
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    folder = "msr_action_data"
    sequences, labels = load_all_sequences(folder)

    # Simple split: 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.3, random_state=42)

    correct = 0
    for test_seq, true_label in zip(X_test, y_test):
        pred = classify_knn(test_seq, X_train, y_train)
        print(f"True: {true_label}, Predicted: {pred}")
        if pred == true_label:
            correct += 1

    accuracy = correct / len(y_test)
    print(f"\n✅ Classification Accuracy: {accuracy:.2%}")
