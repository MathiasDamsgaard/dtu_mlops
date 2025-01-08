from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Annotated
import pickle
import typer

app = typer.Typer()
train_app = typer.Typer()
app.add_typer(train_app, name="train")


# Load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# Get the training dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


@train_app.command("svm")
def train_svm(output: Annotated[str, typer.Option("--output", "-o")] = "svm_model.ckpt",
              kernel: str = "linear") -> None:
    """Train and evaluate the model."""
    # Train a Support Vector Machine (SVM) model
    model = SVC(kernel=kernel, random_state=42)
    model.fit(x_train, y_train)

    # Save the model
    with open(output, "wb") as file:
        pickle.dump(model, file)


@train_app.command("knn")
def train_knn(output: Annotated[str, typer.Option("--output", "-o")] = "knn_model.ckpt",
              k: int = 5) -> None:
    """Train and evaluate the model."""
    # Train a K-Nearest Neighbors (KNN) model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    # Save the model
    with open(output, "wb") as file:
        pickle.dump(model, file)


@app.command()
def evaluate(model_checkpoint: str):
    """Evaluate the model."""
    # Load the model
    with open(model_checkpoint, "rb") as file:
        model = pickle.load(file)

    # Make predictions on the test set
    y_pred = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return accuracy, report


if __name__ == "__main__":
    app()
