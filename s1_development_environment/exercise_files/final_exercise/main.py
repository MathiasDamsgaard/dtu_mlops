import torch
from torch.utils.data import DataLoader
import typer
import matplotlib.pyplot as plt
from data import corrupt_mnist
from model import CorruptClassifier

app = typer.Typer()


@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # TODO: Implement training loop here
    model = CorruptClassifier()
    
    train_dataset, _ = corrupt_mnist()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (image, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            prediction = model(image)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())
            
            accuracy = (prediction.argmax(1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch+1}, Iter {i}, Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "model.pth")
    print("Training complete")

    # Training plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")


@app.command()
def evaluate(model_checkpoint: str, batch_size: int = 32) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = CorruptClassifier()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_dataset = corrupt_mnist()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.eval()
    correct, total = 0, 0
    for image, target in test_dataloader:
        prediction = model(image)
        correct += (prediction.argmax(1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total * 100:.2f}%")

if __name__ == "__main__":
    app()
