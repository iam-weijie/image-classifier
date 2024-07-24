import argparse
import json
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict


def load_checkpoint(checkpoint_path):
    # Load checkpoint from the file
    checkpoint = torch.load(checkpoint_path)

    # Rebuild the model architecture
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        OrderedDict(
            [
                (
                    "fc1",
                    nn.Linear(checkpoint["input_size"], checkpoint["hidden_layers"]),
                ),
                ("relu", nn.ReLU()),
                ("dropout", nn.Dropout(p=0.5)),
                (
                    "fc2",
                    nn.Linear(checkpoint["hidden_layers"], checkpoint["output_size"]),
                ),
                ("output", nn.LogSoftmax(dim=1)),
            ]
        )
    )

    try:
        # Load the model state dictionary
        model.load_state_dict(checkpoint["state_dict"])

        # Load additional information if needed
        model.class_to_idx = checkpoint["class_to_idx"]
    except:
        raise ValueError("Something is wrong with the checkpoint or the architecture")

    return model


def process_image(image):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return preprocess(image).unsqueeze(0)


def predict(image_path, model, topk=5, gpu=False):
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = Image.open(image_path)
    image = process_image(image).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs, indices = torch.exp(outputs).topk(topk)

        probs = probs.cpu().numpy().flatten()
        indices = indices.cpu().numpy().flatten()

        if hasattr(model, "class_to_idx"):
            idx_to_classes = {v: k for k, v in model.class_to_idx.items()}
            indices = [idx_to_classes[idx] for idx in indices]

        return probs, indices


def load_category_names(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image")
    parser.add_argument("image_path", help="Path to the image")
    parser.add_argument("checkpoint", help="Path to the checkpoint")
    parser.add_argument(
        "--top_k", type=int, default=3, help="Return top K most likely classes"
    )
    parser.add_argument(
        "--category_names", help="JSON file mapping category indices to names"
    )
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    probs, indices = predict(args.image_path, model, args.top_k, args.gpu)

    if args.category_names:
        category_names = load_category_names(args.category_names)
        output_names = [category_names[str(idx)] for idx in indices]
    else:
        output_names = indices

    for i in range(args.top_k):
        print(
            f"Class: {output_names[i] if args.category_names else indices[i]}, Probability: {probs[i]:.4f}"
        )
