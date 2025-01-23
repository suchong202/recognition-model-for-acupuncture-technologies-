import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, recall_score
import logging
import time
import datetime

def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 50)
        logger.info('-' * 50)


        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_labels = []
            all_preds = []

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_recall = recall_score(all_labels, all_preds, average='macro')

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f}')
            logger.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f}')

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, f"best_model_{model_choose}.pth")

        scheduler.step()

        time.sleep(0.2)

    print(f'Best val Acc: {best_acc:.4f}')
    logger.info(f'Best val Acc: {best_acc:.4f}')
    logger.info(f"best_model: best_model_{model_choose}.pth")
    return model


def test_model(model):
    model.eval()
    running_corrects = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloaders["test"]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds, average='macro')

    print(f'Test Acc: {test_acc:.4f} Recall: {test_recall:.4f}')
    logger.info(f'Test Acc: {test_acc:.4f} Recall: {test_recall:.4f}')
    print("Per-class accuracy:")
    logger.info("Per-class accuracy:")
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    logger.info(report)


if __name__ == "__main__":
    model_choose = "resnet18"  # "shuffle_net_v2"
    assert model_choose in ["resnet18", "shuffle_net_v2"]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"train_{timestamp}_{model_choose}.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()
    data_dir = {
        "train": "./train" ,
        "val": "./val",
        "test": "./test"
    }

    # 指定类别标签
    hagrid_cate_file = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
                        "peace_inverted",
                        "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

    hagrid_cate_dict = {hagrid_cate_file[i]: i for i in range(len(hagrid_cate_file))}
    print(hagrid_cate_dict)
    logger.info(f": {hagrid_cate_dict}")

    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir[x]), data_transforms[x]) for x in
                      ["train", "val", "test"]}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0) for x in
                   ["train", "val", "test"]}
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
    class_names = image_datasets["train"].classes

    assert len(class_names) == len(
        hagrid_cate_file), f"[{len(class_names)}],[{len(hagrid_cate_file)}]"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_choose == "resnet18":
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(hagrid_cate_file))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

    model = train_model(model, criterion, optimizer, scheduler, num_epochs=num_epochs)

    test_model(model)