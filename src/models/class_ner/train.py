import torch
from torch import optim
from src.models.class_ner.data_module import CustomDatamodule
from src.models.class_ner.class_ner import ClassNer
from src.models.class_ner.utils import create_vocabulary
from src.models.class_ner.embeddings import IdfEmbedder, LabelEmbedder
import random
import numpy as np

from torch import nn

from torcheval.metrics.classification.accuracy import MulticlassAccuracy
from torcheval.metrics.classification.f1_score import MulticlassF1Score

SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(chekpoint_save, train_dataset, test_dataset, num_epochs=100, batch_size=32, max_sentence_len = 16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sentences_train, tags_train = train_dataset["Sentence"], train_dataset["Tags"]
    sentences_test, tags_test = test_dataset["Sentence"], test_dataset["Tags"]
    
    print("[INFO] Create vocabulary")
    vocabulary = create_vocabulary(sentences_train, vocabulary_num=20000)
    word_embedder = IdfEmbedder(vocabulary=vocabulary)
    tags_embedder = LabelEmbedder()

    print("[INFO] Train embedders")
    word_embedder.train(sentences_train.map(" ".join))

    datamodule = CustomDatamodule(sentences_train, tags_train, sentences_test, tags_test, word_embedder, tags_embedder, train_bs=batch_size, test_bs=batch_size)
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    test_dataloader = datamodule.test_dataloader()

    ner_model = ClassNer().to(device)
    print("[INFO] Start ClassNer initialization")
    ner_model.apply(ClassNer.initialize)

    optimizer = optim.AdamW(ner_model.parameters(), lr=1e-3, betas=(0.7, 0.9), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97048695)
    ce_loss = nn.CrossEntropyLoss().to(device)
    
    accuracy = MulticlassAccuracy(average="macro", num_classes=7).to(device)
    f1_score = MulticlassF1Score(average="micro", num_classes=7).to(device)

    best_f1score = 0.0
    train_metrics = {"accuracy": [], "f1score": []}
    val_metrics = {"accuracy": [], "f1score": []}
    train_losses = {"ce": []}
    val_losses = {"ce": []}
    print(f"[INFO] Start training, epochs = {num_epochs}")
    for epoch in range(num_epochs):
        ner_model.train()
        accuracy.reset()
        f1_score.reset()

        celosses = []
        for word_embeddings, tags_embeddings, padding_slice in train_dataloader:
            print(f"{word_embeddings.shape=}")
            print(f"{tags_embeddings.shape=}")
            
            #X = X.unsqueeze(dim=1)
            ner_model.zero_grad()
            pred = ner_model(word_embeddings)
            celoss = ce_loss(pred, tags_embeddings)
            celosses.append(celoss.item())

            celoss.backward()
            optimizer.step()
            
            y_pred = torch.softmax(pred, dim=1).argmax(dim=1)
            accuracy.update(y_pred, tags_embeddings)
            f1_score.update(y_pred, tags_embeddings)
            
        train_metrics["accuracy"].append(accuracy.compute().item())
        train_metrics["f1score"].append(f1_score.compute().item())
        train_losses["ce"].append(np.array(celosses).mean())

        val_loss, val_accuracy, val_f1 = linear_val(ner_model, device, test_dataloader)
        val_metrics["accuracy"].append(val_accuracy)
        val_metrics["f1score"].append(val_f1)
        val_losses["ce"].append(val_loss)

        if epoch % 5 == 0:
            print(f"EPOCH={epoch}/TRAIN")
            print(f"train_ce_loss={train_losses["ce"][-1]}, val_ce_loss={val_losses['ce'][-1]}, train_f1score={train_metrics['f1score'][-1]}, val_f1score={val_metrics["f1score"][-1]}")

        scheduler.step()

        if val_f1 > best_f1score:
            best_f1score = val_f1
            torch.save(ner_model.state_dict(), f"{chekpoint_save}")

    print(f"[INFO] Training end")
    return train_metrics, val_metrics, train_losses, val_losses
    
def linear_val(ner_model, device, test_dataloader):
    ner_model.eval()

    celosses = []
    ce_loss = nn.CrossEntropyLoss(torch.Tensor([4.0, 5.0, 1.0, 1.0, 16.0, 8, 1.5])).to(device)
    f1_score = MulticlassF1Score(average="micro", num_classes=7).to(device)
    accuracy = MulticlassAccuracy(average="macro", num_classes=7).to(device)
    f1_score.reset()
    accuracy.reset()
    with torch.no_grad():
        for word_embeddings, tags_embeddings, padding_slice in test_dataloader:
            X = X.unsqueeze(dim=1)
            pred = ner_model(X).squeeze()

            celoss = ce_loss(pred, y)
            celosses.append(celoss.item())

            y_pred = torch.softmax(pred, dim=1).argmax(dim=1)
            f1_score.update(y_pred, y)
            accuracy.update(y_pred, y)

    ner_model.train()
    return np.array(celosses).mean(), accuracy.compute().item(), f1_score.compute().item()