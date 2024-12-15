import torch
from torch import optim
from src.models.class_ner.data_module import CustomDatamodule
from src.models.class_ner.class_ner import ClassNer
from src.models.class_ner.utils import create_vocabulary
from src.models.class_ner.embeddings import IdfEmbedder, LabelEmbedder
import random
import numpy as np

from torch import nn

from seqeval.metrics import f1_score

SEED = 4200
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def train(chekpoint_save, train_dataset, test_dataset, num_epochs=100, batch_size=64, max_sentence_len = 16):
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
    
    best_f1score = 0.0
    train_metrics = {"f1": []}
    val_metrics = {"f1": []}
    train_losses = {"ce": []}
    val_losses = {"ce": []}
    print(f"[INFO] Start training, epochs = {num_epochs}")
    for epoch in range(num_epochs):
        ner_model.train()
        f1score = []
        celosses = []
        for word_embeddings, tags_embeddings, padding_slice in train_dataloader:
            word_embeddings = word_embeddings.unsqueeze(dim=1).unsqueeze(dim=2)
            tags_embeddings = tags_embeddings.unsqueeze(dim=1).unsqueeze(dim=2)
            #print(f"{word_embeddings.shape=}")
            #print(f"{tags_embeddings.shape=}")
            
            #X = X.unsqueeze(dim=1)
            ner_model.zero_grad()

            pred = ner_model(word_embeddings)
            total_celoss = 0
            splitting_tags = torch.split(tags_embeddings, 1, dim=3)
            for i, logits in enumerate(pred):
                # print(f"{logits.squeeze().shape=}")
                # print(f"{splitting_tags[i].squeeze().shape=}")
                total_celoss += ce_loss(logits.squeeze(), splitting_tags[i].squeeze())

            total_celoss /= len(pred)
            #mean_logits = [torch.mean(pred[i], dim=3).unsqueeze(dim=3).float() for i in range(len(pred))]
            #mean_logits = torch.cat(mean_logits, dim=3)
            #print(f"{mean_logits.dtype=}")
            #print(f"{mean_logits.shape=}")
            #celoss = ce_loss(mean_logits, tags_embeddings)
            celosses.append(total_celoss.item())

            total_celoss.backward()
            optimizer.step()

            y_pred = [torch.softmax(pred[i], dim=3).argmax(dim=3).unsqueeze(dim=3) for i in range(len(pred))]
            #print(f"{y_pred[0].shape=}")
            y_pred = torch.cat(y_pred, dim=3)
            #print(f"{y_pred.shape=}")

            target = tags_embedder.inverse(tags_embeddings.reshape(1, -1).squeeze()).tolist()
            prediction = tags_embedder.inverse(y_pred.reshape(1, -1).squeeze()).tolist()
            #(f"{target=}")
            #print(f"{prediction=}")
            f1score.append(f1_score([target], [prediction]))

        train_metrics["f1"].append(np.array(f1score).mean())    
        train_losses["ce"].append(np.array(celosses).mean())

        val_loss, val_f1 = eval(ner_model, device, test_dataloader, tags_embedder)
        val_metrics["f1"].append(val_f1)
        val_losses["ce"].append(val_loss)

        if epoch % 2 == 0:
            print(f"EPOCH={epoch}/TRAIN")
            print(f"train_ce_loss={train_losses["ce"][-1]}, val_ce_loss={val_losses['ce'][-1]}, train_f1score={train_metrics['f1'][-1]}, val_f1score={val_metrics["f1"][-1]}")

        scheduler.step()

        if val_f1 > best_f1score:
            best_f1score = val_f1
            torch.save(ner_model.state_dict(), f"{chekpoint_save}")

    print(f"[INFO] Training end")
    return train_metrics, val_metrics, train_losses, val_losses
    
def eval(ner_model, device, test_dataloader, tags_embedder):
    ner_model.eval()

    f1score = []
    celosses = []
    ce_loss = nn.CrossEntropyLoss().to(device)    
    with torch.no_grad():
        for word_embeddings, tags_embeddings, padding_slice in test_dataloader:
            word_embeddings = word_embeddings.unsqueeze(dim=1).unsqueeze(dim=2)
            tags_embeddings = tags_embeddings.unsqueeze(dim=1).unsqueeze(dim=2)

            pred = ner_model(word_embeddings)
            total_celoss = 0
            splitting_tags = torch.split(tags_embeddings, 1, dim=3)
            for i, logits in enumerate(pred):
                # print(f"{logits.squeeze().shape=}")
                # print(f"{splitting_tags[i].squeeze().shape=}")
                total_celoss += ce_loss(logits.squeeze(), splitting_tags[i].squeeze())

            total_celoss /= len(pred)
            celosses.append(total_celoss.item())

            y_pred = [torch.softmax(pred[i], dim=3).argmax(dim=3).unsqueeze(dim=3) for i in range(len(pred))]
            #print(f"{y_pred[0].shape=}")
            y_pred = torch.cat(y_pred, dim=3)
            #print(f"{y_pred.shape=}")

            target = tags_embedder.inverse(tags_embeddings.reshape(1, -1).squeeze()).tolist()
            prediction = tags_embedder.inverse(y_pred.reshape(1, -1).squeeze()).tolist()
            #(f"{target=}")
            #print(f"{prediction=}")
            f1score.append(f1_score([target], [prediction]))

    ner_model.train()
    return np.array(celosses).mean(), np.array(f1score).mean()