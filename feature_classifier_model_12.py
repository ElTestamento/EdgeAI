import os
import datetime
import pandas as pd
import numpy as np
import torchaudio
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torchaudio.pipelines import WAV2VEC2_BASE
from IPython.display import clear_output
import torch.nn.functional as F
import warnings
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# GPU-Memory komplett leeren:
torch.cuda.empty_cache()
gc.collect()  # Garbage-Collector
# Warnings ignore: clean output
warnings.filterwarnings('ignore')
# vortrainiertes Modell(WAV2Vec) aus torchaudio laden---------------------
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model()
# Use CUDA wenn available-----------------------------------------------
print('Modell wird auf GPU verschoben')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
if device == 'cuda':
    print('Modell auf GPU verschoben')
print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'No CUDA'}")

# Paths für Ordnerparsing definieren-------------------------------------------------------------------------
data_train_set_path = r'C:\GitHub\EdgeAI\Data\TrainSet'
data_test_set_path = r'C:\GitHub\EdgeAI\Data\TestSet'
df = pd.DataFrame(columns=['file', 'label'])

BATCH_SIZE_EXTRACT = 128
BATCH_SIZE_CLASSIFIER = 128

feature_lst = ["marvin", "go", "on", "off", "up", "yes", "no", "down", "left", "right", "stop", "background_noise"]

# Custom Dataclass für Dataloader notwendig----------------------------------------------
class Custom_Dataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df  # Dataframe nutzen, welches aus der Parsefunktion entstand----

    def __len__(self):  # FESTE PyTorch Magic Method die so heißen muss, damit der Datalaoder sie automatisch nutzt
        return len(self.df)

    def __getitem__(self,
                    idx):  # FESTE PyTorch Magic Method die so heißen muss, damit der Datalaoder sie automatisch nutzt
        if torch.is_tensor(idx):
            idx = idx.tolist()  # wandelt idx in integer um falls zufällig als tensor übergebe

        self.label = self.df.iloc[idx, 1]  # extrahiert label aus DF
        self.path = self.df.iloc[idx, 2]  # extrahiert path aus DF

        self.waveform, self.sample_rate = torchaudio.load(
            self.path)  # wandels WAV in tensor um und gibt sample_rate zurück

        return self.waveform, self.label, self.sample_rate

class custom_net(nn.Module):#Custome Netz zur Klassifizierung
    def __init__(self, num_classes):
        self.count_label = num_classes
        super().__init__()
        self.linear_relu = nn.Sequential(  # Sequential gibt die Werte automatisch durchs Netz

            nn.Linear(768, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, self.count_label),
        )
    def forward(self, x):
        logits = self.linear_relu(x)
        return logits


def collate_fn(batch):  # gleicht waveformlängen(also tensoren die die Waveform präsentieren) innerhalb einer Batch an
    waveform, labels, sample_rates = zip(*batch)
    max_len = min(max([w.shape[-1] for w in waveform]), 32000)
    padded = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveform]

    batched_wf = torch.stack(padded)
    return batched_wf, labels, sample_rates

# Ordner-Struktur parsen und Pandas-DF aus Metadaten der Datenstruktur erstellen-
# file/label/path
def parse_dir(path, df, train_set_type):
    path = path
    new_row = []
    tensor_lst = []
    df = df
    test_size = 0
    train_set_type = train_set_type

    for root, dirs, files in os.walk(path):
        folder = os.path.basename(root)
        if folder in feature_lst:
            for file in files:
                if file.endswith('.wav'):
                    path = os.path.join(root, file)
                    if folder in feature_lst:
                        new_row.append({'file': file, 'label': folder, 'path': path})
    if new_row:
        new_df_row = pd.DataFrame(new_row)

        df = pd.concat([df, new_df_row], ignore_index=True)  # Originalgröße! Für späteres echtes Training

    if train_set_type == False:
        test_size = len(df)
    else:
        test_size = min(len(df), 100)

    df_small_test = df.sample(n=test_size,
                              random_state=42)  # Test-Dataset für Extractionstest und Optimierung des Classifiernetz
    print(df['label'].unique())
    return df# gibt das komplette DF zurück(nicht mehr df_small_test)


# Checkpoint Speichern - am Ende jeder Epoch --->Colabtraining sonst unmöglich
def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, filename):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, filename)


def train_loop(clf, df, resume_from=None):
    EPOCHS = 8 # Ab hier(8) war keine Modellverbesserung mehr
    train_losses = []
    val_losses = []
    start_epoch = 0
    # Setup wie gewohnt
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Custom_Dataset(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_CLASSIFIER, shuffle=True,
                                                collate_fn=collate_fn)
    val_dataset = Custom_Dataset(val_df)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_CLASSIFIER, shuffle=False,
                                            collate_fn=collate_fn)
    label_encoder = LabelEncoder()
    all_labels = df['label'].unique()
    label_encoder.fit(all_labels)

    # Checkpoint laden falls vorhanden
    if resume_from and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from)
        clf.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', [])
        print(f"Resuming from epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, EPOCHS):
        start_time = datetime.datetime.now()
        print(f'Epoch {epoch + 1}/{EPOCHS} - Started at {start_time.strftime("%H:%M:%S")}')

        # Training Phase
        running_loss = 0
        clf.train()
        for i, (audio_batch, label_batch, _) in enumerate(train_loader):
            if i % 20 == 0:
                print(f"Training Batch {i + 1}/{len(train_loader)}")

            audio_batch = audio_batch.to(device)
            if audio_batch.shape[1] == 1:
                audio_batch = audio_batch.squeeze(1)

            batch_features, _ = model.extract_features(audio_batch)
            features = batch_features[-1].mean(dim=1)

            encoded_labels = label_encoder.transform(label_batch)
            encoded_labels = torch.tensor(encoded_labels).to(device)

            optimizer.zero_grad()
            outputs = clf(features)
            loss = criterion(outputs, encoded_labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if i % 50 == 0:
                torch.cuda.empty_cache()

        # Validation Phase
        running_vloss = 0
        clf.eval()
        with torch.no_grad():
            for i, (v_audio, v_label_batch, _) in enumerate(val_loader):
                if i % 10 == 0:
                    print(f"Validation Batch {i+1}/{len(val_loader)}")

                v_audio = v_audio.to(device)
                if v_audio.shape[1] == 1:
                    v_audio = v_audio.squeeze(1)

                v_features, _ = model.extract_features(v_audio)
                v_features = v_features[-1].mean(dim=1)

                v_encoded = label_encoder.transform(v_label_batch)
                v_encoded = torch.tensor(v_encoded).to(device)

                v_outputs = clf(v_features)
                v_loss = criterion(v_outputs, v_encoded)
                running_vloss += v_loss.item()

        # Loss berechnen und speichern
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = running_vloss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        end_time = datetime.datetime.now()
        epoch_duration = end_time - start_time
        print(
            f'Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f} - Duration: {epoch_duration}')

        # Checkpoint speichern
        save_checkpoint(epoch, clf, optimizer, train_losses, val_losses,
                        'checkpoint_12.pt')

    print('Training beendet')

    # Plot
    plt.plot(train_losses, label='Train Loss', color='red')
    plt.plot(val_losses, label='Validation Loss', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves_12.png', dpi=300, bbox_inches='tight')
    plt.show(block=False)
    plt.pause(0.1)

    return label_encoder

# Main--------------------------------------------------------------------
if __name__ == '__main__':

    print('Pipeline gestartet.....')
    df = parse_dir(data_train_set_path, df,
                   train_set_type=True)  # Dataframe bauen: Filename, Label, Path/anzahl der Label zählen für flexibles Netz
    num_classes = df['label'].nunique()
    cust_DS = Custom_Dataset(df)  # CustomDataset für Dataloader instanzieren

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print('Pipeline beendet')

    # Klassifizierungsnetz (selbstgeschrieben)----------------------------------------------
    print('Modell wird erstellt.............')
    classifier = custom_net(num_classes).to(device)  # Modellinstanz erstellen und verschieben wenn GPU verfügbar
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training starten:
    print('Modell wird trainiert.............')
    label_encoder = train_loop(classifier, df, resume_from='checkpoint_12.pt')
    # Modell speichern:
    torch.save({
        'model_state_dict': classifier.state_dict(),
        'label_encoder': label_encoder,
        'num_classes': num_classes
    }, 'classifier_model_12.pt')

    # Modell testen-------------------------------------
    # Test-Dataframe laden:
    t_df = pd.DataFrame(columns=['file', 'label'])
    test_df = parse_dir(data_test_set_path, t_df, train_set_type=False)

    # Nur Labels behalten die im Training vorkamen
    train_labels = set(df['label'].unique())
    test_df = test_df[test_df['label'].isin(train_labels)]
    print(f'TestSet after filtering: {len(test_df)} samples with {test_df["label"].nunique()} classes')

    test_dataset = Custom_Dataset(test_df)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    classifier.eval()
    all_preds = []
    all_true_labels = []

    with torch.no_grad():
        for i, (audio_batch, label_batch, _) in enumerate(test_loader):
            if i % 20 == 0:  # Alle 20 Batches
                print(f"TestSet Batch {i + 1}/{len(test_loader)}")

            audio_batch = audio_batch.to(device)
            if audio_batch.shape[1] == 1:
                audio_batch = audio_batch.squeeze(1)

            test_features, _ = model.extract_features(audio_batch)
            test_features = test_features[-1].mean(dim=1)

            preds = classifier(test_features)
            pred_class = preds.argmax(dim=1).cpu().numpy()
            all_preds.extend(pred_class)
            all_true_labels.extend(label_batch)

    all_preds = np.array(all_preds)
    true_encoded = label_encoder.transform(all_true_labels)
    accuracy = (all_preds == true_encoded).mean()
    print(f'Test Accuracy: {accuracy:.2%}')