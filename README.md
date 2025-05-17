
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
    <img src="Blekinge.jpeg" alt="Blekinge Institute of Technology Logo" width="200"/>
    <img src="Mercatorum.png" alt="Universitas Mercatorum Logo" width="200"/>
</div>

# BLEKFL2 - Federated Learning Heterogeneity Explorer 
Version 1.0.0

This framework is the result of a wonderful collaboration between the Blekinge Institute of Technology, BTH (Karlskrona, Sweden) and the University of the Italian Chambers of Commerce, Universitas Mercatorum (Rome, Italy).

This platform is designed for educational purposes to demonstrate the effects of different types of heterogeneity in federated learning environments. By running controlled experiments, users can visualize and understand how statistical, model, communication, and hardware heterogeneity impact federated learning performance.


## Features
Interactive experiments with various heterogeneity types
Support for multiple standard datasets (MNIST, Fashion MNIST, CIFAR-10)
Implementation of key federated learning algorithms (FedAvg, FedProx)
Real-time visualization of training progress
Detailed performance metrics and analysis


# Documentazione Completa del Simulatore di Federated Learning

## Indice
- [BLEKFL2 - Federated Learning Heterogeneity Explorer](#blekfl2---federated-learning-heterogeneity-explorer)
  - [Features](#features)
- [Documentazione Completa del Simulatore di Federated Learning](#documentazione-completa-del-simulatore-di-federated-learning)
  - [Indice](#indice)
  - [Introduzione](#introduzione)
  - [Architettura del Sistema](#architettura-del-sistema)
  - [File del Progetto](#file-del-progetto)
    - [app.py](#apppy)
    - [dataset\_manager.py](#dataset_managerpy)
    - [federated\_learning.py](#federated_learningpy)
    - [index.html](#indexhtml)
    - [main.js](#mainjs)
  - [Principi del Federated Learning](#principi-del-federated-learning)
  - [Algoritmi Implementati](#algoritmi-implementati)
    - [FedAvg (Federated Averaging)](#fedavg-federated-averaging)
    - [FedProx (Federated Proximal)](#fedprox-federated-proximal)
  - [Dataset Supportati](#dataset-supportati)
    - [MNIST](#mnist)
    - [Fashion MNIST](#fashion-mnist)
    - [CIFAR10](#cifar10)
  - [Modelli Neurali](#modelli-neurali)
    - [MNISTNet](#mnistnet)
    - [CIFAR10Net](#cifar10net)
  - [Distribuzione dei Dati](#distribuzione-dei-dati)
    - [IID (Independent and Identically Distributed)](#iid-independent-and-identically-distributed)
    - [Non-IID (Dirichlet Distribution)](#non-iid-dirichlet-distribution)
  - [Metriche e Valutazione](#metriche-e-valutazione)
    - [Accuratezza Globale](#accuratezza-globale)
    - [Divergenza dei Client](#divergenza-dei-client)
    - [Tempo di Addestramento](#tempo-di-addestramento)
  - [Interfaccia Web](#interfaccia-web)
  - [Requisiti del Sistema](#requisiti-del-sistema)
  - [Possibili Estensioni](#possibili-estensioni)

## Introduzione

Il simulatore di Federated Learning è un'applicazione web che permette di sperimentare e visualizzare i principi del Federated Learning, un approccio al machine learning che consente di addestrare modelli distribuiti su dispositivi o server decentralizzati senza condividere direttamente i dati. Questo paradigma è particolarmente importante in contesti dove la privacy dei dati è cruciale.

L'applicazione permette di:
- Selezionare diversi dataset standard (MNIST, Fashion MNIST, CIFAR10)
- Configurare numerosi parametri della simulazione 
- Visualizzare in tempo reale il progresso dell'addestramento
- Analizzare le prestazioni dei modelli attraverso metriche e grafici

## Architettura del Sistema

Il sistema è basato su un'architettura client-server:

1. **Backend (Python/Flask)**:
   - Gestione dataset e preprocessamento
   - Implementazione degli algoritmi di Federated Learning
   - API RESTful per l'interazione con il frontend
   - Generazione risultati e visualizzazioni

2. **Frontend (HTML/CSS/JavaScript)**:
   - Interfaccia utente interattiva
   - Visualizzazione in tempo reale dello stato della simulazione
   - Presentazione dei risultati e grafici

La comunicazione tra frontend e backend avviene tramite chiamate API REST con scambio di dati in formato JSON.

## File del Progetto

### app.py

Il file `app.py` implementa l'applicazione server utilizzando il framework Flask e coordina tutte le funzionalità principali.

**Componenti principali**:
- Server web Flask
- Endpoint API per gestire le richieste dal frontend
- Gestione delle simulazioni in thread separati per non bloccare l'interfaccia utente

**API implementate**:
1. `/api/datasets` (GET): Restituisce la lista dei dataset disponibili
2. `/api/dataset/load` (POST): Carica un dataset specificato
3. `/api/federated/initialize` (POST): Inizializza e avvia una simulazione di federated learning
4. `/api/federated/status` (GET): Restituisce lo stato corrente della simulazione
5. `/api/federated/results` (GET): Restituisce i risultati della simulazione

**Funzione di simulazione in background**:
```python
def run_simulation_background(dataset_name, distribution_type, num_clients, 
                             num_rounds, client_epochs, algorithm, alpha):
    """Esegue la simulazione in background"""
    global simulation_status
    
    try:
        # Esegui la simulazione
        results = fl.run_federated_learning(
            dataset_name=dataset_name,
            distribution_type=distribution_type,
            num_clients=num_clients,
            num_rounds=num_rounds,
            client_epochs=client_epochs,
            algorithm=algorithm,
            non_iid_alpha=alpha
        )
    except Exception as e:
        simulation_status["error"] = str(e)
    finally:
        simulation_status["running"] = False
```

### dataset_manager.py

Questo file gestisce il caricamento, la preparazione e la distribuzione dei dataset tra i client simulati.

**Funzioni principali**:
- `load_dataset(dataset_name)`: Carica un dataset specifico (MNIST, Fashion MNIST o CIFAR10)
- `create_client_datasets(train_dataset, num_clients, distribution_type, batch_size, alpha)`: Divide il dataset tra i client in base alla strategia di distribuzione

**Classe DatasetManager**:
Implementa metodi per:
- Caricamento dei dataset
- Normalizzazione dei dati 
- Distribuzione dei dati tra i client
- Gestione delle informazioni sui dataset

**Distribuzione dei dati tra i client**:
Per la distribuzione non-IID, viene implementata la distribuzione di Dirichlet per creare distribuzioni di dati sbilanciate:

```python
# Use Dirichlet distribution for non-IID data partitioning
np.random.seed(42)  # For reproducibility
class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
```

### federated_learning.py

Questo file è il cuore del simulatore e implementa gli algoritmi di Federated Learning, le reti neurali e le funzionalità di valutazione.

**Componenti principali**:
- Classe `FederatedLearning` che coordina il processo di apprendimento federato
- Implementazione degli algoritmi FedAvg e FedProx
- Definizione delle architetture di rete neurale per diversi dataset
- Funzioni di training client e di aggregazione dei modelli
- Funzioni di valutazione e monitoraggio delle prestazioni

**Algoritmi di Federated Learning implementati**:
1. FedAvg (Federated Averaging)
2. FedProx (Federated Proximal)

**Modelli di rete neurale**:
1. `MNISTNet`: CNN per dataset MNIST e Fashion MNIST
2. `CIFAR10Net`: CNN per dataset CIFAR10

**Funzioni principali**:
- `run_federated_learning()`: Esegue il ciclo completo di federated learning
- `client_update()`: Esegue l'aggiornamento del modello su un singolo client
- `client_update_fedprox()`: Versione FedProx dell'aggiornamento client
- `federated_averaging()`: Aggrega i modelli dei client
- `evaluate_model()`: Valuta le performance del modello globale
- `calculate_client_divergence()`: Calcola la divergenza tra i modelli client

### index.html

Questo file definisce l'interfaccia utente del simulatore, includendo:
- Layout responsive con Bootstrap
- Schede per selezione dataset, configurazione e risultati
- Controlli interattivi per i parametri di simulazione
- Visualizzazione in tempo reale del progresso
- Visualizzatori per grafici e risultati

**Componenti dell'interfaccia**:
1. **Dataset Selection**: Selezione del dataset da utilizzare
2. **Simulation Configuration**: Configurazione dei parametri della simulazione
3. **Simulation Progress**: Monitoraggio in tempo reale della simulazione
4. **Simulation Results**: Visualizzazione dei risultati finali

### main.js

Questo file gestisce la logica frontend dell'applicazione, implementando:
- Caricamento dinamico dei dataset disponibili
- Gestione degli eventi e interazione utente
- Comunicazione con il backend tramite API REST
- Aggiornamento in tempo reale dello stato della simulazione
- Visualizzazione dei risultati e grafici

**Funzioni principali**:
- `loadDatasets()`: Carica la lista dei dataset disponibili
- `loadDataset()`: Carica un dataset specifico
- `startSimulation()`: Avvia una nuova simulazione
- `pollSimulationStatus()`: Monitora lo stato della simulazione
- `displaySimulationResults()`: Mostra i risultati della simulazione

## Principi del Federated Learning

Il Federated Learning è un approccio al machine learning dove un modello viene addestrato su più dispositivi o server che possiedono dati locali, senza che questi dati vengano condivisi centralmente. I principi fondamentali implementati in questo simulatore sono:

1. **Addestramento locale**: I client addestrano localmente il modello sui propri dati
2. **Aggregazione centrale**: Un server centrale aggrega i parametri dei modelli locali
3. **Iterazione**: Il processo viene ripetuto per più round fino alla convergenza
4. **Privacy dei dati**: I dati originali rimangono sui client e non vengono condivisi

Il processo generale implementato segue questo flusso:
1. Il server inizializza un modello globale
2. Per ogni round:
   - Il server distribuisce il modello globale ai client
   - Ogni client addestra il modello sui propri dati
   - I client inviano gli aggiornamenti (parametri del modello) al server
   - Il server aggrega gli aggiornamenti in un nuovo modello globale
3. Al termine dei round, il modello globale viene valutato

## Algoritmi Implementati

### FedAvg (Federated Averaging)

L'algoritmo FedAvg (McMahan et al., 2017) è l'approccio fondamentale implementato nel simulatore. 

**Processo**:
1. Inizializzazione di un modello globale
2. In ogni round:
   - Distribuzione del modello globale a un sottoinsieme di client
   - Ogni client esegue più epoche di training locale (SGD)
   - Il server aggrega i modelli utilizzando una media ponderata basata sulla dimensione dei dataset locali

**Formula di aggregazione**:
L'aggregazione dei parametri del modello è implementata come:

$$w_{global} = \sum_{k=1}^{K} \frac{n_k}{n} w_k$$

Dove:
- $w_{global}$ sono i parametri del modello globale
- $w_k$ sono i parametri del modello del client $k$
- $n_k$ è la dimensione del dataset locale del client $k$
- $n$ è la dimensione totale dei dati

**Implementazione**:
```python
def federated_averaging(client_models, client_data_sizes):
    # Inizializza il modello globale con la stessa struttura dei modelli client
    global_model = copy.deepcopy(client_models[0])
    total_data = sum(client_data_sizes)
    
    # Inizializza i parametri del modello globale a zero
    for param in global_model.parameters():
        param.data.zero_()
    
    # Aggrega i modelli client
    for i, model in enumerate(client_models):
        weight = client_data_sizes[i] / total_data
        for global_param, client_param in zip(global_model.parameters(), model.parameters()):
            global_param.data += client_param.data * weight
    
    return global_model
```

### FedProx (Federated Proximal)

FedProx (Li et al., 2020) è un'estensione di FedAvg che aggiunge un termine di regolarizzazione per limitare l'eterogeneità dei modelli client.

**Processo**:
- Simile a FedAvg, ma durante l'addestramento locale viene aggiunto un termine di prossimità
- Questo termine penalizza i modelli locali che si allontanano troppo dal modello globale

**Formula di ottimizzazione locale**:
La funzione obiettivo locale diventa:

$$\min_{w} F_k(w) + \frac{\mu}{2} ||w - w^t||^2$$

Dove:
- $F_k(w)$ è la funzione di perdita originale sul dataset locale
- $w^t$ sono i parametri del modello globale corrente
- $\mu$ è il parametro di regolarizzazione che controlla la forza del vincolo

**Implementazione**:
```python
def client_update_fedprox(model, global_model, train_loader, epochs=1, lr=0.01, mu=0.01):
    """
    Aggiorna il modello del client con FedProx che include un termine di prossimità
    per limitare la divergenza dal modello globale
    """
    device = torch.device('cpu')
    model.to(device)
    global_model.to(device)
    model.train()
    
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            
            # Loss standard
            loss = criterion(output, target)
            
            # Termine di prossimità
            proximal_term = 0
            for w, w_t in zip(model.parameters(), global_model.parameters()):
                proximal_term += (w - w_t).norm(2)**2
            
            loss += (mu / 2) * proximal_term
            
            loss.backward()
            optimizer.step()
    
    return model
```

## Dataset Supportati

Il simulatore supporta tre dataset standard per il machine learning:

### MNIST
- **Descrizione**: Immagini in scala di grigi di cifre scritte a mano (0-9)
- **Dimensioni**: 60,000 immagini di training, 10,000 immagini di test
- **Formato immagini**: 28x28 pixel, 1 canale
- **Classi**: 10 (cifre da 0 a 9)
- **Normalizzazione applicata**: (0.1307, 0.3081) - media e deviazione standard del dataset

### Fashion MNIST
- **Descrizione**: Immagini in scala di grigi di capi di abbigliamento
- **Dimensioni**: 60,000 immagini di training, 10,000 immagini di test
- **Formato immagini**: 28x28 pixel, 1 canale
- **Classi**: 10 (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)
- **Normalizzazione applicata**: (0.2860, 0.3530) - media e deviazione standard del dataset

### CIFAR10
- **Descrizione**: Immagini a colori di 10 diverse categorie di oggetti
- **Dimensioni**: 50,000 immagini di training, 10,000 immagini di test
- **Formato immagini**: 32x32 pixel, 3 canali (RGB)
- **Classi**: 10 (aereo, automobile, uccello, gatto, cervo, cane, rana, cavallo, nave, camion)
- **Normalizzazione applicata**: ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)) - medie e deviazioni standard per i 3 canali

## Modelli Neurali

### MNISTNet

Rete neurale convoluzionale utilizzata per i dataset MNIST e Fashion MNIST.

**Architettura**:
```
MNISTNet(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (dropout1): Dropout2d(p=0.25, inplace=False)
  (dropout2): Dropout2d(p=0.5, inplace=False)
  (fc1): Linear(in_features=9216, out_features=128, bias=True)
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)
```

**Dettagli della rete**:
- Input: 28x28x1 (immagini in scala di grigi)
- Primo layer convoluzionale: 32 filtri di dimensione 3x3
- Secondo layer convoluzionale: 64 filtri di dimensione 3x3
- Max pooling con dimensione 2x2
- Dropout del 25% dopo il pooling
- Fully connected layer con 128 neuroni
- Dropout del 50% dopo il primo fully connected layer
- Output layer con 10 neuroni (uno per classe)
- Attivazione finale: log_softmax

### CIFAR10Net

Rete neurale convoluzionale più profonda utilizzata per il dataset CIFAR10.

**Architettura**:
```
CIFAR10Net(
  (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc1): Linear(in_features=2048, out_features=512, bias=True)
  (fc2): Linear(in_features=512, out_features=10, bias=True)
  (dropout): Dropout(p=0.25, inplace=False)
)
```

**Dettagli della rete**:
- Input: 32x32x3 (immagini a colori RGB)
- Primo layer convoluzionale: 32 filtri di dimensione 3x3 con padding
- Secondo layer convoluzionale: 64 filtri di dimensione 3x3 con padding
- Terzo layer convoluzionale: 128 filtri di dimensione 3x3 con padding
- Max pooling con dimensione 2x2 dopo ogni layer convoluzionale
- Fully connected layer con 512 neuroni
- Dropout del 25% dopo il fully connected layer
- Output layer con 10 neuroni (uno per classe)
- Attivazione finale: log_softmax

## Distribuzione dei Dati

### IID (Independent and Identically Distributed)

La distribuzione IID è la più semplice, dove i dati vengono distribuiti uniformemente tra i client senza considerare le classi.

**Implementazione**:
```python
# IID distribution: divide data equally
data_per_client = len(train_dataset) // num_clients
indices = torch.randperm(len(train_dataset))

client_loaders = []
client_data_sizes = []

for i in range(num_clients):
    start_idx = i * data_per_client
    end_idx = (i+1) * data_per_client if i < num_clients-1 else len(train_dataset)
    client_indices = indices[start_idx:end_idx]
    
    client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
    client_loader = torch.utils.data.DataLoader(
        client_dataset, batch_size=batch_size, shuffle=True
    )
    
    client_loaders.append(client_loader)
    client_data_sizes.append(len(client_indices))
```

### Non-IID (Dirichlet Distribution)

La distribuzione non-IID utilizza una distribuzione di Dirichlet per creare partizioni sbilanciate dei dati tra i client, dove ogni client può avere una distribuzione di classi diversa.

**Formula della distribuzione di Dirichlet**:

La distribuzione di Dirichlet con parametro di concentrazione $\alpha$ è una distribuzione di probabilità multivariata parametrizzata come:

$$p(x_1, x_2, ..., x_K) = \frac{1}{B(\alpha)} \prod_{i=1}^{K} x_i^{\alpha_i - 1}$$

Dove:
- $x_i$ sono proporzioni che sommano a 1
- $\alpha_i$ sono parametri di concentrazione
- $B(\alpha)$ è la funzione beta multivariata

Nel contesto del simulatore, $\alpha$ controlla quanto "non-IID" saranno i dati:
- $\alpha$ piccolo (es. 0.1): distribuzione molto sbilanciata (più non-IID)
- $\alpha$ grande (es. 100): distribuzione più uniforme (più simile a IID)

**Implementazione**:
```python
# Use Dirichlet distribution for non-IID data partitioning
np.random.seed(42)  # For reproducibility
class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_clients)

# Get indices for each class
class_indices = [torch.where(labels == cls)[0] for cls in range(num_classes)]

# Assign samples to each client
client_indices = [[] for _ in range(num_clients)]

# For each class, distribute indices according to class_priors
for c, indices in enumerate(class_indices):
    # Shuffle indices for this class
    indices = indices[torch.randperm(len(indices))]
    
    # Calculate how many samples of this class go to each client
    class_sizes = (class_priors[:, c] * len(indices)).astype(int)
    class_sizes = np.minimum(class_sizes, len(indices))
    
    # Ensure all samples are assigned by adding remainder to last client
    remaining = len(indices) - np.sum(class_sizes)
    if remaining > 0:
        class_sizes[-1] += remaining
    
    # Assign samples
    start_idx = 0
    for i, size in enumerate(class_sizes):
        end_idx = min(start_idx + size, len(indices))
        if start_idx < end_idx:
            client_indices[i].extend(indices[start_idx:end_idx].tolist())
            start_idx = end_idx
```

## Metriche e Valutazione

Il simulatore implementa diverse metriche per valutare le prestazioni del federated learning:

### Accuratezza Globale

Misura la percentuale di previsioni corrette del modello globale sul dataset di test.

**Formula**:
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$$

**Implementazione**:
```python
def evaluate_model(model, test_loader):
    """
    Valuta il modello sul dataset di test
    """
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = correct / total
    return accuracy
```

### Divergenza dei Client

Misura quanto i modelli dei client sono divergenti tra loro, indicatore importante per valutare l'eterogeneità.

**Formula**:
La divergenza è calcolata come la media delle distanze euclidee (norma L2) tra i parametri di ogni coppia di modelli client:

$$\text{Divergence} = \frac{1}{n_{pairs}} \sum_{i=1}^{n} \sum_{j=i+1}^{n} ||\theta_i - \theta_j||_2$$

Dove:
- $\theta_i$ sono i parametri del modello del client $i$
- $n_{pairs}$ è il numero di coppie uniche di client
- $||\cdot||_2$ è la norma euclidea

**Implementazione**:
```python
def calculate_client_divergence(client_models):
    """
    Calcola una misura della divergenza tra i modelli client
    """
    n_clients = len(client_models)
    total_divergence = 0.0
    
    # Per ogni coppia di client, calcola la divergenza
    for i in range(n_clients):
        for j in range(i+1, n_clients):
            model_i = client_models[i]
            model_j = client_models[j]
            
            # Calcola la distanza L2 tra i parametri
            divergence = 0.0
            for param_i, param_j in zip(model_i.parameters(), model_j.parameters()):
                divergence += torch.norm(param_i - param_j).item()
            
            total_divergence += divergence
    
    # Normalizza per il numero di coppie
    total_pairs = (n_clients * (n_clients - 1)) / 2
    avg_divergence = total_divergence / total_pairs
    
    return avg_divergence
```

### Tempo di Addestramento

Misura il tempo totale richiesto per completare il processo di federated learning.

## Interfaccia Web

L'interfaccia web del simulatore è organizzata in diverse sezioni:

1. **Dataset Selection**:
   - Selezione del dataset da utilizzare per la simulazione
   - Visualizzazione di informazioni di base sul dataset selezionato

2. **Simulation Configuration**:
   - Scelta del tipo di distribuzione dei dati (IID o Non-IID)
   - Impostazione del parametro alpha per la distribuzione Dirichlet (Non-IID)
   - Selezione dell'algoritmo (FedAvg o FedProx)
   - Configurazione del numero di client, round e epoche

3. **Simulation Progress**:
   - Barra di progresso per il monitoraggio in tempo reale
   - Log dettagliato delle operazioni in corso
   - Visualizzazione del tempo trascorso

4. **Simulation Results**:
   - Metriche principali (accuratezza finale, tempo totale, divergenza)
   - Grafici di accuratezza e divergenza per round
   - Opzione per scaricare il modello addestrato

## Requisiti del Sistema

- **Python**: 3.7 o superiore
- **Librerie Python**:
  - Flask: Framework web
  - PyTorch: Implementazione reti neurali e training
  - NumPy: Calcolo numerico
  - Matplotlib: Generazione grafici
  - TensorFlow: Caricamento dataset (usato insieme a PyTorch)
- **Librerie Frontend**:
  - Bootstrap 5: Framework CSS
  - JavaScript: Interazioni client-side

## Possibili Estensioni

1. **Algoritmi Aggiuntivi**:
   - FedOpt (algoritmi di ottimizzazione adattiva)
   - Clustered FL (federated learning con clustering)
   - Personalized FL (modelli personalizzati per client)

2. **Dataset Personalizzati**:
   - Supporto per il caricamento di dataset esterni
   - Visualizzazione della distribuzione dei dati

3. **Privacy**:
   - Implementazione di tecniche di differential privacy
   - Integrazione con Secure Aggregation

4. **Modelli Avanzati**:
   - Supporto per architetture più complesse (ResNet, Transformer)
   - Integrazione con modelli pre-addestrati

5. **Simulazione Avanzata**:
   - Simulazione di latenza e dropout dei client
   - Implementazione di strategie di selezione dei client
   - Strategie asincrone di aggiornamento