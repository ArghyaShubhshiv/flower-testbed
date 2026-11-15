import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl

sys.path.append("/app")
from common.model import CIFAR100CNN, trainable_mask  # noqa: E402
from common import attacks
from common.data_utils import get_client_dataloader  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIENT_ID = int(os.environ["CLIENT_ID"])  # required, must be unique per container
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "20"))
ALPHA = float(os.environ.get("DIRICHLET_ALPHA", "0.5"))
LOCAL_EPOCHS_DEFAULT = int(os.environ.get("LOCAL_EPOCHS", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
LR = float(os.environ.get("LR", "0.01"))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "server:8080")
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SEED = int(os.environ.get("SEED", "42"))
MIN_SAMPLES_PER_CLIENT = int(os.environ.get("MIN_SAMPLES_PER_CLIENT", "20"))

torch.manual_seed(SEED + CLIENT_ID)
ATTACK = attacks.AttackConfig.from_env()
IS_BYZ = ATTACK.is_byzantine(CLIENT_ID)
if IS_BYZ:
    print(f"[Client {CLIENT_ID}] BYZANTINE: attack={ATTACK.name} scale={ATTACK.scale} "
          f"start_round={ATTACK.start_round}")

model = CIFAR100CNN().to(DEVICE)
PARAM_MASK = trainable_mask(model)

trainloader = get_client_dataloader(CLIENT_ID, NUM_CLIENTS, ALPHA, BATCH_SIZE, DATA_DIR, SEED,
                                     MIN_SAMPLES_PER_CLIENT)
print(f"[Client {CLIENT_ID}] local shard size = {len(trainloader.dataset)} "
      f"(alpha={ALPHA}, num_clients={NUM_CLIENTS})")


def get_parameters(net):
    return [val.cpu().numpy() for val in net.state_dict().values()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def train(net, loader, epochs, lr,poison = False):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        running_loss, n = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if poison:
                images,labels = attacks.poison_batch(images,labels,ATTACK)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            n += labels.size(0)
        if n > 0:
            print(f"[Client {CLIENT_ID}] local epoch {epoch + 1}/{epochs} loss={running_loss / n:.4f}")


class DeltaClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return get_parameters(model)

    def fit(self, parameters, config):
        # `parameters` = the current global weights the server sent this round.
        global_weights = [np.copy(p) for p in parameters]
        set_parameters(model, parameters)

        epochs = int(config.get("local_epochs", LOCAL_EPOCHS_DEFAULT))
        server_round = int(config.get("server_round",0))
        attack_on = IS_BYZ and ATTACK.active(server_round)
        train(model, trainloader, epochs, LR, poison = attack_on)

        new_local_weights = get_parameters(model)
        # Only the DELTA (update) is sent back to the server, not the raw weights.
        deltas = [new - old for new, old in zip(new_local_weights, global_weights)]
        if attack_on:
            rng = np.random.default_rng(SEED + CLIENT_ID * 1000 + server_round)
            deltas = attacks.poison_update(deltas, PARAM_MASK, ATTACK, rng)

        return deltas, len(trainloader.dataset), {"client_id": CLIENT_ID, "byzantine": int(attack_on)}

    def evaluate(self, parameters, config):
        # This setup evaluates centrally on the server's held-out test set;
        # clients don't run local evaluation.
        return 0.0, len(trainloader.dataset), {}


if __name__ == "__main__":
    fl.client.start_numpy_client(server_address=SERVER_ADDRESS, client=DeltaClient())
