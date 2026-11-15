"""
Flower NumPyClient that returns weight DELTAS, runnable as a thread.

Many DeltaClient instances live in one process, so nothing here uses module-level
state: each instance owns its model, its data shard (loaded from its own
client_data/client_<i>/ folder) and its own RNG for batching/augmentation.
"""
import contextlib
import traceback

import numpy as np
import torch
import flwr as fl

from common import attacks
from common.data_utils import get_client_dataloader
from common.model import CIFAR100CNN, get_parameters, set_parameters, trainable_mask


class DeltaClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, client_dir: str, attack: attacks.AttackConfig, *,
                 local_epochs: int = 5, batch_size: int = 32, lr: float = 0.01, seed: int = 42,
                 device=None, torch_threads: int = 1, train_slots=None, verbose: bool = False):
        self.client_id = client_id
        self.attack = attack
        self.is_byz = attack.is_byzantine(client_id)
        self.local_epochs = local_epochs
        self.lr = lr
        self.seed = seed
        self.device = device or torch.device("cpu")
        self.torch_threads = torch_threads
        # Optional semaphore bounding how many clients train at the same time
        # (avoids CPU oversubscription when num_clients >> cores).
        self.train_slots = train_slots
        self.verbose = verbose

        self.model = CIFAR100CNN().to(self.device)
        self.param_mask = trainable_mask(self.model)
        self.trainloader = get_client_dataloader(client_dir, batch_size, seed + client_id, self.device)
        self.num_examples = self.trainloader.num_samples

        self._log(f"shard size = {self.num_examples} (from {client_dir})"
                  + (f" | BYZANTINE: attack={attack.name} scale={attack.scale} "
                     f"start_round={attack.start_round}" if self.is_byz else ""))

    def _log(self, msg: str):
        print(f"[Client {self.client_id}] {msg}", flush=True)

    # ------------------------------------------------------------------ local training
    def _train(self, epochs: int, poison: bool) -> float:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.model.train()
        epoch_loss = float("nan")
        for epoch in range(epochs):
            running_loss, n = 0.0, 0
            for images, labels in self.trainloader:
                if poison:
                    images, labels = attacks.poison_batch(images, labels, self.attack)
                optimizer.zero_grad()
                loss = criterion(self.model(images), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * labels.size(0)
                n += labels.size(0)
            if n > 0:
                epoch_loss = running_loss / n
                if self.verbose:
                    self._log(f"local epoch {epoch + 1}/{epochs} loss={epoch_loss:.4f}")
        return epoch_loss

    # ------------------------------------------------------------------ Flower API
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # `parameters` = the current global weights the server sent this round.
        global_weights = [np.copy(p) for p in parameters]
        epochs = int(config.get("local_epochs", self.local_epochs))
        server_round = int(config.get("server_round", 0))
        attack_on = self.is_byz and self.attack.active(server_round)

        slot = self.train_slots if self.train_slots is not None else contextlib.nullcontext()
        with slot:
            torch.set_num_threads(self.torch_threads)  # intra-op threads are per calling thread
            set_parameters(self.model, parameters)
            loss = self._train(epochs, poison=attack_on)

        new_local_weights = get_parameters(self.model)
        # Only the DELTA (update) is sent back to the server, not the raw weights.
        deltas = [new - old for new, old in zip(new_local_weights, global_weights)]
        if attack_on:
            rng = np.random.default_rng(self.seed + self.client_id * 1000 + server_round)
            deltas = attacks.poison_update(deltas, self.param_mask, self.attack, rng)

        self._log(f"round {server_round}: {epochs} epochs, final loss={loss:.4f}"
                  + (" [BYZANTINE]" if attack_on else ""))
        return deltas, self.num_examples, {"client_id": self.client_id, "byzantine": int(attack_on)}

    def evaluate(self, parameters, config):
        # Evaluation is centralized on the server's held-out test set;
        # clients don't run local evaluation.
        return 0.0, self.num_examples, {}


def run_client(client: DeltaClient, server_address: str, errors: list):
    """Thread target: connect to the (in-process) Flower server and serve fit() calls until it disconnects us."""
    try:
        fl.client.start_numpy_client(server_address=server_address, client=client)
    except BaseException as exc:  # noqa: BLE001 - surface any crash to the launcher
        errors.append((f"client-{client.client_id}", exc, traceback.format_exc()))
