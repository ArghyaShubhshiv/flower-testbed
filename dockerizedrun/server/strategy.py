"""
FedAvgDelta: a FedAvg variant where clients return WEIGHT DELTAS
(local_weights - weights_they_started_the_round_with) instead of full
weights.

Flower's built-in FedAvg.aggregate_fit() already computes a num-examples
weighted average of whatever arrays clients return in FitRes.parameters.
Since our clients return deltas, that weighted average IS the weighted
average delta. All we add on top is:

    new_global_weights = old_global_weights + avg_delta

which is mathematically identical to plain FedAvg
(weighted average of the clients' resulting local weights), but makes the
"send only the update, not the full model" behaviour explicit and lets the
server track the canonical global weights itself.
"""
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from common import attacks


class FedAvgDelta(fl.server.strategy.FedAvg):
    def __init__(self, initial_parameters: Parameters, local_epochs: int = 5,
                 attack: Optional[attacks.AttackConfig] = None, param_mask=None, **kwargs):
        super().__init__(initial_parameters=initial_parameters, **kwargs)
        self.current_weights = parameters_to_ndarrays(initial_parameters)
        self.local_epochs = local_epochs
        self.attack = attack or attacks.AttackConfig()
        self.param_mask = param_mask

    def configure_fit(self, server_round, parameters, client_manager):
        """Send current global weights + the local_epochs (E) hyperparameter."""
        fit_ins_list = super().configure_fit(server_round, parameters, client_manager)
        for _, fit_ins in fit_ins_list:
            fit_ins.config["local_epochs"] = self.local_epochs
            fit_ins.config["server_round"] = server_round
        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        # Omniscient/colluding attacks (e.g. ALIE): craft Byzantine updates from benign ones.
        if self.attack.name in attacks.SERVER_ATTACKS and self.attack.active(server_round):
            byz = [fr for _, fr in results if fr.metrics.get("byzantine")]
            benign = [parameters_to_ndarrays(fr.parameters) for _, fr in results
                      if not fr.metrics.get("byzantine")]
            if byz and benign:
                mal = attacks.SERVER_ATTACKS[self.attack.name](benign, len(byz), self.param_mask, self.attack)
                for fr in byz:
                    fr.parameters = ndarrays_to_parameters(mal)

        n_byz = sum(1 for _, fr in results if fr.metrics.get("byzantine"))
        print(f"[Round {server_round}] attack={self.attack.name} byzantine_clients={n_byz}/{len(results)}")
        
        # super().aggregate_fit() returns the num-examples-weighted average
        # of the arrays clients sent back -- here, that's the average delta.
        aggregated_delta_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_delta_parameters is None:
            return None, metrics

        avg_delta = parameters_to_ndarrays(aggregated_delta_parameters)
        self.current_weights = [w + d for w, d in zip(self.current_weights, avg_delta)]

        n_clients = len(results)
        n_examples = sum(fit_res.num_examples for _, fit_res in results)
        print(f"[Round {server_round}] aggregated deltas from {n_clients} clients "
              f"({n_examples} total examples)")

        return ndarrays_to_parameters(self.current_weights), metrics
