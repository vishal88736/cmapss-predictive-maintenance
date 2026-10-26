"""One model definition for notebook training and website inference."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_rnn(rnn, x, valid):
    """Pack only real timesteps, then restore left-padded positions."""
    lengths = valid.sum(1).long()
    if bool((lengths == 0).any()):
        raise ValueError("Every window needs at least one real observation.")
    # Compact/restore the whole batch at once. Per-window boolean indexing and
    # scalar lengths otherwise force hundreds of CPU/GPU synchronizations.
    positions = (valid.long().cumsum(1) - 1).clamp_min(0)
    compact = torch.zeros_like(x).scatter_add(
        1, positions.unsqueeze(-1).expand_as(x), x.masked_fill(~valid.unsqueeze(-1), 0))
    packed = pack_padded_sequence(compact, lengths.cpu(), batch_first=True, enforce_sorted=False)
    packed_output, _ = rnn(packed)
    unpacked, _ = pad_packed_sequence(packed_output, batch_first=True)
    restored = unpacked.gather(1, positions.unsqueeze(-1).expand(-1, -1, unpacked.shape[-1]))
    return restored.masked_fill(~valid.unsqueeze(-1), 0)


def valid_mean(x, valid):
    return (x * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


def last_valid(x, valid):
    positions = torch.arange(x.shape[1], device=x.device).expand_as(valid)
    index = positions.masked_fill(~valid, -1).max(1).values
    return x[torch.arange(len(x), device=x.device), index]


def has_finite_fault_detector(model):
    """Expose updated detector scores independently of the repair-path choice.

    Updated weights establish availability, not detection accuracy. The original
    zero-weight/constant-score head must not be presented as fault evidence.
    """
    return model.config["train_repair"] and bool(torch.count_nonzero(model.detector.head.weight))


class FaultDetector(nn.Module):
    def __init__(self, config):
        super().__init__()
        c, h = config["cnn_channels"], config["gru_hidden"]
        self.conv3, self.conv5 = nn.Conv1d(45, c, 3, padding=1), nn.Conv1d(45, c, 5, padding=2)
        self.gru = nn.GRU(2 * c, h, batch_first=True)
        self.head = nn.Linear(h, 21)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, -4.)

    def forward(self, sensors, mask, settings, valid):
        x = torch.cat([sensors, mask, settings], -1) * valid.unsqueeze(-1)
        local = torch.cat([F.relu(self.conv3(x.transpose(1, 2))), F.relu(self.conv5(x.transpose(1, 2)))], 1).transpose(1, 2)
        return self.head(masked_rnn(self.gru, local, valid))


class SensorGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        h = config["generator_hidden"]
        self.encoder = nn.LSTM(66, h, num_layers=config["generator_layers"], batch_first=True, bidirectional=True)
        self.decoder = nn.Sequential(nn.Linear(2 * h, 64), nn.ReLU(), nn.Linear(64, 21))
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, sensors, mask, probabilities, settings, valid):
        x = torch.cat([sensors, mask, probabilities, settings], -1)
        # Begin with identity on available sensors and mean imputation on missing
        # ones, then learn corrections instead of reconstructing every value.
        return sensors + self.decoder(masked_rnn(self.encoder, x, valid))


class RULPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c, h, d = config["cnn_channels"], config["gru_hidden"], config["transformer_dim"]
        self.conv3, self.conv5 = nn.Conv1d(66, c, 3, padding=1), nn.Conv1d(66, c, 5, padding=2)
        self.gru = nn.GRU(2 * c, h, batch_first=True)
        self.local_projection = nn.Linear(h, d)
        self.input_projection = nn.Linear(66, d)
        position = torch.arange(config["sequence_length"]).float()[:, None]
        frequency = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        encoding = torch.zeros(config["sequence_length"], d)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        self.register_buffer("position", encoding[None])
        layer = nn.TransformerEncoderLayer(d, config["transformer_heads"], 2 * d,
            config["dropout"], batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, config["transformer_layers"], enable_nested_tensor=False)
        self.gate = nn.Sequential(nn.Linear(2 * d + 64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.head = nn.Sequential(nn.Linear(d + 64, 64), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(64, 1))
        self.local_head, self.global_head = nn.Linear(d, 1), nn.Linear(d, 1)

    def forward(self, repaired, settings, mask, weights, observed, valid, branch="both", equal_fusion=False):
        x = torch.cat([repaired, settings, mask, weights], -1) * valid.unsqueeze(-1)
        local = global_features = x.new_zeros((len(x), self.config["transformer_dim"]))
        if branch != "global":
            n = min(self.config["local_length"], x.shape[1])
            local_x, local_valid = x[:, -n:], valid[:, -n:]
            conv = torch.cat([F.relu(self.conv3(local_x.transpose(1, 2))), F.relu(self.conv5(local_x.transpose(1, 2)))], 1).transpose(1, 2)
            local = self.local_projection(last_valid(masked_rnn(self.gru, conv, local_valid), local_valid))
        if branch != "local":
            embedding = self.input_projection(x) + self.position[:, :x.shape[1]]
            global_features = last_valid(self.transformer(embedding, src_key_padding_mask=~valid), valid)
        quality = torch.cat([valid_mean(mask, valid), valid_mean(weights, valid),
            valid_mean((repaired - observed).abs(), valid), valid.float().mean(1, keepdim=True)], -1)
        if branch in {"local", "global"}:
            gates = x.new_tensor([1., 0.] if branch == "local" else [0., 1.]).expand(len(x), -1)
        elif equal_fusion:
            gates = x.new_full((len(x), 2), 0.5)
        else:
            gates = self.gate(torch.cat([local, global_features, quality], -1)).softmax(-1)
        fused = gates[:, :1] * local + gates[:, 1:] * global_features
        scale = self.config["target_scale"]
        return {"base": F.softplus(self.head(torch.cat([fused, quality], -1))).flatten() * scale,
                "local_prediction": F.softplus(self.local_head(local)).flatten() * scale if branch != "global" else x.new_zeros((len(x),)),
                "global_prediction": F.softplus(self.global_head(global_features)).flatten() * scale if branch != "local" else x.new_zeros((len(x),)),
                "features": fused, "quality": quality, "gates": gates}


class GraceRUL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.register_buffer("repair_enabled", torch.tensor(bool(config["train_repair"])))
        self.register_buffer("correction_bound", torch.tensor(float(config["correction_bound"])))
        self.detector = FaultDetector(config)
        self.generator = SensorGenerator(config)
        self.predictor = RULPredictor(config) if config["predictor_family"] == "gated" else SingleBranchPredictor(config)
        # These summaries have the same meaning across independently fitted OOF
        # models. Their hidden embedding coordinates do not share a common basis.
        self.corrector = nn.Sequential(nn.Linear(67, 32), nn.ReLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.corrector[-1].weight)
        nn.init.zeros_(self.corrector[-1].bias)

    def forward(self, x, valid=None, use_repair=True, use_correction=True, branch=None, equal_fusion=None):
        use_repair = use_repair and self.config["train_repair"] and bool(self.repair_enabled)
        branch = self.config.get("predictor_branch", "both") if branch is None else branch
        equal_fusion = self.config.get("equal_fusion", False) if equal_fusion is None else equal_fusion
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if valid is None else valid.bool()
        settings = torch.nan_to_num(x[..., :3]) * valid.unsqueeze(-1)
        mask = torch.isfinite(x[..., 3:]).float() * valid.unsqueeze(-1)
        observed = torch.nan_to_num(x[..., 3:], nan=0., posinf=0., neginf=0.) * valid.unsqueeze(-1)
        logits = self.detector(observed, mask, settings, valid) if self.config["train_repair"] else torch.full_like(observed, -20.)
        probabilities = logits.sigmoid()
        generated = self.generator(observed, mask, probabilities, settings, valid) if use_repair else observed
        weights = ((1 - mask) + mask * probabilities) * valid.unsqueeze(-1)
        repaired = ((1 - weights) * observed + weights * generated) * valid.unsqueeze(-1) if use_repair else observed
        result = self.predictor(repaired, settings, mask, weights if use_repair else torch.zeros_like(weights), observed, valid, branch, equal_fusion)
        features = torch.cat([result["quality"], result["base"][:, None] / self.config["target_scale"],
            result["local_prediction"][:, None] / self.config["target_scale"], result["global_prediction"][:, None] / self.config["target_scale"]], -1)
        delta = self.correction_bound * self.corrector(features).tanh().flatten() if use_correction else torch.zeros_like(result["base"])
        return {**result, "prediction": (result["base"] + delta).clamp_min(0), "delta": delta,
                "correction_features": features, "logits": logits, "fault_scores": probabilities,
                "generated": generated, "repaired": repaired, "mask": mask, "weights": weights, "observed": observed}


class SingleBranchPredictor(nn.Module):
    """Architectural comparators with 24 measured inputs and a shared protocol.

    These are adaptations, not exact reproductions of the original papers.
    """
    def __init__(self, config):
        super().__init__()
        self.config, self.family = config, config["predictor_family"]
        c, h, d = config["cnn_channels"], config["gru_hidden"], config["transformer_dim"]
        if self.family in {"cnn_gru", "cnn_lstm"}:
            self.conv3, self.conv5 = nn.Conv1d(24, c, 3, padding=1), nn.Conv1d(24, c, 5, padding=2)
            self.gru = (nn.GRU if self.family == "cnn_gru" else nn.LSTM)(2 * c, h, batch_first=True)
            self.projection = nn.Linear(h, d)
        elif self.family == "transformer":
            self.projection = nn.Linear(24, d)
            layer = nn.TransformerEncoderLayer(d, config["transformer_heads"], 2 * d,
                config["dropout"], batch_first=True, activation="gelu")
            self.transformer = nn.TransformerEncoder(layer, config["transformer_layers"], enable_nested_tensor=False)
            position = torch.arange(config["sequence_length"]).float()[:, None]
            frequency = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.) / d))
            encoding = torch.zeros(config["sequence_length"], d)
            encoding[:, 0::2], encoding[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
            self.register_buffer("position", encoding[None])
        else:
            raise ValueError(f"Unknown predictor family: {self.family}")
        self.head = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Dropout(config["dropout"]), nn.Linear(64, 1))

    def forward(self, repaired, settings, mask, weights, observed, valid, branch="both", equal_fusion=False):
        x = torch.cat([repaired, settings], -1) * valid.unsqueeze(-1)
        if self.family == "transformer":
            encoded = self.projection(x) + self.position[:, :x.shape[1]]
            features = last_valid(self.transformer(encoded, src_key_padding_mask=~valid), valid)
        else:
            n = min(self.config["local_length"], x.shape[1]) if self.family == "cnn_gru" else x.shape[1]
            x, local_valid = x[:, -n:], valid[:, -n:]
            conv = torch.cat([F.relu(self.conv3(x.transpose(1, 2))), F.relu(self.conv5(x.transpose(1, 2)))], 1).transpose(1, 2)
            features = self.projection(last_valid(masked_rnn(self.gru, conv, local_valid), local_valid))
        base = F.softplus(self.head(features)).flatten() * self.config["target_scale"]
        quality = torch.cat([valid_mean(mask, valid), valid_mean(weights, valid),
            valid_mean((repaired - observed).abs(), valid), valid.float().mean(1, keepdim=True)], -1)
        gates = base.new_tensor([0., 1.] if self.family == "transformer" else [1., 0.]).expand(len(base), -1)
        return {"base": base, "local_prediction": base, "global_prediction": base,
                "features": features, "quality": quality, "gates": gates}


class TemporalCritic(nn.Module):
    """Training-only conditional Wasserstein critic."""
    def __init__(self, config):
        super().__init__()
        c, h = config["cnn_channels"], config["gru_hidden"]
        self.conv = nn.Sequential(nn.Conv1d(24, c, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv1d(c, 2 * c, 3, padding=1), nn.LeakyReLU(0.2))
        self.lstm = nn.LSTM(2 * c, h, batch_first=True)
        self.head = nn.Linear(h, 1)

    def forward(self, sensors, settings, valid):
        x = torch.cat([sensors, settings], -1) * valid.unsqueeze(-1)
        conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.head(last_valid(masked_rnn(self.lstm, conv, valid), valid)).flatten()
