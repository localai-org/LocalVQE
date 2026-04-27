import math

import torch
import torch.nn as nn


class AlignBlock(nn.Module):
    """Cross-attention soft delay alignment.

    Estimates a delay distribution D(t) over [0, dmax) for each frame,
    then applies it to align the reference signal to the microphone signal.

    Fixed bugs from Xiaobin-Rong implementation:
    - Line 57 used K.shape[1] (hidden_channels) instead of x_ref.shape[1]
      (in_channels) for the weighted sum reshape.
    """

    def __init__(self, in_channels, hidden_channels, dmax=32, temperature=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.dmax = dmax
        # Registered as a buffer so the trained value persists in state_dict.
        # Mutate via .fill_(), never re-assign.
        self.register_buffer("temperature", torch.tensor(float(temperature)))

        # Pointwise projections for Q and K
        self.pconv_mic = nn.Conv2d(in_channels, hidden_channels, 1)
        self.pconv_ref = nn.Conv2d(in_channels, hidden_channels, 1)

        # Causal unfold: pad dmax-1 frames on top (past), zero on bottom
        self.unfold_k = nn.Sequential(
            nn.ZeroPad2d([0, 0, dmax - 1, 0]),
            nn.Unfold((dmax, 1)),
        )

        # Smoothing conv over the similarity scores
        self.conv = nn.Sequential(
            nn.ZeroPad2d([1, 1, 4, 0]),  # causal
            nn.Conv2d(hidden_channels, 1, (5, 3)),
        )

    def fold_temperature(self, temperature=None):
        """Bake the AlignBlock softmax temperature into the smoothing-conv
        weights.

        Call after `load_state_dict`. With no argument, reads from
        `self.temperature` (a buffer carried by the checkpoint, holding
        the trained value). The GGML graph has no temperature parameter,
        so PyTorch inference must call this to match its behavior;
        without it the model runs at the default 1.0 and loses several
        dB of FE-ST ERLE on real recordings.
        """
        t = float(temperature if temperature is not None else self.temperature)
        if t == 1.0:
            return
        with torch.no_grad():
            self.conv[1].weight.div_(t)
            self.conv[1].bias.div_(t)
            self.temperature.fill_(1.0)

    def forward(self, x_mic, x_ref, return_delay=False):
        """
        x_mic: (B, C, T, F) — microphone encoder features
        x_ref: (B, C, T, F) — far-end encoder features

        Returns:
            aligned: (B, C, T, F) — aligned far-end features
            delay_dist: (B, T, dmax) — delay distribution (if return_delay=True)
        """
        B, C, T, F = x_ref.shape

        # Compute Q and K
        Q = self.pconv_mic(x_mic)  # (B, H, T, F)
        K = self.pconv_ref(x_ref)  # (B, H, T, F)

        # Unfold K along time: creates dmax delayed copies
        Ku = self.unfold_k(K)  # (B, H*dmax, T*F)
        Ku = Ku.view(B, self.hidden_channels, self.dmax, T, F)
        Ku = Ku.permute(0, 1, 3, 2, 4).contiguous()  # (B, H, T, dmax, F)

        # Cross-attention similarity: sum over frequency, scaled by sqrt(F)
        V = torch.sum(Q.unsqueeze(-2) * Ku, dim=-1)  # (B, H, T, dmax)
        V = V / math.sqrt(F)

        # Smooth and reduce to single-head attention
        V = self.conv(V)  # (B, 1, T, dmax)

        # Softmax over delay dimension (temperature < 1 sharpens the distribution)
        A = torch.softmax(V / self.temperature, dim=-1)  # (B, 1, T, dmax)

        # Unfold x_ref (full channels) for the weighted sum
        # BUG FIX: use in_channels (C), not hidden_channels
        unfold_ref = nn.functional.pad(x_ref, [0, 0, self.dmax - 1, 0])
        # Manual unfold along time dimension
        # unfold_ref: (B, C, T+dmax-1, F)
        ref_unfolded = unfold_ref.unfold(2, self.dmax, 1)  # (B, C, T, F, dmax)
        ref_unfolded = ref_unfolded.permute(0, 1, 2, 4, 3).contiguous()  # (B, C, T, dmax, F)

        # Weighted sum using attention weights
        A_expanded = A[:, :, :, :, None]  # (B, 1, T, dmax, 1)
        aligned = torch.sum(ref_unfolded * A_expanded, dim=-2)  # (B, C, T, F)

        if return_delay:
            return aligned, A.squeeze(1)  # (B, T, dmax)
        return aligned
