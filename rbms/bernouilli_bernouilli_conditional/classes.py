from rbms.classes import RBM
from .implement import _get_dynamic_biases, _sample_hiddens_cond, _sample_visibles_cond

class BBCRBM(RBM):
    def __init__(self, W, vbias, hbias, A, B, n_past, device=None, dtype=None):
        self.weight_matrix = W
        self.vbias = vbias
        self.hbias = hbias
        self.A = A
        self.B = B
        self.n_past = n_past
        self.name = "BBCRBM"

    def sample_hiddens(self, chains, beta=1.0, context=None):
        dv, dh = _get_dynamic_biases(self.vbias, self.hbias, self.A, self.B, context)
        chains["hidden"], chains["hidden_mag"] = _sample_hiddens_cond(chains["visible"], self.weight_matrix, dh, beta)
        return chains

    def generate_sequence(self, seed_frames, n_steps_future, gibbs_steps=10):
        # Implementation of your autoregressive prototype 
        # using the internal modular sampling methods.
        generated_seq = seed_frames.clone()
        for _ in range(n_steps_future):
            u = generated_seq[:, -self.n_past:, :].reshape(seed_frames.shape[0], -1)
            # ... (Gibbs sampling steps) ...
            generated_seq = torch.cat((generated_seq, v_new), dim=1)
        return generated_seq

    def named_parameters(self):
        # Includes A and B for HDF5 storage
        params = super().named_parameters()
        params.update({
            "A": self.A.cpu().numpy(),
            "B": self.B.cpu().numpy(),
            "n_past": np.asarray(self.n_past)
        })
        return params