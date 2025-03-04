from utils import *
from transformer_lens import ActivationCache
import tqdm
from warnings import warn

class Buffer:
    """
    This defines a data buffer, to store a stack of acts across both model that can be used to train the autoencoder.

    I swap out the buffer completely when it runs out instead of swapping out some subset
    """

    def __init__(self, cfg, model_A, model_B, all_tokens):
        assert model_A.cfg.d_model == model_B.cfg.d_model
        self.cfg = cfg
        tot_tokens_initial_estimate = cfg["batch_size"] * cfg["buffer_mult"]
        self.buffer_batches = tot_tokens_initial_estimate // ((cfg["seq_len"] - 1) * cfg["model_batch_size"])
        self.buffer_size = self.buffer_batches * (cfg["seq_len"] - 1) * cfg["model_batch_size"]
        print(f"We will have {self.buffer_size} tokens in the buffer")
        self.buffer = torch.zeros(
            (self.buffer_size, 2, model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(cfg["device"]) # hardcoding 2 for model diffing
        self.cfg = cfg
        self.model_A = model_A
        self.model_B = model_B
        self.token_pointer = 0
        self.first = True
        self.normalize = True
        self.all_tokens = all_tokens
        
        estimated_norm_scaling_factor_A = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_A)
        estimated_norm_scaling_factor_B = self.estimate_norm_scaling_factor(cfg["model_batch_size"], model_B)
        
        self.normalisation_factor = torch.tensor(
        [
            estimated_norm_scaling_factor_A,
            estimated_norm_scaling_factor_B,
        ],
        device="cuda:0",
        dtype=torch.float32,
        )
        self.refresh()

    @torch.no_grad()
    def estimate_norm_scaling_factor(self, batch_size, model, n_batches_for_norm_estimate: int = 50):
        # stolen from SAELens https://github.com/jbloomAus/SAELens/blob/6d6eaef343fd72add6e26d4c13307643a62c41bf/sae_lens/training/activations_store.py#L370
        norms_per_batch = []
        for i in range(n_batches_for_norm_estimate):
            tokens = self.all_tokens[i * batch_size : (i + 1) * batch_size]
            _, cache = model.run_with_cache(
                tokens,
                names_filter=self.cfg["hook_point"],
                return_type=None,
            )
            acts = cache[self.cfg["hook_point"]]
            acts = acts[:, 1:, :]
            norms_per_batch.append(acts.norm(dim=-1).mean().item())
        mean_norm = np.mean(norms_per_batch)
        scaling_factor = np.sqrt(model.cfg.d_model) / mean_norm

        return scaling_factor

    @torch.no_grad()
    def refresh(self):
        print("Refreshing buffer")
        self.pointer = 0
        self.buffer = torch.zeros(
            (self.buffer_size, 2, self.model_A.cfg.d_model),
            dtype=torch.bfloat16,
            requires_grad=False,
        ).to(self.cfg["device"])
        torch.cuda.empty_cache()
        with torch.autocast("cuda", torch.bfloat16):
            self.first = False
            #print(f"We're going to advance {self.buffer_batches} batches")
            for i in range(0, self.buffer_batches):

                tokens = self.all_tokens[self.token_pointer:self.token_pointer + self.cfg["model_batch_size"]]

                assert tokens.shape == (self.cfg["model_batch_size"], self.cfg["seq_len"])

                _, cache_A = self.model_A.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_A: ActivationCache

                _, cache_B = self.model_B.run_with_cache(
                    tokens, names_filter=self.cfg["hook_point"]
                )
                cache_B: ActivationCache

                acts = torch.stack([cache_A[self.cfg["hook_point"]], cache_B[self.cfg["hook_point"]]], dim=0)
                acts = acts[:, :, 1:, :] # Drop BOS
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # [2, batch, seq_len, d_model]
                assert acts.shape == (2, tokens.shape[0], tokens.shape[1]-1, self.model_A.cfg.d_model) # [2, batch, seq_len, d_model]
                acts = einops.rearrange(
                    acts,
                    "n_layers batch seq_len d_model -> (batch seq_len) n_layers d_model",
                )

                self.buffer[self.pointer : self.pointer + acts.shape[0]] = acts
                self.pointer += acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        # Check that the buffer is filled
        assert self.buffer.shape[0] == self.pointer, f"Buffer size {self.buffer.shape} does not match pointer {self.pointer}"
        assert torch.sum(self.buffer[-1].abs()) > 1, f"Last batch in buffer is not refreshed \n\n {self.buffer[-1]}"

        self.pointer = 0
        self.buffer = self.buffer[
            torch.randperm(self.buffer.shape[0]).to(self.cfg["device"])
        ]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer : self.pointer + self.cfg["batch_size"]].float()
        # out: [batch_size, n_layers, d_model]
        self.pointer += self.cfg["batch_size"]
        if self.pointer + self.cfg["batch_size"] > self.buffer.shape[0]:
            self.refresh()
        if self.normalize:
            out = out * self.normalisation_factor[None, :, None]
        return out
