import torch
import numpy as np
import tqdm

def generate_conditional_sequence(
    params, 
    seed_data: torch.Tensor, 
    gen_steps: int, 
    gibbs_steps: int, 
    num_seqs: int
) -> np.ndarray:
    """
    Generates a sequence using a sliding window context.
    """
    num_visibles = params.num_visibles
    n_past = params.n_past

    if seed_data.dim() == 1:
        # single selection of a seed repited for num_seqs times
        current_context = seed_data.unsqueeze(0).repeat(num_seqs, 1)
    else:
        # random selection of seeds across testset
        current_context = seed_data

    generated_sequence = []

    for _ in tqdm.tqdm(range(gen_steps), desc="Generating CRBM sequence"):
        chains = params.init_chains(num_samples=num_seqs)
        
        chains = params.sample_state(
            chains=chains, 
            n_steps=gibbs_steps, 
            beta=1.0, 
            context=current_context
        )
        
        v_t = chains["visible"]
        generated_sequence.append(v_t.cpu().numpy())
        
        # Deslizamos la ventana
        current_context = torch.cat([current_context[:, num_visibles:], v_t], dim=1)

    # Convertimos a (Num_Seqs, Timesteps, Num_Visibles)
    return np.array(generated_sequence).transpose(1, 0, 2)