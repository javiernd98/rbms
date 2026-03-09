import argparse
import h5py
import torch

from rbms.io import load_params
from rbms.map_model import map_model
from rbms.parser import add_args_pytorch, match_args_dtype
from rbms.dataset.load_h5 import load_HDF5

# Importamos el motor matemático que acabamos de crear
from rbms.crbm_generation import generate_conditional_sequence

def create_parser():
    parser = argparse.ArgumentParser("Conditional sampling with sliding window for CRBMs")
    parser.add_argument("-i", "--filename", type=str, required=True, help="Trained CRBM model (.h5)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Dataset to extract the initial context/seed")
    parser.add_argument("-o", "--out_file", type=str, required=True, help="Path to save the generated sequences")
    parser.add_argument("--gen_steps", default=100, type=int, help="Number of future timesteps to generate.")
    parser.add_argument("--gibbs_steps", default=50, type=int, help="Gibbs steps per generated timestep.")
    parser.add_argument("--num_seqs", default=10, type=int, help="Number of parallel sequences to generate.")
    parser = add_args_pytorch(parser)
    return parser

def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    args = match_args_dtype(args)

    print("Loading model and seed data...")
    params = load_params(
        filename=args["filename"], index=None, device=args["device"], dtype=args["dtype"], map_model=map_model
    )
    
    data, _, _, _ = load_HDF5(filename=args["dataset"], use_weights=False, device=args["device"])
    data = torch.from_numpy(data).to(device=args["device"], dtype=args["dtype"])
    
    # Extraemos la semilla real (primeros n_past pasos)
    seed_data = data[:params.n_past].flatten()

    # Llamamos a nuestra función pura separada
    generated_sequence = generate_conditional_sequence(
        params=params,
        seed_data=seed_data,
        gen_steps=args["gen_steps"],
        gibbs_steps=args["gibbs_steps"],
        num_seqs=args["num_seqs"]
    )

    print(f"Saving generated sequences to {args['out_file']}...")
    with h5py.File(args["out_file"], "w") as f:
        f.create_dataset("generated_sequences", data=generated_sequence)

if __name__ == "__main__":
    main()