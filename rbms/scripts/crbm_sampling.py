import argparse
import h5py
import torch

from rbms.io import load_params
from rbms.map_model import map_model
# ¡Importamos nuestros bloques modulares del parser central!
from rbms.parser import (
    add_args_pytorch, 
    match_args_dtype, 
    add_args_dataset, 
    add_sampling_args,
    add_args_generation  # <-- Nuestro nuevo bloque
)
from rbms.dataset.load_h5 import load_HDF5
from rbms.sampling.crbm_generation import generate_conditional_sequence

def create_parser():
    parser = argparse.ArgumentParser(description="Conditional sampling with sliding window for CRBMs")
    
    # Argumentos exclusivos de este script (entrada/salida)
    parser.add_argument("-i", "--filename", type=str, required=True, help="Trained CRBM model (.h5)")
    parser.add_argument("-o", "--out_file", type=str, required=True, help="Path to save the generated sequences")
    
    # Añadimos los bloques modulares del repo
    parser = add_args_dataset(parser)     # Nos da -d y --test_dataset
    parser = add_sampling_args(parser)    # Nos da --gibbs_steps y --beta
    parser = add_args_generation(parser)  # Nos da --gen_steps, --num_seqs y --seed_origin
    parser = add_args_pytorch(parser)     # Nos da --device y --dtype
    
    return parser

def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    args = match_args_dtype(args)

    print("Loading model...")
    params = load_params(
        filename=args["filename"], index=None, device=args["device"], dtype=args["dtype"], map_model=map_model
    )
    
    # Seed origin
    if args["seed_origin"] == "train":
        dataset_path = args["dataset"]
    else: 
        dataset_path = args["test_dataset"]
        if dataset_path is None:
            raise ValueError("You selected --seed_origin 'test', but no --test_dataset was provided!")

    print(f"Loading seed data from {args['seed_origin']} dataset: {dataset_path}...")
    data, _, _, _ = load_HDF5(filename=dataset_path, use_weights=False, device=args["device"])
    data = torch.from_numpy(data).to(device=args["device"], dtype=args["dtype"])
    
    # Extraemos la semilla real (primeros n_past pasos)
    seed_data = data[:params.n_past].flatten()

    # Como el parser de sampling puede traer gibbs_steps=None por defecto, lo protegemos:
    gibbs_steps = args.get("gibbs_steps") if args.get("gibbs_steps") is not None else 50

    print("Generating sequences...")
    generated_sequence = generate_conditional_sequence(
        params=params,
        seed_data=seed_data,
        gen_steps=args["gen_steps"],
        gibbs_steps=gibbs_steps,
        num_seqs=args["num_seqs"]
    )

    print(f"Saving generated sequences to {args['out_file']}...")
    with h5py.File(args["out_file"], "w") as f:
        f.create_dataset("generated_sequences", data=generated_sequence)
        
    print("Done!")

if __name__ == "__main__":
    main()