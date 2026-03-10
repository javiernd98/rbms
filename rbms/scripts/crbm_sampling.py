import argparse
import h5py
import torch

from rbms.io import load_params
from rbms.utils import get_saved_updates
from rbms.map_model import map_model
from rbms.parser import (
    add_args_pytorch, 
    match_args_dtype,  
    add_sampling_args,
    add_args_generation
)
from rbms.dataset.load_h5 import load_HDF5
from rbms.dataset.parser import add_args_dataset
from rbms.crbm_generation import generate_conditional_sequence

def create_parser():
    parser = argparse.ArgumentParser(description="Conditional sampling with sliding window for CRBMs")
    parser.add_argument("-i", "--filename", type=str, required=True, help="Trained CRBM model (.h5)")
    parser.add_argument("-o", "--out_file", type=str, required=True, help="Path to save the generated sequences")
    
    parser = add_args_dataset(parser)     # Nos da -d (obligatorio)
    parser = add_sampling_args(parser)    # Nos da --gibbs_steps
    parser = add_args_generation(parser)  # Nos da --gen_steps, --num_seqs, --seed_origin y --seed_mode
    parser = add_args_pytorch(parser)     # Nos da --device y --dtype
    
    return parser

def main():
    parser = create_parser()
    args = vars(parser.parse_args())
    args = match_args_dtype(args)

    print("Loading model...")
    # Buscamos todos los updates guardados y nos quedamos con el último (-1)
    last_update = get_saved_updates(args["filename"])[-1]
    print(f"Restoring from update: {last_update}")
    
    params = load_params(
        filename=args["filename"], 
        index=last_update,   
        device=args["device"], 
        dtype=args["dtype"], 
        map_model=map_model
    )
    
    
    dataset_path = args["dataset"]
    print(f"Loading full dataset from {dataset_path}...")
    data, _, _, _ = load_HDF5(filename=dataset_path, use_weights=False, device=args["device"])
    data = torch.from_numpy(data).to(device=args["device"], dtype=args["dtype"])
    
    total_samples = data.shape[0]

    # 2. Leemos el train_size original con el que se entrenó (desde el archivo RBM.h5)
    with h5py.File(args["filename"], "r") as f:
        if "dataset_args" in f.keys() and "train_size" in f["dataset_args"].keys():
            train_size = f["dataset_args"]["train_size"][()].item()
        else:
            train_size = args.get("train_size", 0.6) # Default del repo
            
    # Calculamos en qué instante de tiempo se cortó el train y empezó el test
    split_idx = int(train_size * total_samples)

    # 3. Lógica de SEED_ORIGIN: Definimos el "pool" de datos disponibles
    if args["seed_origin"] == "train":
        print(f"Using TRAIN set (Start index: 0, End index: {split_idx})...")
        pool = data[:split_idx]
    elif args["seed_origin"] == "test":
        print(f"Using TEST set (Start index: {split_idx})...")
        pool = data[split_idx:]
    else:
        raise ValueError(f"Unknown seed origin: {args['seed_origin']}")

    # 4. Lógica de SEED_MODE: Extraemos la(s) semilla(s) del pool
    # Usamos .get por si no has actualizado el parser.py todavía, por defecto será 'single'
    seed_mode = args.get("seed_mode", "single")

    if seed_mode == "single":
        print(f"Modo SINGLE: Extrayendo la primera semilla disponible...")
        seed_data = pool[:params.n_past].flatten()
    
    elif seed_mode == "random":
        print(f"Modo RANDOM: Extrayendo {args['num_seqs']} semillas aleatorias...")
        max_start_idx = len(pool) - params.n_past
        
        # Generamos 'num_seqs' índices de inicio aleatorios dentro del pool
        start_indices = torch.randint(0, max_start_idx, (args["num_seqs"],))
        
        semillas = []
        for idx in start_indices:
            semillas.append(pool[idx : idx + params.n_past].flatten())
        # Apilamos todas las semillas en una matriz 2D
        seed_data = torch.stack(semillas)

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