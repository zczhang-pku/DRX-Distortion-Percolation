"""Fine-tune CHGNet from a dataset JSON file."""
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="CHGNet dataset JSON")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print-freq", type=int, default=1280)
    parser.add_argument("--output", type=Path, default=Path("chgnet_finetuned"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        import torch
        from chgnet.data.dataset import StructureData, get_train_val_test_loader
        from chgnet.model import CHGNet
        from chgnet.trainer import Trainer
        from pymatgen.core import Structure
        from chgnet.utils import read_json
    except ImportError as exc:
        raise SystemExit("train_chgnet requires chgnet, pymatgen, and torch") from exc

    data = read_json(str(args.dataset))
    structures = [Structure.from_dict(item) for item in data["structure"]]
    dataset = StructureData(
        structures=structures,
        energies=data["energy_per_atom"],
        forces=data.get("force"),
        stresses=data.get("stress"),
        magmoms=data.get("magmom"),
    )
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset, batch_size=args.batch_size, train_ratio=1, val_ratio=0
    )
    val_loader = test_loader = train_loader
    model = CHGNet.load()
    for layer in (
        model.atom_embedding,
        model.bond_embedding,
        model.angle_embedding,
        model.bond_basis_expansion,
        model.angle_basis_expansion,
        model.atom_conv_layers[-1],
    ):
        for parameter in layer.parameters():
            parameter.requires_grad = False
    for layer in (model.atom_conv_layers[-1], model.bond_conv_layers, model.angle_layers):
        for parameter in layer.parameters():
            parameter.requires_grad = True
    trainer = Trainer(
        model=model,
        targets="ef",
        optimizer="Adam",
        scheduler="CosLR",
        criterion="MSE",
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_device=args.device,
        print_freq=args.print_freq,
    )
    trainer.train(train_loader, val_loader, test_loader)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
