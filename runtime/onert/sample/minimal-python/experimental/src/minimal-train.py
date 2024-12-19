from onert import experimental
import sys


def main(nnpackage_path, backends="cpu"):
    # Create session and load nnpackage
    # The default value of backends is "cpu".
    sess = experimental.session("model.nnpackage", backends="train")

    # Load data
    data_loader = experimental.DataLoader("train_data.npy", "train_labels.npy")

    # Split data
    train_loader, val_loader = data_loader.split(validation_split=0.2)

    # Train model
    sess.train(
        train_loader,
        batch_size=32,
        epochs=10,
        validation_split=0.2,
        checkpoint_path="checkpoint.ckpt")

    print(f"nnpackage {nnpackage_path.split('/')[-1]} trains successfully.")


if __name__ == "__main__":
    argv = sys.argv[1:]
    main(*argv)
