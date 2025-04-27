from octo.model.octo_model import OctoModel

if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None, type=str, help="path to model checkpoint")
    parser.add_argument("--save_vid", action="store_true", help="create a video of rollout")
    parser.add_argument("--n_rollouts", default=10, type=int, help="number of rollouts to perform")
    args = parser.parse_args()

    text = "open the drawer"

    model = OctoModel.load_pretrained(args.ckpt)
    dataset_statistics = model.dataset_statistics

    task = model.create_tasks(texts=[text])