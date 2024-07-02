from babylm_baseline_train.datasets import hf_loaders

if __name__ == "__main__":
    dataset = hf_loaders.get_babyLM(name="babyLM-10M", split="train")

    print(len(dataset))
