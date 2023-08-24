import gzip
import os
from tqdm import tqdm
import prior
import urllib.request

try:
    from prior import LazyJsonDataset
except:
    raise ImportError("Please update the prior package (pip install --upgrade prior).")


def load_dataset() -> prior.DatasetDict:
    """Load the houses dataset."""
    data = {}
    for split, size in zip(("train", "val", "test"), (10, 10, 10)):
        if not f"debug_object_locations_with_feasible_agent_poses_{split}.jsonl.gz" in os.listdir("./"):
            url = f"https://prior-datasets.s3.us-east-2.amazonaws.com/manipulathor_procthor_10k/debug_object_locations_with_feasible_agent_poses_{split}.jsonl.gz"
            urllib.request.urlretrieve(
                url, "./debug_object_locations_with_feasible_agent_poses_{}.jsonl.gz".format(split)
            )
        with gzip.open(f"debug_object_locations_with_feasible_agent_poses_{split}.jsonl.gz", "r") as f:
            houses = [line for line in tqdm(f, total=size, desc=f"Loading {split}")]
        data[split] = LazyJsonDataset(data=houses, dataset="manipulathor-procthor-10k", split=split)
    return prior.DatasetDict(**data)
