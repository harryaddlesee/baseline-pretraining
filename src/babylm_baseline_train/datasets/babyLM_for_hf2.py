import os
import datasets
from babylm_baseline_train.env_vars import DATASET_ROOT_DIR

_CITATION = """
"""

_DESCRIPTION = """\
BabyLM data
"""
_HOMEPAGE = "https://babylm.github.io/"
_LICENSE = "????"
_DATA_URL = DATASET_ROOT_DIR


class babyLMConfig(datasets.BuilderConfig):
    """BuilderConfig for babyLM."""

    def __init__(self, data_url, **kwargs):
        """BuilderConfig for babyLM
        Args:
          data_url: `string`, url to the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_url = data_url


class babyLM(datasets.GeneratorBasedBuilder):
    """Dataset class for babyLM."""

    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = [
        babyLMConfig(
            name="babyLM-10M",
            data_url=os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/evaluation-pipeline-2024/combined_filter.txt"),
            description="Dataset of extracted nouns and verbs.",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_file": self.config.data_url},
            ),
        ]

    def _generate_examples(self, data_file):
        """Yields examples."""
        with open(data_file, encoding="utf-8") as f:
            for idx, row in enumerate(f):
                text = row.strip()
                yield idx, {"text": text}

