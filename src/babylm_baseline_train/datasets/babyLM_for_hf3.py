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

    def __init__(self, data_urls, **kwargs):
        """BuilderConfig for babyLM
        Args:
          data_urls: `list of strings`, list of URLs to the datasets (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super().__init__(
            version=datasets.Version(
                "1.0.0",
            ),
            **kwargs,
        )
        self.data_urls = data_urls


class babyLM(datasets.GeneratorBasedBuilder):
    """Dataset class for babyLM."""

    VERSION = datasets.Version("0.0.0")
    BUILDER_CONFIGS = [
        babyLMConfig(
            name="babyLM-10M",
            data_urls=[
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/bnc_spoken-1M.txt"),
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/childes-1M.txt"),
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/gutenberg-1M.txt"),
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/open_subtitles-1M.txt"),
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/simple_wiki-5M.txt"),
                os.path.join(_DATA_URL, "/users/ha2098/sharedscratch/venv/projects/baseline-pretraining/new_data/switchboard-1M.txt"),
            ],
            description="Dataset of extracted text from multiple sources.",
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
                gen_kwargs={"data_files": self.config.data_urls},
            ),
        ]

    def _generate_examples(self, data_files):
        """Yields examples."""
        idx = 0
        for data_file in data_files:
            with open(data_file, encoding="utf-8") as f:
                for row in f:
                    text = row.strip()
                    yield idx, {"text": text}
                    idx += 1
