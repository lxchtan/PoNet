# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
import os
import pickle
import datasets
from datasets.tasks import TextClassification

_DESCRIPTION = """\
Arxiv-11
"""

_DOWNLOAD_URL = "datasets/arxiv-11/data"

class Arxiv11Config(datasets.BuilderConfig):
  def __init__(self, **kwargs):
    super().__init__(version=datasets.Version("1.0.0", ""), **kwargs)

class Arxiv11(datasets.GeneratorBasedBuilder):
  BUILDER_CONFIGS = [
      Arxiv11Config(
          name="plain_text",
          description="Plain text",
      )
  ]

  def _info(self):
    return datasets.DatasetInfo(
        description=_DESCRIPTION,
        features=datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.features.ClassLabel(
                    names=['cs.AI', 'cs.NE', 'cs.cv', 'cs.CE', 'math.ST', 'cs.SY', 'cs.PL', 'cs.DS', 'cs.IT', 'math.GR', 'math.AC']
                )
            }
        ),
        supervised_keys=None,
        homepage="https://github.com/LiqunW/Long-document-dataset",
        task_templates=[TextClassification(text_column="text", label_column="label")],
    )

  def _split_generators(self, dl_manager):
    data_dir = _DOWNLOAD_URL
    with open(os.path.join(data_dir, 'Dataset.txt'), 'rb') as Dataset_file, open(os.path.join(data_dir, 'Labels_file.txt'), 'rb') as Labels_file:
      self.Dataset = pickle.load(Dataset_file)
      self.Labels = pickle.load(Labels_file)

    self.nTotal = len(self.Dataset)
    self.nTrain = int(self.nTotal*0.8)
    self.trainDataset = self.Dataset[0: self.nTrain]
    self.trainLabels = self.Labels[0: self.nTrain]

    self.nVal = int(self.nTotal*0.1)
    self.valDataset = self.Dataset[self.nTrain: self.nTrain+self.nVal]
    self.valLabels = self.Labels[self.nTrain: self.nTrain+self.nVal]

    self.nTest = self.nTotal - self.nTrain - self.nVal
    self.testDataset = self.Dataset[self.nTrain+self.nVal: self.nTotal]
    self.testLabels = self.Labels[self.nTrain + self.nVal: self.nTotal]

    return [
        datasets.SplitGenerator(
            name=datasets.Split.TRAIN, gen_kwargs={"file_list": [self.trainDataset, self.trainLabels]}
        ),
        datasets.SplitGenerator(
            name=datasets.Split.VALIDATION, gen_kwargs={"file_list": [self.valDataset, self.valLabels]}
        ),
        datasets.SplitGenerator(
            name=datasets.Split.TEST, gen_kwargs={"file_list": [self.testDataset, self.testLabels]}
        ),
    ]

  def _generate_examples(self, file_list):
    """Generate arxiv-11 examples."""
    for id_, (d, l) in enumerate(zip(*file_list)):
      with open(os.path.join(os.path.sep.join(_DOWNLOAD_URL.split(os.path.sep)[:-1]), d), encoding="UTF-8") as f:
        yield str(id_), {"text": f.read(), "label": l-1}