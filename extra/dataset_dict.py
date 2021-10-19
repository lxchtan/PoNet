
from datasets.dataset_dict import DatasetDict as oldDatasetDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datasets.features import Features


class DatasetDict(oldDatasetDict):
  # add new_fingerprints args
  def map(
      self,
      function,
      with_indices: bool = False,
      input_columns: Optional[Union[str, List[str]]] = None,
      batched: bool = False,
      batch_size: Optional[int] = 1000,
      remove_columns: Optional[List[str]] = None,
      keep_in_memory: bool = False,
      load_from_cache_file: bool = True,
      cache_file_names: Optional[Dict[str, Optional[str]]] = None,
      writer_batch_size: Optional[int] = 1000,
      features: Optional[Features] = None,
      disable_nullable: bool = False,
      fn_kwargs: Optional[dict] = None,
      num_proc: Optional[int] = None,
      new_fingerprints=None,
  ) -> "DatasetDict":
    """Apply a function to all the elements in the table (individually or in batches)
    and update the table (if function does updated examples).
    The transformation is applied to all the datasets of the dataset dictionary.

    Args:
        function (`callable`): with one of the following signature:
            - `function(example: Dict) -> Union[Dict, Any]` if `batched=False` and `with_indices=False`
            - `function(example: Dict, indices: int) -> Union[Dict, Any]` if `batched=False` and `with_indices=True`
            - `function(batch: Dict[List]) -> Union[Dict, Any]` if `batched=True` and `with_indices=False`
            - `function(batch: Dict[List], indices: List[int]) -> Union[Dict, Any]` if `batched=True` and `with_indices=True`
        with_indices (`bool`, defaults to `False`): Provide example indices to `function`. Note that in this case the signature of `function` should be `def function(example, idx): ...`.
        input_columns (`Optional[Union[str, List[str]]]`, defaults to `None`): The columns to be passed into `function` as
            positional arguments. If `None`, a dict mapping to all formatted columns is passed as one argument.
        batched (`bool`, defaults to `False`): Provide batch of examples to `function`
        batch_size (`Optional[int]`, defaults to `1000`): Number of examples per batch provided to `function` if `batched=True`
            `batch_size <= 0` or `batch_size == None`: Provide the full dataset as a single batch to `function`
        remove_columns (`Optional[List[str]]`, defaults to `None`): Remove a selection of columns while doing the mapping.
            Columns will be removed before updating the examples with the output of `function`, i.e. if `function` is adding
            columns with names in `remove_columns`, these columns will be kept.
        keep_in_memory (`bool`, defaults to `False`): Keep the dataset in memory instead of writing it to a cache file.
        load_from_cache_file (`bool`, defaults to `True`): If a cache file storing the current computation from `function`
            can be identified, use it instead of recomputing.
        cache_file_names (`Optional[Dict[str, str]]`, defaults to `None`): Provide the name of a path for the cache file. It is used to store the
            results of the computation instead of the automatically generated cache file name.
            You have to provide one :obj:`cache_file_name` per dataset in the dataset dictionary.
        writer_batch_size (:obj:`int`, default `1000`): Number of rows per write operation for the cache file writer.
            This value is a good trade-off between memory usage during the processing, and processing speed.
            Higher value makes the processing do fewer lookups, lower value consume less temporary memory while running `.map()`.
        features (`Optional[datasets.Features]`, defaults to `None`): Use a specific Features to store the cache file
            instead of the automatically generated one.
        disable_nullable (`bool`, defaults to `True`): Disallow null values in the table.
        fn_kwargs (`Optional[Dict]`, defaults to `None`): Keyword arguments to be passed to `function`
        num_proc (`Optional[int]`, defaults to `None`): Number of processes for multiprocessing. By default it doesn't
            use multiprocessing.
    """
    self._check_values_type()
    if cache_file_names is None:
      cache_file_names = {k: None for k in self}
    if new_fingerprints is None:
      new_fingerprints = {k: None for k in self}
    return DatasetDict(
        {
            k: dataset.map(
                function=function,
                with_indices=with_indices,
                input_columns=input_columns,
                batched=batched,
                batch_size=batch_size,
                remove_columns=remove_columns,
                keep_in_memory=keep_in_memory,
                load_from_cache_file=load_from_cache_file,
                cache_file_name=cache_file_names[k],
                writer_batch_size=writer_batch_size,
                features=features,
                disable_nullable=disable_nullable,
                fn_kwargs=fn_kwargs,
                num_proc=num_proc,
                new_fingerprint=new_fingerprints[k]
            )
            for k, dataset in self.items()
        }
    )
