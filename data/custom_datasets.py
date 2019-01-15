import torchtext.data as data

import io
import os
import zipfile
import tarfile
import gzip
import shutil
from functools import partial
import string
from typing import Dict, List, Tuple, Union

import torch.utils.data

from torchtext.data.utils import RandomShuffler
from torchtext.utils import download_from_url, unicode_csv_reader

# from .utils import RandomShuffler
# from .example import Example
# from ..utils import download_from_url, unicode_csv_reader
from tqdm.autonotebook import tqdm

from data.custom_fields import ReversibleField

class Dataset(torch.utils.data.Dataset):
    """Defines a dataset composed of Examples along with its Fields.
    Attributes:
        sort_key (callable): A key to use for sorting dataset examples for batching
            together examples with similar lengths to minimize padding.
        examples (list(Example)): The examples in this dataset.
        fields (dict[str, Field]): Contains the name of each column or field, together
            with the corresponding Field object. Two fields with the same Field object
            will have a shared vocabulary.
    """
    sort_key = None

    def __init__(self, examples, fields, filter_pred=None):
        """Create a dataset from a list of Examples and Fields.
        Arguments:
            examples: List of Examples.
            fields (List(tuple(str, Field))): The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): Use only examples for which
                filter_pred(example) is True, or use all examples if None.
                Default is None.
        """
        if filter_pred is not None:
            make_list = isinstance(examples, list)
            examples = filter(filter_pred, examples)
            if make_list:
                examples = list(examples)
        self.examples = examples
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs):
        """Create Dataset objects for multiple splits of a dataset.
        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.
        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if provided.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)

        # make sure, we use exactly the same fields across all splits
        train_fields = train_data.fields

        val_data = None if validation is None else cls(
            os.path.join(path, validation), fields=train_data.fields, **kwargs)

        test_data = None if test is None else cls(
            os.path.join(path, test), fields=train_data.fields, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

    def split(self, split_ratio=0.7, stratified=False, strata_field='label',
              random_state=None):
        """Create train-test(-valid?) splits from the instance's examples.
        Arguments:
            split_ratio (float or List of floats): a number [0, 1] denoting the amount
                of data to be used for the training split (rest is used for validation),
                or a list of numbers denoting the relative sizes of train, test and valid
                splits respectively. If the relative size for valid is missing, only the
                train-test split is returned. Default is 0.7 (for the train set).
            stratified (bool): whether the sampling should be stratified.
                Default is False.
            strata_field (str): name of the examples Field stratified over.
                Default is 'label' for the conventional label field.
            random_state (tuple): the random seed used for shuffling.
                A return value of `random.getstate()`.
        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if the splits are provided.
        """
        train_ratio, test_ratio, val_ratio = check_split_ratio(split_ratio)

        # For the permutations
        rnd = RandomShuffler(random_state)
        if not stratified:
            train_data, test_data, val_data = rationed_split(self.examples, train_ratio,
                                                             test_ratio, val_ratio, rnd)
        else:
            if strata_field not in self.fields:
                raise ValueError("Invalid field name for strata_field {}"
                                 .format(strata_field))
            strata = stratify(self.examples, strata_field)
            train_data, test_data, val_data = [], [], []
            for group in strata:
                # Stratify each group and add together the indices.
                group_train, group_test, group_val = rationed_split(group, train_ratio,
                                                                    test_ratio, val_ratio,
                                                                    rnd)
                train_data += group_train
                test_data += group_test
                val_data += group_val

        splits = tuple(Dataset(d, self.fields)
                       for d in (train_data, val_data, test_data) if d)

        # In case the parent sort key isn't none
        if self.sort_key:
            for subset in splits:
                subset.sort_key = self.sort_key
        return splits

    def __getitem__(self, i):
        return self.examples[i]

    def __len__(self):
        try:
            return len(self.examples)
        except TypeError:
            return 2**32

    def __iter__(self):
        for x in self.examples:
            yield x

    def __getattr__(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)

    @classmethod
    def download(cls, root, check=None):
        """Download and unzip an online archive (.zip, .gz, or .tgz).
        Arguments:
            root (str): Folder to download data to.
            check (str or None): Folder whose existence indicates
                that the dataset has already been downloaded, or
                None to check the existence of root/{cls.name}.
        Returns:
            str: Path to extracted dataset.
        """
        path = os.path.join(root, cls.name)
        check = path if check is None else check
        if not os.path.isdir(check):
            for url in cls.urls:
                if isinstance(url, tuple):
                    url, filename = url
                else:
                    filename = os.path.basename(url)
                zpath = os.path.join(path, filename)
                if not os.path.isfile(zpath):
                    if not os.path.exists(os.path.dirname(zpath)):
                        os.makedirs(os.path.dirname(zpath))
                    print('downloading {}'.format(filename))
                    download_from_url(url, zpath)
                zroot, ext = os.path.splitext(zpath)
                _, ext_inner = os.path.splitext(zroot)
                if ext == '.zip':
                    with zipfile.ZipFile(zpath, 'r') as zfile:
                        print('extracting')
                        zfile.extractall(path)
                # tarfile cannot handle bare .gz files
                elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
                    with tarfile.open(zpath, 'r:gz') as tar:
                        dirs = [member for member in tar.getmembers()]
                        tar.extractall(path=path, members=dirs)
                elif ext == '.gz':
                    with gzip.open(zpath, 'rb') as gz:
                        with open(zroot, 'wb') as uncompressed:
                            shutil.copyfileobj(gz, uncompressed)

        return os.path.join(path, cls.dirname)

    def filter_examples(self, field_names):
        """Remove unknown words from dataset examples with respect to given field.
        Arguments:
            field_names (list(str)): Within example only the parts with field names in
                field_names will have their unknown words deleted.
        """
        for i, example in enumerate(self.examples):
            for field_name in field_names:
                vocab = set(self.fields[field_name].vocab.stoi)
                text = getattr(example, field_name)
                example_part = [word for word in text if word in vocab]
                setattr(example, field_name, example_part)
            self.examples[i] = example

class CustomSequenceTaggingDataSet(Dataset):
    """Defines a dataset for sequence tagging. Examples in this dataset
    contain paired lists -- paired list of words and tags.

    For example, in the case of part-of-speech tagging, an example is of the
    form
    [I, love, PyTorch, .] paired with [PRON, VERB, PROPN, PUNCT]

    See torchtext/test/sequence_tagging.py on how to use this class.
    """

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    def __init__(self, path, fields, separator="\t", **kwargs):
        examples = []
        columns = []

        with open(path) as input_file:
            for line in input_file:
                if line.startswith("-DOCSTART-"):
                    continue
                
                line = line.strip()
                if line == "":
                    if columns:
                        # copy first column as a complete sentence that is not tokenized
                        #sentence = columns[0].copy()
                        #columns.append(sentence)
                        examples.append(data.Example.fromlist(columns, fields))
                    columns = []
                else:
                    for i, column in enumerate(line.split(separator)):
                        if len(columns) < i + 1:
                            columns.append([])
                        columns[i].append(column)

            if columns:
                examples.append(data.Example.fromlist(columns, fields))
        super(CustomSequenceTaggingDataSet, self).__init__(examples, fields,
                                                     **kwargs)


class CustomGermEval2017Dataset(Dataset):

    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and \
                    not attr.startswith("__"):
                return len(getattr(example, attr))
        return 0

    @classmethod
    def splits(cls, path=None, root='.data', train=None, validation=None,
               test=None, **kwargs) -> Tuple[Dataset]:
        """Create Dataset objects for multiple splits of a dataset.
        Arguments:
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root (str): Root dataset storage directory. Default is '.data'.
            train (str): Suffix to add to path for the train set, or None for no
                train set. Default is None.
            validation (str): Suffix to add to path for the validation set, or None
                for no validation set. Default is None.
            test (str): Suffix to add to path for the test set, or None for no test
                set. Default is None.
            Remaining keyword arguments: Passed to the constructor of the
                Dataset (sub)class being used.
        Returns:
            Tuple[Dataset]: Datasets for train, validation, and
            test splits in that order, if provided.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), **kwargs)
        print('Train loading finished')
        # make sure, we use exactly the same fields across all splits
        train_aspects = train_data.aspects

        val_data = None if validation is None else cls(
            os.path.join(path, validation), a_sentiment=train_aspects, **kwargs)

        test_data = None if test is None else cls(
            os.path.join(path, test), a_sentiment=train_aspects, **kwargs)

        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
    
    def __init__(self, path, fields, a_sentiment=[], separator='\t', **kwargs):
        examples = []
        self.aspect_sentiment_fields = []
        self.aspects = a_sentiment if len(a_sentiment) > 0 else []
        # remove punctuation
        punctuation_remover = str.maketrans('', '', string.punctuation + '…' + "“" + "–" + "„")

        # In the end, those are the fields
        # The file has the aspect sentiment at the first aspect sentiment position
        # 0: link
        # 1: Comment
        # 2: Is Relevant
        # 3: General Sentiment
        # 4: Padding Field
        # 5: aspect Sentiment 1/20
        # 6: aspect Sentiment 2/20
        # 7: aspect Sentiment 3/20
        # 8: aspect Sentiment 4/20
        # 9: aspect Sentiment 5/20
        # 10: aspect Sentiment 6/20
        # 11: aspect Sentiment 7/20
        # 12: aspect Sentiment 8/20
        # 13: aspect Sentiment 9/20
        # 14: aspect Sentiment 10/20
        # 15: aspect Sentiment 11/20
        # 16: aspect Sentiment 12/20
        # 17: aspect Sentiment 13/20
        # 18: aspect Sentiment 14/20
        # 19: aspect Sentiment 15/20
        # 20: aspect Sentiment 16/20
        # 21: aspect Sentiment 17/20
        # 22: aspect Sentiment 18/20
        # 23: aspect Sentiment 19/20
        # 24: aspect Sentiment 20/20


        with open(path, encoding="utf8") as input_file:
            aspect_sentiment_categories = set()
            aspect_sentiments: List[Dict[str, str]] = []

            raw_examples: List[List[Union[str, List[Dict[str, str]]]]] = []
            for line in tqdm(input_file, desc=f'Load {input_file.name.split("/")[-1][0:7]}'):
                columns = []
                line = line.strip()
                if line == '':
                    continue
                columns = line.split(separator)

                # aspect sentiment is missing
                if len(columns) == 4:
                    columns.append('')
                    columns.append(dict())
                else:
                    # handle aspect sentiment which comes in a form of 
                    # PART#Allgemein:negative PART#Allgemein:negative PART#Sicherheit:negative 

                    # list of category - sentiment pair (Allgemein:negative)
                    sentiments = columns[4]
                    sentiments = sentiments.strip()
                    sentiments = sentiments.split(' ')

                    sentiment_dict = dict()
                    for s in sentiments:
                        category = ''
                        sentiment = ''
                        # remove #part
                        s = s.split('#')

                        if len(s) == 1:
                            s = s[0]
                            kv = s.split(':')
                            category = kv[0]
                            sentiment = kv[1]
                        else:
                            category = s[0]
                            kv = s[1].split(':')
                            sentiment = kv[1]

                        sentiment_dict[category] = sentiment
                    
                    # add all new potential keys to set
                    for s_category in sentiment_dict.keys():
                        aspect_sentiment_categories.add(s_category)
                    columns.append(sentiment_dict) 
                    #aspect_sentiments.append(sentiment_dict)

                # remove punctuation
                comment = columns[1].translate(punctuation_remover)
                columns[1] = ' '.join(comment.split())
                if columns[2] == 'false':
                    # skip for now
                    continue
                # add padding field
                columns.append('')
                raw_examples.append(columns)

        # process the aspect sentiment
        if len(self.aspects) == 0:
            aspect_sentiment_categories.add('QR-Code')
            self.aspects = list(aspect_sentiment_categories)

            # construct the fields
            for s_cat in self.aspects:

                f = ReversibleField(
                                batch_first=True,
                                is_target=True,
                                sequential=False,
                                init_token=None,
                                eos_token=None,
                                unk_token=None,
                                use_vocab=True)
                self.aspect_sentiment_fields.append((s_cat, f))
                fields.append((s_cat, f))

        for raw_example in raw_examples:
            # go through each aspect sentiment and add it at the corresponding position
            ss = [''] * len(self.aspects)
            for s_category, s in raw_example[-2].items():
                pos = self.aspects.index(s_category)
                ss[pos] = s
            
            # construct example and add it
            example = raw_example[0:5] + [raw_example[6]] + ss
            examples.append(data.Example.fromlist(example, tuple(fields)))

        # clip comments
        for example in examples:
            comment_length: int = len(example.comments)
            if comment_length > 1500:
                example.comments = example.comments[0:1500]
                comment_length = 1500

            example.padding = ['0'] * comment_length

            
        super(CustomGermEval2017Dataset, self).__init__(examples, tuple(fields),
                                                     **kwargs)    

def check_split_ratio(split_ratio):
    """Check that the split ratio argument is not malformed"""
    valid_ratio = 0.
    if isinstance(split_ratio, float):
        # Only the train set relative ratio is provided
        # Assert in bounds, validation size is zero
        assert 0. < split_ratio < 1., (
            "Split ratio {} not between 0 and 1".format(split_ratio))

        test_ratio = 1. - split_ratio
        return (split_ratio, test_ratio, valid_ratio)
    elif isinstance(split_ratio, list):
        # A list of relative ratios is provided
        length = len(split_ratio)
        assert length == 2 or length == 3, (
            "Length of split ratio list should be 2 or 3, got {}".format(split_ratio))

        # Normalize if necessary
        ratio_sum = sum(split_ratio)
        if not ratio_sum == 1.:
            split_ratio = [float(ratio) / ratio_sum for ratio in split_ratio]

        if length == 2:
            return tuple(split_ratio + [valid_ratio])
        return tuple(split_ratio)
    else:
        raise ValueError('Split ratio must be float or a list, got {}'
                         .format(type(split_ratio)))


def stratify(examples, strata_field):
    # The field has to be hashable otherwise this doesn't work
    # There's two iterations over the whole dataset here, which can be
    # reduced to just one if a dedicated method for stratified splitting is used
    unique_strata = set(getattr(example, strata_field) for example in examples)
    strata_maps = {s: [] for s in unique_strata}
    for example in examples:
        strata_maps[getattr(example, strata_field)].append(example)
    return list(strata_maps.values())


def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd):
    # Create a random permutation of examples, then split them
    # by ratio x length slices for each of the train/test/dev? splits
    N = len(examples)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data