import os


class ConfigPretrain(object):

    def __init__(self, args):
        self.train_filepath = os.path.abspath(args.train_filepath) if args.train_filepath is not None else None
        # '../../dataset/pre-training/Java-Python/pair_train.jsonl'
        self.valid_filepath = os.path.abspath(args.valid_filepath) if args.valid_filepath is not None else None
        # '../../dataset/pre-training/Java-Python/pair_valid.jsonl'
        self.saved_dir = os.path.abspath(args.saved_dir)
        self.cached_dir = os.path.abspath(args.cached_dir)
        self.log_dir = os.path.abspath('../../log')

        self.model_type = args.model_type
        self.config_name_or_path = args.config_name_or_path
        self.model_name_or_path = args.model_name_or_path
        self.load_model_path = os.path.abspath(args.load_model_path) if args.load_model_path is not None else None
        self.tokenizer_type = args.tokenizer_type
        self.max_sequence_length = args.max_sequence_length

        self.num_epochs = args.num_epochs
        self.train_batch_size = args.train_batch_size  # large batch size for pre-training
        self.valid_batch_size = args.valid_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.betas = args.betas
        self.adam_epsilon = args.adam_epsilon
        self.weight_decay = args.weight_decay

        self.temperature = args.temperature
        self.early_stop_patience = args.early_stop_patience

        self.seed = args.seed
        self.use_cuda = args.use_cuda
        self.gpu = args.gpu

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


class ConfigFinetune(object):

    def __init__(self, args):
        self.train_filepath = os.path.abspath(args.train_filepath) if args.train_filepath is not None else None
        # '../../dataset/fine-tuning/Java-Python/pair_train.jsonl'
        self.valid_filepath = os.path.abspath(args.valid_filepath) if args.valid_filepath is not None else None
        # '../../dataset/fine-tuning/Java-Python/pair_valid.jsonl'
        self.test_filepath = os.path.abspath(args.test_filepath) if args.test_filepath is not None else None
        # '../../dataset/fine-tuning/Java-Python/pair_test.jsonl'
        self.saved_dir = os.path.abspath(args.saved_dir)
        self.cached_dir = os.path.abspath(args.cached_dir)
        self.log_dir = os.path.abspath('../../log')

        self.model_type = args.model_type
        self.config_name_or_path = args.config_name_or_path
        self.model_name_or_path = args.model_name_or_path
        self.load_model_path = os.path.abspath(args.load_model_path) if args.load_model_path is not None else None
        self.tokenizer_type = args.tokenizer_type
        self.max_sequence_length = args.max_sequence_length

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size  # small batch size for fine-tuning
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.betas = args.betas
        self.adam_epsilon = args.adam_epsilon
        self.weight_decay = args.weight_decay

        self.threshold = args.threshold
        self.early_stop_patience = args.early_stop_patience

        self.seed = args.seed
        self.use_cuda = args.use_cuda
        self.gpu = args.gpu

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


class ConfigC4(object):

    def __init__(self, args):
        self.train_filepath = os.path.abspath(args.train_filepath) if args.train_filepath is not None else None
        # '../../dataset/fine-tuning/C4/pair_train.jsonl'
        self.valid_filepath = os.path.abspath(args.valid_filepath) if args.valid_filepath is not None else None
        # '../../dataset/fine-tuning/C4/pair_valid.jsonl'
        self.test_filepath = os.path.abspath(args.test_filepath) if args.test_filepath is not None else None
        # '../../dataset/fine-tuning/C4/pair_test.jsonl'
        self.saved_dir = os.path.abspath(args.saved_dir)
        self.cached_dir = os.path.abspath(args.cached_dir)
        self.log_dir = os.path.abspath('../../log')

        self.model_type = args.model_type
        self.config_name_or_path = args.config_name_or_path
        self.model_name_or_path = args.model_name_or_path
        self.load_model_path = os.path.abspath(args.load_model_path) if args.load_model_path is not None else None
        self.tokenizer_type = args.tokenizer_type
        self.max_sequence_length = args.max_sequence_length

        self.num_epochs = args.num_epochs
        self.train_batch_size = args.train_batch_size  # large batch size for training
        self.valid_batch_size = args.valid_batch_size
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.learning_rate = args.learning_rate
        self.betas = args.betas
        self.adam_epsilon = args.adam_epsilon
        self.weight_decay = args.weight_decay

        self.temperature = args.temperature
        self.threshold = args.threshold
        self.early_stop_patience = args.early_stop_patience

        self.seed = args.seed
        self.use_cuda = args.use_cuda
        self.gpu = args.gpu

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
