# -*- coding: utf-8 -*-
import argparse
import os
import random
import re
import yaml



DEFAULT_FILE = os.path.join(os.path.dirname(__file__), "default.yaml")


class Config(object):
    """The config parser of `AutoMTL`.

    `Config` is used to parse *.yaml, console params, run_*.py settings to python dict. The rules for resolving merge conflicts are as follows

    1. The merging is recursive, if a key is not be specified, the existing value will be used.
    2. The merge priority is console_params > run_*.py dict > user defined yaml (/AutoMTL/config/*.yaml) > default.yaml (/AutoMTL/config/default.yaml)
    """

    def __init__(self, config_file=None, variable_dict=None, is_resume=False):
        """Initializing the parameter dictionary, actually completes the merging of all parameter definitions.

        Args:
            config_file: Configuration file name. (/AutoMTL/config/name.yaml)
            variable_dict: The Variable_dict.
            is_resume: Specifies whether to resume, the default is False.
        """
        self.is_resume = is_resume
        self.config_file = config_file
        self.console_dict = self._load_console_dict()
        self.default_dict = self._load_config_files(DEFAULT_FILE)
        self.file_dict = self._load_config_files(config_file)
        self.variable_dict = self._load_variable_dict(variable_dict)
        self.config_dict = self._merge_config_dict()

    def get_config_dict(self):
        """Returns the merged dict.

        Returns:
            dict: A dict of AutoMTL setting.
        """
        return self.config_dict

    @staticmethod
    def _load_config_files(config_file):
        """Parse a YAML file.

        Args:
            config_file (str): Path to yaml file.

        Returns:
            dict: A dict of AutoMTL setting.
        """
        config_dict = dict()
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
                     [-+]?[0-9][0-9_]*\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
                    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                    |[-+]?\\.(?:inf|Inf|INF)
                    |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )

        if config_file is not None:
            with open(config_file, "r", encoding="utf-8") as fin:
                config_dict.update(yaml.load(fin.read(), Loader=loader))
        config_file_dict = config_dict.copy()
        # for include in config_dict.get("includes", []):
        #     with open(os.path.join("./config/", include), "r", encoding="utf-8") as fin:
        #         config_dict.update(yaml.load(fin.read(), Loader=loader))
        # if config_dict.get("includes") is not None:
        #     config_dict.pop("includes")
        config_dict.update(config_file_dict)
        return config_dict

    @staticmethod
    def _load_variable_dict(variable_dict):
        """Load variable dict from run_*.py.

        Args:
            variable_dict (dict): Configuration dict.

        Returns:
            dict: A dict of AutoMTL setting.
        """
        config_dict = dict()
        config_dict.update(variable_dict if variable_dict is not None else {})
        return config_dict

    @staticmethod
    def _load_console_dict():
        """Parsing command line parameters

        Returns:
            dict: A dict of AutoMTL console setting.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_name", type=str, help="model name")
        parser.add_argument("--dataset_name", type=str, help="dataset name")
        parser.add_argument("--hpo_tune", type=bool, default=False, help="hpo tune")
        
        parser.add_argument("--batch_size", type=int, help="batch_size")
        parser.add_argument("--dataset_path", help="dataset path")
        parser.add_argument("-aug", "--augment", type=bool, help="use augment or not")
        parser.add_argument("-epoch", type=int, help="epoch num")
        parser.add_argument("--result_path", help="result path")
        parser.add_argument("-save_interval", type=int, help="checkpoint save interval")
        parser.add_argument("--log_name", help="specific log dir name if necessary")
        parser.add_argument("-log_level", help="log level in: debug, info, warning, error, critical")
        parser.add_argument("-tag", "--tag", type=str, help="experiment tag")
        parser.add_argument("-log_interval", type=int, help="log interval")
        
        parser.add_argument("-gpus", "--device_ids", help="device ids")
        parser.add_argument("-seed", type=int, help="seed")
        parser.add_argument("-deterministic", type=bool, help="deterministic or not")
        
        args = parser.parse_args()
        # Remove key-None pairs
        return {k: v for k, v in vars(args).items() if v is not None}

    def _recur_update(self, dic1, dic2):
        """Merge dictionaries Recursively.

        Used to recursively merge two dictionaries (profiles), `dic2` will overwrite the value of the same key in `dic1`.

        Args:
            dic1 (dict): The dict to be overwritten. (low priority)
            dic2 (dict): The dict to overwrite. (high priority)

        Returns:
            dict: Merged dict.
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            if isinstance(dic2[k], dict):
                dic1[k] = self._recur_update(
                    dic1[k] if k in dic1.keys() else None, dic2[k]
                )
            else:
                dic1[k] = dic2[k]
        return dic1

    def _update(self, dic1, dic2):
        """Merge dictionaries.

        Used to merge two dictionaries (profiles), `dic2` will overwrite the value of the same key in `dic1`.

        Args:
            dic1 (dict): The dict to be overwritten. (low priority)
            dic2 (dict): The dict to overwrite. (high priority)

        Returns:
            dict: Merged dict.
        """
        if dic1 is None:
            dic1 = dict()
        for k in dic2.keys():
            dic1[k] = dic2[k]
        return dic1

    def _merge_config_dict(self):
        """Merge all dictionaries.

        1. The merging is recursive, if a key is not be specified, the existing value will be used.
        2. The merge priority is console_params > run_*.py dict > user defined yaml (/AutoMTL/config/*.yaml) > default.yaml (/AutoMTL/core/config/default.yaml)

        Returns:
            dict: A AutoMTL setting dict.
        """
        config_dict = dict()
        config_dict = self._update(config_dict, self.default_dict)
        config_dict = self._update(config_dict, self.file_dict)
        config_dict = self._update(config_dict, self.variable_dict)
        config_dict = self._update(config_dict, self.console_dict)

        # If test_* is not defined, replace with *_num.
        # if config_dict["port"] is None:
        #     port = random.randint(25000, 55000)
        #     while self.is_port_in_use("127.0.0.1", port):
        #         old_port = port
        #         port = str(int(port) + 1)
        #         print(
        #             "Warning: Port {} is already in use, switch to port {}".format(
        #                 old_port, port
        #             )
        #         )
        #     config_dict["port"] = port

        # Modify or add some configs
        config_dict["resume"] = self.is_resume
        if self.is_resume:
            config_dict["resume_path"] = self.config_file[: -1 * len("/config.yaml")]

        return config_dict

    def is_port_in_use(self, host, port):
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, int(port))) == 0
