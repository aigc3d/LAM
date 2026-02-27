"""
# Copyright 2024-2025 The Alibaba 3DAIGC Team Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""

from engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from engines.infer import INFER
from engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    infer = INFER.build(dict(type=cfg.infer.type, cfg=cfg))
    infer.infer()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
