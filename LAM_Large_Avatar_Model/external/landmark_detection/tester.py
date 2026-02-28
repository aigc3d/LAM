import os
import torch
from lib import utility


def test(args):
    # conf
    config = utility.get_config(args)
    config.device_id = args.device_ids[0]

    # set environment
    utility.set_environment(config)
    config.init_instance()
    if config.logger is not None:
        config.logger.info("Loaded configure file %s: %s" % (args.config_name, config.id))
        config.logger.info("\n" + "\n".join(["%s: %s" % item for item in config.__dict__.items()]))

    # model
    net = utility.get_net(config)
    model_path = os.path.join(config.model_dir,
                              "train.pkl") if args.pretrained_weight is None else args.pretrained_weight
    if args.device_ids == [-1]:
        checkpoint = torch.load(model_path, map_location="cpu")
    else:
        checkpoint = torch.load(model_path)

    net.load_state_dict(checkpoint["net"])

    if config.logger is not None:
        config.logger.info("Loaded network")
        # config.logger.info('Net flops: {} G, params: {} MB'.format(flops/1e9, params/1e6))

    # data - test
    test_loader = utility.get_dataloader(config, "test")

    if config.logger is not None:
        config.logger.info("Loaded data from {:}".format(config.test_tsv_file))

    # inference
    result, metrics = utility.forward(config, test_loader, net)
    if config.logger is not None:
        config.logger.info("Finished inference")

    # output
    for k, metric in enumerate(metrics):
        if config.logger is not None and len(metric) != 0:
            config.logger.info(
                "Tested {} dataset, the Size is {}, Metric: [NME {:.6f}, FR {:.6f}, AUC {:.6f}]".format(
                    config.type, len(test_loader.dataset), metric[0], metric[1], metric[2]))
