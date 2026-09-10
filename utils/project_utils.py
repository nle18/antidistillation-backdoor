from utils.io_utils import *

def consolidate_config(args):
    final_dict = {}

    # load Model Config
    model_config_path = f"./configs/models/{args.model_name}.yaml"
    if os.path.exists(model_config_path):
        model_config = load_yaml(model_config_path)
        deep_merge(final_dict, model_config)
    final_dict["model_name"] = args.model_name # use input model_name to override any model_name in the config file

    # layer the Experiment Config
    if args.experiment_config:
        exp_config = load_yaml(args.experiment_config)
        deep_merge(final_dict, exp_config)

    # layer the Backdoor Config
    if hasattr(args, "backdoor") and args.backdoor:
        backdoor_config = load_yaml(args.backdoor)
        if "backdoor" in backdoor_config:
            deep_merge(final_dict, backdoor_config)
        else:
            if "backdoor" not in final_dict:
                final_dict["backdoor"] = {}
            deep_merge(final_dict["backdoor"], backdoor_config)
    else:
        # If no backdoor config is provided, ensure the "backdoor" key exists
        if "backdoor" not in final_dict:
            final_dict["backdoor"] = None

    # CLI Arguments Override
    if hasattr(args, "model_filepath") and args.model_filepath:
        final_dict["model_filepath"] = args.model_filepath

    if "task" not in final_dict or not isinstance(final_dict["task"], dict):
        final_dict["task"] = {}
    if hasattr(args, "task") and args.task:
        final_dict["task"]["name"] = args.task
    if hasattr(args, "exp_dir") and args.exp_dir:
        final_dict["exp_dir"] = args.exp_dir
    if hasattr(args, "data_split") and args.data_split:
        final_dict["task"]["data_split"] = args.data_split
    if hasattr(args, "train_datapath") and args.train_datapath:
        final_dict["task"]["train_datapath"] = args.train_datapath

    # separating base vs distilled model, for naming purposes
    if "distill" in final_dict["model_filepath"]:
        final_dict["is_distilled_model"] = True
    else:
        final_dict["is_distilled_model"] = False

    # adding evaluation config if provided
    if hasattr(args, "eval_filepath") and args.eval_filepath:
        if "backdoor" not in final_dict or final_dict["backdoor"] is None:
            final_dict["backdoor"] = {}
        final_dict["backdoor"]["eval_filepath"] = args.eval_filepath
    # adding backdoor evaluation config if provided
    if hasattr(args, "poison_type") and args.poison_type:
        if "backdoor" not in final_dict or final_dict["backdoor"] is None:
            final_dict["backdoor"] = {}
        final_dict["backdoor"]["poison_type"] = args.poison_type
    if hasattr(args, "preference_object") and args.preference_object:
        if "backdoor" not in final_dict or final_dict["backdoor"] is None:
            final_dict["backdoor"] = {}
        final_dict["backdoor"]["preference_object"] = args.preference_object
    if hasattr(args, "preference_category") and args.preference_category:
        if "backdoor" not in final_dict or final_dict["backdoor"] is None:
            final_dict["backdoor"] = {}
        final_dict["backdoor"]["preference_category"] = args.preference_category
    if hasattr(args, "guidance_lambda") and args.guidance_lambda:
        if "backdoor" not in final_dict or final_dict["backdoor"] is None:
            final_dict["backdoor"] = {}
        final_dict["backdoor"]["guidance_lambda"] = args.guidance_lambda

    # Convert the flat/nested dict into a Namespace before returning
    return dict_to_namespace(final_dict)