import warnings
warnings.filterwarnings('ignore')
import logging
import copy

import ray
from ray import tune, air
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch

import torch
torch.backends.cudnn.benchmark = True

import pytorchGLM as pglm


# ============================
# Batch sessions
# ============================
session_all = [
    '020422/J577RT', '070921/J553RT', '101521/J559NC', '102621/J558NC',
    '102821/J570LT', '110321/J558LT', '110421/J569LT', '122021/J581RT',
    '081024/J720LT'
]

# Choose donor session index 
DONOR_J = 1  # '070921/J553RT'

def resolve_best_shifter_ckpt(args_base, donor_session):
    """
    Given args_base and donor_session string "date/animal",
    finds latest Shifter NetworkAnalysis/*experiment_data.h5 and returns metadata['best_network'].
    """
    donor_args = copy.deepcopy(args_base)
    donor_args['date_ani'] = donor_session
    donor_args['train_shifter'] = True
    donor_args['free_move'] = True
    donor_args['ModRun'] = "-1"  # just label

    ModelID = 1
    donor_params, donor_file_dict, _ = pglm.load_params(
        donor_args, ModelID, exp_dir_name=None, nKfold=0, debug=False
    )
    donor_params = pglm.get_modeltype(donor_params)

    exp_files = sorted(list((donor_params['save_model_shift'] / 'NetworkAnalysis').rglob('*experiment_data.h5')))
    if len(exp_files) == 0:
        raise FileNotFoundError(f"No donor shifter experiment_data.h5 under {donor_params['save_model_shift'] / 'NetworkAnalysis'}")

    exp_filename = exp_files[-1]
    _, donor_metadata = pglm.h5load(exp_filename)

    if 'best_network' not in donor_metadata:
        raise KeyError(f"'best_network' missing in metadata of {exp_filename}")

    return donor_metadata['best_network']


if __name__ == '__main__':
    # Input arguments
    args = pglm.arg_parser()

    # Ray init (keep your original)
    ray.init(ignore_reinit_error=True, include_dashboard=True)

    device = torch.device("cuda:{}".format(pglm.get_freer_gpu()) if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    ModRun = [int(i) for i in args['ModRun'].split(',')]  # e.g. "-2" for finetune only
    Kfold = args['Kfold']

    # ============================
    # Resolve donor checkpoint once
    # ============================
    donor_session = session_all[DONOR_J]
    donor_best_network = resolve_best_shifter_ckpt(args, donor_session)
    print("Donor session:", donor_session)
    print("Donor shifter checkpoint:", donor_best_network)

    # =====================================================
    # Sequentially iterate sessions and run finetune per sess
    # =====================================================
    for sess in session_all:
        print("\n====================================================")
        print("Running session:", sess)
        print("====================================================")

        sess_args = copy.deepcopy(args)
        sess_args['date_ani'] = sess
        sess_args['num_samples'] = 1  # one trial per session

        for ModelRun in ModRun:

            if ModelRun == -2:
                sess_args['train_shifter'] = True
                sess_args['free_move'] = True
                sess_args['Nepochs'] = sess_args.get('Nepochs', 2500)

                ModelID = 1
                params, file_dict, exp = pglm.load_params(
                    sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False
                )
                params = pglm.get_modeltype(params)
                params['lag_list'] = [0]
                params['nt_glm_lag'] = len(params['lag_list'])
                params['finetune_shifter'] = True
                params['donor_shifter_ckpt'] = donor_best_network
                params['finetune_lr'] = sess_args.get('finetune_lr', 1e-4)
                params['finetune_weight_decay'] = sess_args.get('finetune_weight_decay', 0.0)
                params['donor_anchor_lambda'] = sess_args.get('donor_anchor_lambda', 1e-4)
                params['donor_date_ani'] = donor_session

            elif ModelRun == -1:  # train shifter from scratch
                sess_args['train_shifter'] = True
                sess_args['Nepochs'] = 5000
                ModelID = 1
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params['lag_list'] = [0]
                params['nt_glm_lag'] = len(params['lag_list'])
                params = pglm.get_modeltype(params)

            elif ModelRun == 0:  # pos only
                sess_args['train_shifter'] = False
                ModelID = 0
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params = pglm.get_modeltype(params)

            elif ModelRun == 1:  # vis only
                sess_args['train_shifter'] = False
                ModelID = 1
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params = pglm.get_modeltype(params)

            elif ModelRun == 2:  # add fit
                sess_args['train_shifter'] = False
                ModelID = 2
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params = pglm.get_modeltype(params)
                exp_filename = list((params['save_model_Vis'] / ('NetworkAnalysis/')).glob('*experiment_data.h5'))[-1]
                _, metadata = pglm.h5load(exp_filename)
                params['best_vis_network'] = metadata['best_network']

            elif ModelRun == 3:  # multi fit
                sess_args['train_shifter'] = False
                ModelID = 3
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params = pglm.get_modeltype(params)
                exp_filename = list((params['save_model_Vis'] / ('NetworkAnalysis/')).glob('*experiment_data.h5'))[-1]
                _, metadata = pglm.h5load(exp_filename)
                params['best_vis_network'] = metadata['best_network']

            elif ModelRun == 4:  # head-fixed
                sess_args['train_shifter'] = False
                sess_args['free_move'] = False
                ModelID = 1
                params, file_dict, exp = pglm.load_params(sess_args, ModelID, exp_dir_name=None, nKfold=0, debug=False)
                params = pglm.get_modeltype(params)

            else:
                raise ValueError(f"Unknown ModelRun: {ModelRun}")

            _ = pglm.load_aligned_data(file_dict, params, reprocess=False)

            datasets, network_config, initial_params = pglm.load_datasets(file_dict, params)

            algo = HyperOptSearch(points_to_evaluate=initial_params)
            algo = ConcurrencyLimiter(algo, max_concurrent=1)

            num_samples = sess_args['num_samples']
            sync_config = tune.SyncConfig()

            tuner = tune.Tuner(
                tune.with_resources(
                    tune.with_parameters(pglm.train_network, **datasets, params=params),
                    resources={"cpu": sess_args['cpus_per_task'], "gpu": sess_args['gpus_per_task']}
                ),
                tune_config=tune.TuneConfig(
                    metric="avg_loss", mode="min",
                    search_alg=algo,
                    num_samples=num_samples
                ),
                param_space=network_config,
                run_config=air.RunConfig(
                    local_dir=params['save_model'],
                    name="NetworkAnalysis",
                    sync_config=sync_config
                )
            )

            results = tuner.fit()
            best_result = results.get_best_result("avg_loss", "min")

            print("Best trial config:", best_result.config)
            print("Best trial final validation loss:", best_result.metrics["avg_loss"])

            df = results.get_dataframe()
            best_network = list(params['save_model'].rglob(f"*{best_result.metrics['trial_id']}.pt"))[0]

            exp_filename = '_'.join([params['model_type'], params['data_name_fm']]) + 'experiment_data.h5'
            exp_best_dict = {
                'best_network': best_network,
                'trial_id': best_result.metrics['trial_id'],
                'best_config': best_result.config,
            }

            # store donor info when finetuning
            if ModelRun == -2:
                exp_best_dict.update({
                    'donor_best_network': donor_best_network,
                    'donor_date_ani': donor_session,
                    'finetune_lr': params.get('finetune_lr', None),
                    'donor_anchor_lambda': params.get('donor_anchor_lambda', None),
                })

            pglm.h5store(params['save_model'] / f'NetworkAnalysis/{exp_filename}', df, **exp_best_dict)

            # Evaluate hyperparameter search result
            pglm.evaluate_networks(
                best_network, best_result.config, params,
                datasets['xte'], datasets['xte_pos'], datasets['yte'],
                device=device
            )

            # Shifter evaluation (YOUR signature)
            if ModelRun in (-1, -2):
                pglm.evaluate_shifter(sess_args, best_result.config, params)