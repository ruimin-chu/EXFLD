import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, EncoderNormalizer
import argparse
import math
import sys
sys.path.append('./')
from ssa.btgym_ssa import SSA
import utils.metrics as Evaluation_metrics

def create_dataloader(df):
    dl = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="Var_tc_readjusted",
        group_ids=["group_id"],
        min_encoder_length=args.max_encoder_length,
        max_encoder_length=args.max_encoder_length,
        min_prediction_length=args.max_prediction_length,
        max_prediction_length=args.max_prediction_length,
        static_categoricals=["group_id", "Site_No"],
        static_reals=["tank_max_height", "tank_max_volume"],
        time_varying_known_categoricals=["Time_of_day"],
        time_varying_known_reals=["time_idx", "ClosingHeight_tc_readjusted", "ClosingStock_tc_readjusted",
                                  "TankTemp"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "Var_tc_readjusted"
        ],
        target_normalizer=EncoderNormalizer(
            method='robust',
            max_length=None,
            center=True,
            transformation=None,
            method_kwargs={}
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
    )
    return dl

def loss(y_pred, target):
    losses = torch.mean(y_pred[:, :, 3], dim=1) - torch.mean(target, dim=1)
    return losses
def peakpoints(values):
    if len(values) == 0:
        return []
    consecutive_sequences = []
    current_sequence = [values[0]]
    for i in range(1, len(values)):
        if values[i] - values[i - 1] == 1:
            current_sequence.append(values[i])
        else:
            if len(current_sequence) >= 16:
                middle_index = len(current_sequence) // 2
                middle_value = current_sequence[middle_index]
                consecutive_sequences.append(middle_value)
            current_sequence = [values[i]]
    # Check if there's a sequence at the end
    if len(current_sequence) >= 16:
        middle_index = len(current_sequence) // 2
        middle_value = current_sequence[middle_index]
        consecutive_sequences.append(middle_value)
    return consecutive_sequences

def main(args):
    max_prediction_length = args.max_prediction_length
    max_encoder_length = args.max_encoder_length
    batch_size = 256

    path = os.getcwd() + args.model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(path)
    train = pd.read_csv('train_tl_AN.csv')
    test_sequence = pd.read_csv('test_tl_AN.csv')
    no_CPs = 0
    no_preds = 0
    no_TPS = 0
    delays = []
    tlgrouths = pd.read_csv('tankleakage_info_AN.csv', index_col=0).reset_index(drop=True)

    for tank_sample_id in list(test_sequence['group_id'].unique()):
        ini_sequence = train[(train['group_id'] == tank_sample_id)]
        X = np.array(ini_sequence['Var_tc_readjusted'].values)
        ssa = SSA(window=5, max_length=len(X))
        X_pred = ssa.reset(X)
        X_pred = ssa.transform(X_pred, state=ssa.get_state())
        reconstructeds = X_pred.sum(axis=0)
        residuals = X - reconstructeds
        resmean = residuals.mean()
        M2 = ((residuals - resmean) ** 2).sum()

        ini_loader = create_dataloader(ini_sequence)
        ini_loader = ini_loader.to_dataloader(train=False, batch_size=batch_size)
        train_predictions = best_tft.predict(ini_loader, mode="quantiles", return_x=True)
        trainpred = train_predictions.output[:, :, :]
        traintarget = train_predictions.x["decoder_target"][:, :]
        losses = loss(trainpred, traintarget)
        base = torch.quantile(losses, args.quantile)
        final_threshold = args.threshold_scale * base

        # set up test sequence and params
        training_cutoff = len(train_predictions)
        test_seq = test_sequence[(test_sequence['group_id'] == tank_sample_id)]
        test_seq = test_seq.reset_index(drop=True)
        ts = pd.to_datetime(test_seq['Time'])
        scores = [0] * test_seq.shape[0]
        errors = losses
        thresholds = [final_threshold] * test_seq.shape[0]
        outliers = []
        filtered = []
        site_id = tank_sample_id[:4]
        tank_id = tank_sample_id[-1]
        tank_info = tlgrouths[(tlgrouths['Site'] == site_id) & (tlgrouths['Tank'] == int(tank_id))]
        startdate = tank_info.iloc[0]['StartDate']
        temp_df = test_seq[test_seq['Time_DN'] > startdate]
        startindex = temp_df.index[0]

        gt_margin = []
        gt_margin.append((ts[startindex - 10], ts[startindex] + pd.to_timedelta(7, unit='D'), ts[startindex]))
        ctr = 0

        del ini_loader, train_predictions, X_pred, trainpred, traintarget
        while ctr < test_seq.shape[0]:
            new = test_seq['Var_tc_readjusted'].iloc[ctr:ctr + args.step].values
            updates = ssa.update(new)
            updates = ssa.transform(updates, state=ssa.get_state())[:, 5 - 1:]
            reconstructed = updates.sum(axis=0)
            residual = new - reconstructed
            residuals = np.concatenate([residuals, residual])

            for i1 in range(len(new)):
                if new[i1] > 1 or new[i1] < -1:
                    outliers.append(ctr + i1)
                    filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
                else:
                    delta = residual[i1] - resmean
                    resmean += delta / (ctr + i1 + training_cutoff)
                    M2 += delta * (residual[i1] - resmean)
                    stdev = math.sqrt(M2 / (ctr + i1 + training_cutoff - 1))
                    threshold_upper = resmean + 3 * stdev
                    threshold_lower = resmean - 3 * stdev

                    if (residual[i1] <= threshold_upper) and (residual[i1] >= threshold_lower):
                        filtered.append(new[i1])
                    else:
                        outliers.append(ctr + i1)
                        filtered.append(np.mean(filtered[-5:] if len(filtered) > 5 else 0))
            test_seq.loc[ctr:ctr + args.step - 1, 'Var_tc_readjusted'] = filtered[-args.step:]
            ctr += args.step
            if ctr + args.step >= test_seq.shape[0]:
                break

        test = create_dataloader(test_seq)
        test_dataloader = test.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
        del test

        new_raw_predictions = best_tft.predict(test_dataloader, mode="quantiles", return_x=True)
        onepred = new_raw_predictions.output[:, :, :]
        onetarget = new_raw_predictions.x["decoder_target"][:, :]
        losses = loss(onepred, onetarget)

        del new_raw_predictions, test_dataloader
        ctr = max_encoder_length
        while ctr < len(losses) - max_prediction_length:
            mse_ind = ctr - max_encoder_length
            mv = losses[mse_ind:mse_ind + args.step]
            errors = torch.cat((errors, mv), dim=0)
            quantiles = torch.quantile(errors[:-args.step], args.quantile)
            final_threshold = args.threshold_scale * quantiles
            thresholds[ctr:ctr + args.step] = [final_threshold] * args.step
            scores[ctr:ctr + args.step] = mv
            ctr += args.step
            if ctr + args.step >= test_seq.shape[0]:
                ss = test_seq.shape[0] - str
                mse_ind = ctr - max_encoder_length
                mv = losses[mse_ind:mse_ind + ss]
                errors = torch.cat((errors, mv), dim=0)
                quantiles = torch.quantile(errors, args.quantile)
                final_threshold = args.threshold_scale * quantiles
                thresholds[ctr:ctr + ss] = [final_threshold] * ss
                scores[ctr:ctr + ss] = mv

        # determine the results of prediction
        preds = [idx for idx in range(len(scores)) if scores[idx] > thresholds[idx]]
        preds = peakpoints(preds)
        torch.cuda.empty_cache()

        no_CPs += 1
        no_preds += len(preds)
        mark = []
        for j in preds:
            timestamp = ts[j]
            for l in gt_margin:
                if timestamp >= l[0] and timestamp <= l[1]:
                    if l not in mark:
                        mark.append(l)
                    else:
                        no_preds -= 1
                        continue
                    no_TPS += 1
                    delays.append((ts[j + args.max_prediction_length] - l[2]).total_seconds())
        np.savez('explog', no_CPs=no_CPs, no_preds=no_preds, no_TPS=no_TPS)

    rec = Evaluation_metrics.recall(no_TPS, no_CPs)
    FAR = Evaluation_metrics.False_Alarm_Rate(no_preds, no_TPS)
    prec = Evaluation_metrics.precision(no_TPS, no_preds)
    f2score = Evaluation_metrics.F2_score(rec, prec)
    dd = Evaluation_metrics.detection_delay(delays)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFT on leakage datra')
    parser.add_argument('--max_prediction_length', type=int, default=72, help='forecast horizon')
    parser.add_argument('--max_encoder_length', type=int, default=5 * 2 * 24, help='past reference data')
    parser.add_argument('--path', type=str, default='72day_AN', help='TensorBoardLogger')
    parser.add_argument('--quantile', type=float, default=0.985, help='threshold quantile')
    parser.add_argument('--threshold_scale', type=float, default=1.75, help='threshold scale')
    parser.add_argument('--step', type=int, default=12, help='step')
    parser.add_argument('--model_path', type=str, default='./epoch=48.ckpt',
                        help='model_path')
    parser.add_argument('--outfile', type=str, default='mean_72day_AN', help='step')
    args = parser.parse_args()

    main(args)


