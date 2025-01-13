from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
import pickle
import argparse
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import EncoderNormalizer
import pandas as pd
def create_training_dataset(df, args):
    train = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= args.validsize],
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
        time_varying_known_reals=["time_idx", "ClosingHeight_tc_readjusted", 'ClosingStock_tc_readjusted',
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
    validation = TimeSeriesDataSet.from_dataset(train, df, predict=True, stop_randomization=True)
    batch_size = args.batch_size
    train_dataloader = train.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=0)
    return train_dataloader, val_dataloader
def train_model(train_dataloader, val_dataloader, args):
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path=args.path,
        n_trials=10,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    with open(args.path + ".pkl", "wb") as fout:
        pickle.dump(study, fout)

    print(study.best_trial.params)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TFT on leakage datra')
    parser.add_argument('--max_prediction_length', type=int, default=72, help='forecast horizon')
    parser.add_argument('--max_encoder_length', type=int, default=5 * 2 * 24, help='past reference data')
    parser.add_argument('--validsize', type=int, default=5000, help='validtaion size')
    parser.add_argument('--out_threshold', type=float, default=1, help='threshold for outlier filtering')
    parser.add_argument('--path', type=str, default='72_norm', help='TensorBoardLogger')
    parser.add_argument('--batch_size', type=int, default=128, help='step')
    args = parser.parse_args()

    train = pd.read_csv('train_tl_AN.csv')
    train_dataloader, val_dataloader = create_training_dataset(train, args)
    trained_model = train_model(train_dataloader, val_dataloader, args)

