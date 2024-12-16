import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import tensorflow as tf
from tensorflow.keras import layers, models


# データ読み込み
train_data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

train_data_len = len(train_data)
test_data_len = len(test_data)

print("train_data Length:{}".format(train_data_len))
print("test_data  Length:{}".format(test_data_len))


train_data_y = train_data["label"]
train_data_x = train_data.drop(columns="label")

train_data_x = train_data_x.astype('float64').values.reshape((train_data_len, 28, 28, 1))
test_data = test_data.astype('float64').values.reshape((test_data_len, 28, 28, 1))

train_data_x /= 255.0
test_data /= 255.0


# TensorFlowのチュートリアル参考
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_x, train_data_y, epochs=10)

# 予測を取得
prediction = model.predict(test_data)

# 各サンプルの予測クラスを取得
predicted_classes = np.argmax(prediction, axis=1)

# DataFrameに変換
output = pd.DataFrame({
    "ImageId": np.arange(1, len(predicted_classes) + 1),
    "Label": predicted_classes
})

# CSVに保存
output.to_csv('digit_recognizer_CNN1a.csv', index=False)
print("output was successfully saved!")


#TIPS

# https://chatgpt.com/share/674f5001-d878-800e-96e1-8eaa96be30fa
# 上記コードの改善点を挙げてもらった



以下に提供されたPythonコードを詳細に説明します。このコードは、**「Child Mind Institute — Problematic Internet Use」**というKaggleコンペティションにおける機械学習パイプラインの一部です。コード全体を通じて、データの前処理からモデルの学習、予測、そして結果の保存までの一連のプロセスが実装されています。

コードの概要
ライブラリのインポート
定数の設定
データの読み込みと前処理
特徴量の選択とエンコーディング
評価指標と補正関数の定義
モデルの定義とパラメータ設定
モデルのスタッキングと学習
予測と結果の保存
以下、それぞれのセクションについて詳細に説明します。

1. ライブラリのインポート
python
コードをコピーする
import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import clone
from copy import deepcopy
import optuna
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import seaborn as sns

import re
from colorama import Fore, Style

from tqdm import tqdm
from IPython.display import clear_output
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor, StackingRegressor

from sklearn.model_selection import *
from sklearn.metrics import *
説明:

数値計算とデータ操作:
numpy, pandas, polars: データの操作や数値計算に使用。
機械学習関連:
sklearn: 様々な機械学習アルゴリズムや評価指標、モデル選択のためのツール。
lightgbm, xgboost, catboost: 高性能な勾配ブースティングライブラリ。
最適化とパフォーマンス向上:
optuna, scipy.optimize: ハイパーパラメータチューニングや最適化に使用。
可視化:
matplotlib, seaborn: データの可視化。
その他:
tqdm: プログレスバーの表示。
colorama: ターミナルのテキストカラーを変更。
warnings: 警告メッセージの非表示。
ThreadPoolExecutor: 並列処理。
目的と理由:

このセクションでは、データ処理、機械学習モデルの構築、評価、最適化、そして結果の可視化に必要なライブラリをインポートしています。特に、複数の勾配ブースティングアルゴリズム（LightGBM, XGBoost, CatBoost）を使用しているため、それぞれのライブラリを明示的にインポートしています。

2. 定数の設定
python
コードをコピーする
n_splits = 5
SEED = 42
説明:

n_splits: クロスバリデーションにおける分割数（フォールド数）。ここでは5フォールドクロスバリデーションを使用。
SEED: 再現性を確保するための乱数シード値。
目的と理由:

クロスバリデーションのフォールド数を設定することで、モデルの汎化性能を評価します。シード値を設定することで、データの分割やモデルの学習過程を再現可能にします。

3. データの読み込みと前処理
python
コードをコピーする
train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')

def process_file(filename, dirname):
    df = pd.read_parquet(os.path.join(dirname, filename, 'part-0.parquet'))
    df.drop('step', axis=1, inplace=True)
    return df.describe().values.reshape(-1), filename.split('=')[1]

def load_time_series(dirname) -> pd.DataFrame:
    ids = os.listdir(dirname)
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda fname: process_file(fname, dirname), ids), total=len(ids)))
    stats, indexes = zip(*results)
    df = pd.DataFrame(stats, columns=[f"Stat_{i}" for i in range(len(stats[0]))])
    df['id'] = indexes
    return df

train_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet")
test_ts = load_time_series("/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet")

time_series_cols = train_ts.columns.tolist()
time_series_cols.remove("id")

train = pd.merge(train, train_ts, how="left", on='id')
test = pd.merge(test, test_ts, how="left", on='id')

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)
説明:

データの読み込み:

train.csv, test.csv, sample_submission.csv を読み込み。
時系列データの処理関数:

process_file: 指定されたディレクトリ内のParquetファイルを読み込み、step列を削除。その後、データフレームの統計情報（describe()）を取得し、フラットな配列として返す。また、ファイル名からIDを抽出。
load_time_series: 指定されたディレクトリ内の全ファイルに対してprocess_fileを並列処理で適用し、統計情報とIDを結合したデータフレームを作成。
時系列データの読み込みとマージ:

series_train.parquetとseries_test.parquetから時系列データを読み込み、idを基に元のtrainおよびtestデータにマージ。
マージ後、id列を削除。
目的と理由:

時系列データの統合: 主なデータセットに加え、時系列データの統計情報を特徴量として統合することで、モデルの予測性能を向上させることが期待されます。
並列処理: ThreadPoolExecutorを使用して、複数のファイルを並列に処理することで、処理速度を向上。
データの前処理: 不要な列（step, id）を削除し、統計情報のみを保持することで、データの一貫性と効率的な処理を図ります。
4. 特徴量の選択とエンコーディング
python
コードをコピーする
featuresCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',
                'CGAS-Season', 'CGAS-CGAS_Score',
                'Physical-Season', 'Physical-BMI', 'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference', 'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',
                'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',
                'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU', 'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR', 'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',
                'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW',
                'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season', 'PAQ_C-PAQ_C_Total',
                'SDS-Season', 'SDS-SDS_Total_Raw', 'SDS-SDS_Total_T', 
                'PreInt_EduHx-Season', 'PreInt_EduHx-computerinternet_hoursday',
                'sii']
featuresCols += time_series_cols

train = train[featuresCols]
train = train.dropna(subset='sii')

cat_c = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'Fitness_Endurance-Season', 
          'FGC-Season', 'BIA-Season', 'PAQ_A-Season', 'PAQ_C-Season', 'SDS-Season', 'PreInt_EduHx-Season']

def update(df):
    for c in cat_c: 
        df[c] = df[c].fillna('Missing')
        df[c] = df[c].astype('category')
    return df

train = update(train)
test = update(test)

def create_mapping(column, dataset):
    unique_values = dataset[column].unique()
    return {value: idx for idx, value in enumerate(unique_values)}

for col in cat_c:
    mapping_train = create_mapping(col, train)
    mapping_test = create_mapping(col, test)
    train[col] = train[col].replace(mapping_train).astype(int)
    test[col] = test[col].replace(mapping_test).astype(int)
説明:

特徴量の選択:

featuresCols: 多数の特徴量名がリスト化されています。これには、基本的なデモグラフィック情報、身体測定データ、フィットネス関連データ、時系列データの統計情報などが含まれています。
featuresCols += time_series_cols: 時系列データの統計情報を特徴量に追加。
データのフィルタリング:

train = train[featuresCols]: 選択された特徴量のみを含むようにデータをフィルタリング。
train = train.dropna(subset='sii'): 目的変数（sii）が欠損している行を削除。
カテゴリ特徴量の処理:

cat_c: カテゴリ列のリスト。
update 関数:
各カテゴリ列の欠損値を 'Missing' で埋める。
データ型を category に変換。
データの更新:
train と test データに対して update 関数を適用。
カテゴリ列の数値マッピング:

create_mapping 関数:
各カテゴリ列のユニークな値を整数にマッピングする辞書を作成。
マッピングの適用:
各カテゴリ列に対して、トレーニングデータとテストデータのそれぞれでマッピングを作成。
マッピングを適用してカテゴリデータを整数に変換。
目的と理由:

特徴量の選択:
モデルに有用な情報を含む特徴量のみを選択し、不要なデータを排除することで、モデルの効率と性能を向上させます。
カテゴリ特徴量のエンコーディング:
機械学習アルゴリズムは通常、数値データのみを扱うため、カテゴリデータを数値に変換する必要があります。ここではラベルエンコーディングを採用していますが、場合によってはワンホットエンコーディングやターゲットエンコーディングを検討することも有効です。
欠損値の処理:
カテゴリデータの欠損値を 'Missing' として埋めることで、モデルが欠損値の存在を学習できるようにします。数値データの欠損値は後続の処理（PCA適用時）で補完します。
5. 評価指標と補正関数の定義
python
コードをコピーする
def quadratic_weighted_kappa(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')

def threshold_Rounder(oof_non_rounded, thresholds):
    return np.where(oof_non_rounded < thresholds[0], 0,
                    np.where(oof_non_rounded < thresholds[1], 1,
                             np.where(oof_non_rounded < thresholds[2], 2, 3)))

def evaluate_predictions(thresholds, y_true, oof_non_rounded):
    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)
    return -quadratic_weighted_kappa(y_true, rounded_p)
説明:

quadratic_weighted_kappa:
機能: 実際の値 (y_true) と予測値 (y_pred) の間の二次加重コーエンカッパー（Quadratic Weighted Kappa）を計算。
用途: 多クラス分類問題における評価指標。予測の順位間の一致を評価するのに適しています。
threshold_Rounder:
機能: 連続値の予測結果 (oof_non_rounded) を指定された閾値 (thresholds) に基づいてカテゴリに丸める。
動作:
予測値が最初の閾値より小さい場合はクラス0。
予測値が最初の閾値以上で2番目の閾値より小さい場合はクラス1。
予測値が2番目の閾値以上で3番目の閾値より小さい場合はクラス2。
それ以外はクラス3。
evaluate_predictions:
機能: 閾値を用いて予測結果を丸め、その結果に基づいて負の二次加重カッパーを返す。
用途: 最適化のための評価関数。最小化問題として設定されており、実際には最大化したいカッパーの値を負の値として返します。
目的と理由:

評価指標の定義:
二次加重コーエンカッパーは、予測が実際の値とどれだけ一致しているかを測る優れた指標です。特に、クラス間の順序関係が重要な場合に適しています。
予測結果の丸め:
回帰モデルの出力をクラスに変換するために、閾値を用いて予測結果を丸めています。これにより、連続的な予測値を離散的なクラスに変換し、評価指標としてのカッパーを計算可能にします。
最適化関数の設定:
evaluate_predictions関数は、最適な閾値を見つけるための目的関数として使用されます。閾値を調整することで、予測結果のクラス割り当てを最適化し、カッパーのスコアを最大化（最小化の形で実装）します。
6. モデルの定義とパラメータ設定
6.1. モデル訓練関数 TrainML の定義
python
コードをコピーする
def TrainML(model_class, test_data):
    X = train.drop(['sii'], axis=1)
    y = train['sii']

    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True)

    train_S = []
    val_S = []

    oof_non_rounded = np.zeros(len(y), dtype=float) 
    oof_rounded = np.zeros(len(y), dtype=int) 
    test_preds = np.zeros((len(test_data), n_splits))

    for fold, (train_idx, val_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = clone(model_class)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        oof_non_rounded[val_idx] = y_val_pred
        y_val_pred_rounded = y_val_pred.round(0).astype(int)
        oof_rounded[val_idx] = y_val_pred_rounded

        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))
        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)

        train_S.append(train_kappa)
        val_S.append(val_kappa)

        test_preds[:, fold] = model.predict(test_data)
        
        clear_output(wait=True)

    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")
    print(f"Mean Validation QWK ---> {np.mean(val_S):.4f}")

    KappaOPtimizer = minimize(evaluate_predictions,
                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), 
                              method='Nelder-Mead')
    assert KappaOPtimizer.success, "Optimization did not converge."

    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)
    tKappa = quadratic_weighted_kappa(y, oof_tuned)
    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")

    tpm = test_preds.mean(axis=1)
    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)

    submission = pd.DataFrame({
        'id': sample['id'],
        'sii': tpTuned
    })
    return submission, model
説明:

データの準備:

X: 目的変数 (sii) を除いた特徴量。
y: 目的変数 (sii)。
クロスバリデーションの設定:

StratifiedKFold: クラスの分布を保ったままデータを分割。n_splits=5で5フォールドクロスバリデーション。
アウトオブフォールド予測とテストデータ予測の準備:

oof_non_rounded: 未丸めのアウトオブフォールド（OOF）予測値を格納。
oof_rounded: 丸められたOOF予測値を格納。
test_preds: 各フォールドのテストデータ予測を格納。
クロスバリデーションループ:

各フォールドごとに以下を実施：
データの分割: 訓練データと検証データ。
モデルのクローンと学習: 元のモデルをクローンし、訓練データで学習。
予測:
訓練データと検証データに対する予測。
検証データの未丸め予測値を oof_non_rounded に格納。
丸めた予測値を oof_rounded に格納。
評価:
訓練データと検証データに対する二次加重カッパー（QWK）スコアを計算。
各フォールドのスコアをリストに追加。
テストデータの予測:
各フォールドのテストデータに対する予測を test_preds に格納。
進捗表示のクリア: clear_output(wait=True) で進捗バーを更新。
クロスバリデーション後の評価:

平均スコアの表示: 訓練データと検証データの平均QWKスコアを表示。
閾値の最適化:

minimize を使用して、予測値を丸めるための最適な閾値を探索。目的はQWKスコアの最大化（最小化問題として実装）。
最適化結果の確認: 成功しなかった場合はエラーを投げる。
予測結果の調整と保存:

OOF予測の調整: 最適化された閾値を使用して、未丸めのOOF予測を丸める。
QWKスコアの計算: 調整後の予測で再度QWKスコアを計算。
テストデータの予測調整: 各フォールドのテスト予測を平均し、最適化された閾値で丸める。
提出用データの作成: submission データフレームを作成し、sii予測を保存。
目的と理由:

クロスバリデーション:
モデルの汎化性能を評価し、過学習を防ぐために使用。
アウトオブフォールド予測:
クロスバリデーション中の未見データに対する予測を収集し、全体の予測性能を評価。
閾値の最適化:
回帰モデルの予測をカテゴリに変換する際の閾値を最適化することで、QWKスコアを最大化し、提出時の予測精度を向上させる。
スタッキングとアンサンブル学習:
複数の異なるモデル（LightGBM, XGBoost, CatBoost, TabNet）の予測を組み合わせることで、個々のモデルの弱点を補完し、全体の予測性能を向上させる。
7. モデルのスタッキングと学習
7.1. TabNet用ラッパークラスの定義
python
コードをコピーする
class TabNetWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.model = TabNetRegressor(**kwargs)
        self.kwargs = kwargs
        self.imputer = KNNImputer(n_neighbors=5)
        self.best_model_path = 'best_tabnet_model.pt'

    def fit(self, X, y):
        X_imputed = self.imputer.fit_transform(X)
        if hasattr(y, 'values'):
            y = y.values
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_imputed,
            y,
            test_size=0.2,
            random_state=42
        )

        history = self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_valid, y_valid.reshape(-1, 1))],
            eval_name=['valid'],
            eval_metric=['mse', 'mae', 'rmse'],
            max_epochs=500,
            patience=50,
            batch_size=1024,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            callbacks=[
                TabNetPretrainedModelCheckpoint(
                    filepath=self.best_model_path,
                    monitor='valid_mse',
                    mode='min',
                    save_best_only=True,
                    verbose=True
                )
            ]
        )

        if os.path.exists(self.best_model_path):
            self.model.load_model(self.best_model_path)
            os.remove(self.best_model_path)

        return self

    def predict(self, X):
        X_imputed = self.imputer.transform(X)
        return self.model.predict(X_imputed).flatten()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
説明:

クラスの目的:

TabNetRegressorをスキルーンの互換性を持つ形でラップし、パイプライン内で他のモデルと同様に扱えるようにする。
__init__ メソッド:

self.model: TabNetRegressorのインスタンス。
self.imputer: KNNImputerを使用して数値データの欠損値を補完。
self.best_model_path: 最良モデルを保存するためのパス。
fit メソッド:

欠損値の補完: KNNImputerで欠損値を補完。
データの分割: train_test_splitで訓練データと検証データに分割。
モデルの学習:
TabNetRegressorを使用してモデルを学習。
コールバック: TabNetPretrainedModelCheckpointを使用して、検証データでのMSEが最小になるモデルを保存。
早期停止: patience=50により、50エポック改善がない場合に学習を停止。
最良モデルのロード:
最良モデルが保存されていればロードし、保存ファイルを削除。
predict メソッド:

欠損値の補完: テストデータに対しても同様に補完を実施。
予測: 補完後のデータに対して予測を行い、フラットな配列として返す。
__deepcopy__ メソッド:

モデルのディープコピーをサポート。これは、クローニングやアンサンブル学習時に必要。
目的と理由:

TabNetの統合:
TabNetは高性能なニューラルネットワークベースのモデルであり、スキルーンの他のモデルとアンサンブルすることで、総合的な予測性能を向上させることが期待されます。
欠損値の補完:
TabNet自体は欠損値を直接扱うことができますが、ここではKNNImputerを使用して補完しています。これは、データの一貫性を保ち、他の前処理ステップと統一性を持たせるためです。
コールバックの使用:
TabNetPretrainedModelCheckpointを使用することで、検証データでの最良モデルのみを保存し、過学習を防止。
7.2. TabNet用チェックポイントクラスの定義
python
コードをコピーする
class TabNetPretrainedModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_loss', mode='min',
                 save_best_only=True, verbose=1):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose
        self.best = float('inf') if mode == 'min' else -float('inf')

    def on_train_begin(self, logs=None):
        self.model = self.trainer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        if (self.mode == 'min' and current < self.best) or \
           (self.mode == 'max' and current > self.best):
            if self.verbose:
                print(f'\nEpoch {epoch}: {self.monitor} improved from {self.best:.4f} to {current:.4f}')
            self.best = current
            if self.save_best_only:
                self.model.save_model(self.filepath)
説明:

クラスの目的:

モデルの学習中に特定の指標（デフォルトでは検証損失）を監視し、最良のモデルのみを保存するためのコールバッククラス。
__init__ メソッド:

filepath: モデルを保存するパス。
monitor: 監視する指標（デフォルトは 'val_loss'）。
mode: 'min' または 'max'。指標が最小化または最大化されるべきかを指定。
save_best_only: 最良モデルのみを保存するかどうか。
verbose: 出力の有無。
self.best: 現在の最良指標値。
on_train_begin メソッド:

トレーナー（モデル）への参照を取得。
on_epoch_end メソッド:

各エポック終了時に呼び出され、現在の指標値を取得。
指標値が改善していれば、モデルを保存し、最良指標値を更新。
目的と理由:

モデルの保存:
学習過程で最も良いパフォーマンスを示したモデルのみを保存することで、過学習を防ぎ、最適なモデルを確保します。
監視の柔軟性:
モニタリングする指標や保存条件を柔軟に設定できるため、異なるシナリオやモデルに対応可能。
7.3. モデルパラメータの設定
python
コードをコピーする
LGBM_Params = {
    'learning_rate': 0.04,
    'max_depth': 12,
    'num_leaves': 413,
    'min_data_in_leaf': 14,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.76,
    'bagging_freq': 2,
    'lambda_l1': 4.735,
    'lambda_l2': 4.735e-06,
    'random_state': SEED}

LGBM_Model = lgb.LGBMRegressor(**LGBM_Params, verbose=-1, n_estimators=200)

XGB_Params = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5,
    'random_state': SEED}

XGB_Model = XGBRegressor(**XGB_Params)

CatBoost_Params = {
    'learning_rate': 0.05,
    'depth': 6,
    'iterations': 200,
    'random_seed': SEED,
    'cat_features': cat_c,
    'verbose': 0,
    'l2_leaf_reg': 10
}

CatBoost_Model = CatBoostRegressor(**CatBoost_Params)

TabNet_Params = {
    'n_d': 64,
    'n_a': 64,
    'n_steps': 5,
    'gamma': 1.5,
    'n_independent': 2,
    'n_shared': 2,
    'lambda_sparse': 1e-4,
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': dict(lr=2e-2, weight_decay=1e-5),
    'mask_type': 'entmax',
    'scheduler_params': dict(mode="min", patience=10, min_lr=1e-5, factor=0.5),
    'scheduler_fn': torch.optim.lr_scheduler.ReduceLROnPlateau,
    'verbose': 1,
    'device_name': 'cuda' if torch.cuda.is_available() else 'cpu'
}

TabNet_Model = TabNetWrapper(**TabNet_Params)
説明:

LightGBMのパラメータ設定とモデルの定義:

learning_rate: 学習率。0.04は適度な速度での学習を示す。
max_depth: 木の深さ。12は複雑なモデルを構築可能に。
num_leaves: 葉の数。413は大きな値で、多様な分割を可能に。
min_data_in_leaf: 葉に必要な最小データ数。14は過学習を防ぐバランスの取れた値。
feature_fraction: 各木の特徴量のサブサンプリング割合。0.8は80%の特徴量を使用。
bagging_fraction: データのサブサンプリング割合。0.76は76%のデータを使用。
bagging_freq: バギングの頻度。2は2回ごとにバギングを適用。
lambda_l1, lambda_l2: L1およびL2正則化パラメータ。過学習を防ぐための正則化。
random_state: 再現性のための乱数シード。
n_estimators: ブースティングの繰り返し回数。200は十分な学習を可能に。
XGBoostのパラメータ設定とモデルの定義:

learning_rate: 0.05。適度な学習率。
max_depth: 6。適度な複雑さ。
n_estimators: 200。
subsample: 0.8。データのサブサンプリング割合。
colsample_bytree: 0.8。特徴量のサブサンプリング割合。
reg_alpha, reg_lambda: 正則化パラメータ。
random_state: 再現性のためのシード。
CatBoostのパラメータ設定とモデルの定義:

learning_rate: 0.05。
depth: 6。ツリーの深さ。
iterations: 200。ブースティングの繰り返し回数。
random_seed: 再現性のためのシード。
cat_features: カテゴリ特徴量のリスト。
verbose: 出力を抑制。
l2_leaf_reg: L2正則化パラメータ。
TabNetのパラメータ設定とモデルの定義:

n_d, n_a: 表現の次元。64は適度な表現力を持つ。
n_steps: 学習ステップの数。5はバランスの取れた値。
gamma: 正則化パラメータ。1.5は過学習防止に寄与。
n_independent, n_shared: モデルの構造パラメータ。
lambda_sparse: スパース性の正則化。
optimizer_fn: 最適化関数。Adamを使用。
optimizer_params: 最適化パラメータ。学習率と重み減衰。
mask_type: マスクのタイプ。'entmax'はスパースなマスクを生成。
scheduler_params, scheduler_fn: 学習率スケジューラーの設定。パティエンス10、最小学習率1e-5、減衰率0.5。
verbose: 出力レベル。
device_name: GPUが利用可能なら'cuda'、そうでなければ'cpu'を使用。
目的と理由:

高性能モデルの構築:
LightGBM、XGBoost、CatBoost、TabNetといった異なるアルゴリズムを使用することで、多様な視点からデータを学習し、アンサンブルすることで総合的な性能向上を図ります。
ハイパーパラメータのチューニング:
各モデルのパラメータは、過学習を防ぎつつ、モデルの表現力を最大化するように設定されています。
アンサンブル学習:
異なるモデルを組み合わせることで、個々のモデルの弱点を補完し、より堅牢な予測を実現。
7.4. スタッキングモデルの定義と学習
python
コードをコピーする
Stack_Model = StackingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('tabnet', TabNet_Model)])

Submission, model = TrainML(Stack_Model, test)
Submission.to_csv('submission.csv', index=False)
print(Submission['sii'].value_counts())
説明:

スタッキングモデルの定義:

StackingRegressor: 複数の異なる回帰モデル（ここではLightGBM, XGBoost, CatBoost, TabNet）を基底モデル（base estimators）として組み合わせるアンサンブル手法。
estimators: 各基底モデルに名前とモデルインスタンスをペアで指定。
モデルの訓練と予測:

TrainML(Stack_Model, test): 前述のTrainML関数を使用して、スタッキングモデルを訓練し、テストデータに対する予測を行う。
Submission.to_csv: 予測結果をsubmission.csvとして保存。
print(Submission['sii'].value_counts()): 予測結果のクラス分布を表示。
目的と理由:

スタッキングによる性能向上:
複数の異なるモデルを組み合わせることで、個々のモデルの強みを活かし、全体の予測性能を向上させます。スタッキングは、ベースモデルの予測をメタモデル（この場合、スタッキングレグレッサー自体がメタモデル）で統合します。
予測結果の可視化:
クラス分布を確認することで、予測結果が適切なバランスを保っているかを確認できます。極端なバランス偏りがある場合、さらなる調整が必要かもしれません。
補足説明
polars ライブラリのインポート
python
コードをコピーする
import polars as pl
説明:

Polars: 高速でメモリ効率の良いデータフレームライブラリ。ここではインポートされていますが、コード内で使用されていないようです。
目的と理由:

パフォーマンス向上:
Polarsは大規模データセットの処理に適しています。しかし、現在のコードではPandasが主に使用されているため、Polarsのインポートは不要かもしれません。
モデルのスタッキングにおけるfinal_estimatorの未指定
python
コードをコピーする
Stack_Model = StackingRegressor(estimators=[
    ('lightgbm', LGBM_Model),
    ('xgboost', XGB_Model),
    ('catboost', CatBoost_Model),
    ('tabnet', TabNet_Model)])
説明:

final_estimator: スタッキングの最終段階で使用されるメタモデル。デフォルトではLinearRegressionが使用されますが、明示的に指定していません。
目的と理由:

デフォルト設定の利用:
スタッキングモデルのfinal_estimatorを明示的に指定していない場合、デフォルトでLinearRegressionが使用されます。これは、単純かつ解釈しやすいメタモデルとして有用です。
カスタマイズの可能性:
必要に応じて、より複雑なメタモデル（例えば、別の勾配ブースティングモデルやニューラルネットワーク）を指定することも可能です。
ライブラリのインストール
python
コードをコピーする
!pip -q install pytorch_tabnet=4.1.0
説明:

pytorch_tabnet のインストール: TabNetの実装を提供するライブラリを特定のバージョン（4.1.0）でインストールしています。
目的と理由:

依存関係の確保:
TabNetを使用するために必要なライブラリをインストールし、特定のバージョンを指定することで、互換性の問題を防止します。
コードの改善点と推奨事項
Polarsの使用:

現在、Polarsがインポートされていますが、コード内で使用されていません。不要であればインポートを削除するか、データ処理にPolarsを活用することでパフォーマンスを向上させることができます。
final_estimator の明示的な指定:

スタッキングモデルのメタモデルを明示的に指定することで、モデルの挙動をより詳細に制御できます。例えば、LinearRegressionではなくXGBRegressorを使用するなど、メタモデルの性能を調整することが可能です。
重複インポートの整理:

TabNet関連のライブラリやクラスが重複してインポートされています。これらを整理することで、コードの可読性と効率性が向上します。
欠損値の確認と処理の一貫性:

現在、数値データの欠損値はSimpleImputerで補完されていますが、TabNetWrapperクラス内でもKNNImputerを使用しています。どちらか一方に統一するか、必要に応じて異なる補完方法を適用する基準を設けると良いでしょう。
カテゴリデータのエンコーディング手法の検討:

現在はラベルエンコーディングを使用していますが、カテゴリの数が多い場合や順序がない場合、ワンホットエンコーディングやターゲットエンコーディングを検討することも有用です。
エラー処理の強化:

モデルの訓練や予測時に発生する可能性のあるエラーに対して、適切な例外処理を実装することで、より堅牢なコードとなります。
ドキュメンテーションとコメントの追加:

各セクションや重要な処理に対して、詳細なコメントやドキュメントを追加することで、コードの理解と保守が容易になります。
まとめ
このコードは、データの前処理からモデルの学習、予測、そして結果の保存までを包括的に実装した機械学習パイプラインです。以下の点が特徴的です：

多様なモデルの使用: LightGBM, XGBoost, CatBoost, TabNetを使用し、スタッキングによってアンサンブルを構築。
時系列データの統合: 時系列データの統計情報を特徴量として組み込むことで、モデルの予測精度を向上。
評価指標の最適化: 二次加重コーエンカッパーを使用し、閾値の最適化を通じて予測結果の品質を高める。
効率的なデータ処理: 並列処理を活用し、大規模データの処理速度を向上。
改善点としては、コードの整理、インポートの最適化、カテゴリエンコーディング手法の再検討、エラー処理の強化などが挙げられます。これらを考慮することで、コードの効率性と可読性、そしてモデルの予測性能をさらに向上させることが可能です。

もし他に質問や特定のセクションについての詳細な説明が必要であれば、遠慮なくお知らせください！











