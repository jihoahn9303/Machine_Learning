import argparse
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import rf_trainer
from base_data import query
from utils import init_config, split_train_and_test_period

parser = argparse.ArgumentParser()

parser.add_argument("--dev_env", help="Development Env [local], [development], [production]", type=str, default="local")
parser.add_argument("--mode", help="[train], [predict]", type=str, default='train')

flag = parser.parse_args()

# init config(개발 환경, train, predict 등)
config = init_config(flag.dev_env)
print(config)
model_dir = f"{config['save_folder']}/models/"

# Feature Engineering(using BigQuery)
print('load data')
base_df = pd.read_gbq(query=query, dialect='standard', project_id=config['project'])

# Data Preprocessing(Label Encoding)
zip_code_le = LabelEncoder()
base_df['zip_code_le'] = zip_code_le.fit_transform(base_df['zip_code'])

# Split Train and Test Data
# 현재는 dataframe 업데이트가 이루어지지 않고 있으므로, 고정값을 기준으로 Train / Test를 나눔
# 실제 production 환경을 고려할 때 이 부분이 동적으로 설정되야 함
# 방법 예시 : 현재 일자 기준 최근 1주를 test set으로 분리
train_df, test_df = split_train_and_test_period(base_df, 7)
print('data split end')

# 불필요한 컬럼 제거
del train_df['zip_code']
del train_df['pickup_hour']
del test_df['zip_code']
del test_df['pickup_hour']

y_train_raw = train_df.pop('cnt')
y_test_raw = test_df.pop('cnt')

# backfill method를 이용하여, NaN 값을 미래 시점의 수요 값으로 채우기
train_df = train_df.fillna(method='backfill')
test_df = test_df.fillna(method='backfill')

x_train = train_df.copy()
x_test = test_df.copy()

if __name__ == '__main__':
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if flag.mode == 'train':
        print('train start')
        train_op = rf_trainer.Trainer(config)
        train_op.train(x_train, y_train_raw)

    elif flag.mode == 'predict':
        print('predict start')
        train_op = rf_trainer.Trainer(config)
        train_op.predict(x_test, y_test_raw)
    else:
        raise KeyError(f"Incorrect value flag.mode = {flag.mode}")
