import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def transform_data(data, target_var, scale=True):
    if scale:
        std_scaler = StandardScaler()
        X = pd.DataFrame(
        std_scaler.fit_transform(data.drop(target_var, axis=1)), columns=data.drop(target_var, axis=1).columns
        )
        y = data["charges"].reset_index().drop("index", axis=1)
    else:
        X = data.drop(target_var, axis=1)
        y = data["charges"].reset_index().drop("index", axis=1)
    return X, y


def load_insurance_data():
    insurance_df = pd.read_csv("../../Datasets/insurance.csv")
    categorical_cols = ["sex", "smoker", "region"]

    for col in categorical_cols:
        insurance_df[col] = insurance_df[col].astype('category')

    insurance_df = pd.get_dummies(insurance_df, columns = categorical_cols)
    # for features with only two possible values we don't need both hot encoding columns so i drop them
    insurance_df = insurance_df.drop(["sex_male", "smoker_yes"], axis=1)

    # train test split
    data_training, data_testing = train_test_split(insurance_df, test_size=0.2, random_state=234)

    insurance_X_train, insurance_y_train = transform_data(data_training, "charges")
    insurance_X_test, insurance_y_test = transform_data(data_testing, "charges")

    return insurance_X_train, insurance_y_train, insurance_X_test, insurance_y_test

# display(insurance_X_train.head())
# display(insurance_y_train.head())
