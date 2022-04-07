import logging

from urllib.parse import urlparse
from typing import List, Tuple

from pandas import Categorical, get_dummies, read_csv, DataFrame, concat
from s3fs import S3FileSystem
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from fastparquet import ParquetFile
import joblib
from os import environ
import os
from sqlalchemy.engine import create_engine
import json
import yaml


logger = logging.getLogger(__name__)


def is_s3_uri(target_uri):
    """Are we using s3 or local file"""
    parsed = urlparse(target_uri)
    return parsed.scheme == "s3"


def download_from_s3(local_path, source_uri):
    s3 = S3FileSystem(s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    s3.get(source_uri, local_path)


def upload_to_s3(local_path, target_uri):
    s3 = S3FileSystem(s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    s3.put(local_path, target_uri)


def s3_split(s3_path):
    return s3_path.split('/')[-1]


def s3_feed_dir(client, feed, version, file_type="parquet", ending_slash=True,
                base_dir='processed-feed', bucket="qh-clinicaldata-phi"):
    fdir = "s3://{bucket}/{base_dir}/{client}/{version}/".format(
        bucket=bucket, base_dir=base_dir, client=client, version=version
    )

    if file_type:
        fdir += (file_type + "/")
    fdir += feed

    if ending_slash:
        fdir += "/"
    logger.debug("S3 Feed Dir: {}".format(fdir))
    return fdir


def s3_model_dir(client, version, bucket="qh-clinicaldata-phi"):
    mdir = "s3://{bucket}/model-training/{client}/{version}/".format(
        bucket=bucket, client=client, version=version
    )

    logger.debug("S3 Model Dir: {}".format(mdir))
    return mdir


def get_latest_build(client, filename="sample_build_date.txt", manifest_type="sample", src="processed-feed"):
    if manifest_type == "sample":
        s3_path = "s3://qh-clinicaldata-qa-phi/sample/"\
                  + src + "/"\
                  + client\
                  + "/_latest_version"
    elif manifest_type == "prod":
        s3_path = "s3://qh-clinicaldata-phi/"\
                  + src + "/"\
                  + client\
                  + "/_latest_version"
    else:
        raise TypeError("specify the right manifest type lol no pun intended")
    download_from_s3(filename, s3_path)
    with open(filename, "r") as f:
        first_line = f.readline().strip()
    logger.debug("Latest Version: {}".format(first_line))
    return first_line


def get_final_feed_version(version, payer):
    if version == "_latest_version":
        final_version = get_latest_build(
            payer,
            filename="_latest_version",
            manifest_type="prod",
            src="processed-feed"
        )
    else:
        final_version = version
    return final_version


def get_final_model_version(version, payer):
    if version == "_latest_version":
        final_version = get_latest_build(
            payer,
            filename="_latest_version",
            manifest_type="prod",
            src="model-training"
        )
    else:
        final_version = version
    return final_version


def sparsity_ratio(mat):
    """Calculate how much sparsity we have in a given matrix

    :param mat: pandas dataframe
    :return sparsity_ratio: float
    """
    return 1.0 - np.count_nonzero(mat) / float(mat.shape[0] * mat.shape[1])


def sparsify(dataframe):
    """Returns a sparse matrix from the given df if enough sparsity detected

    :param dataframe: pandas dataframe (after prepare_data and clearing nulls)
    :return X: a sparse dataframe if meet sparse levels else
    """
    if sparsity_ratio(dataframe) >= 0.9:
        from scipy.sparse import csc_matrix
        return csc_matrix(dataframe)
    else:
        return dataframe


def make_training_data(df, response, drop_cols=None, n_lower=None, n_upper=None):
    """
    Generates data in a format convenient for model_training models with sklearn.
    Drop features bad for modeling like member_id and splits by X, y format.

    Note: if you want to set bounds on max and min num rows of incoming df, n_lower
    and n_upper must be specified as ints.

    :param df: dataframe with model_training data
    :param response: string name of response column
    :param drop_cols: list of strings indicating names of columns to drop
    :param n_lower: lower bound for number of rows of df
    :param n_upper: upper bound for number of rows of df
    :return: X, y: tuple model_training dataframe and response array
    """
    if (n_lower is not None) and (n_upper is not None):
        df = df_size_checker(df, n_lower=n_lower, n_upper=n_upper)
    response_array = df[response]
    if drop_cols is None:
        drop_cols = []
    if response not in drop_cols:
        drop_cols.append(response)
    df = df.drop(drop_cols, axis=1)
    return df, response_array


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns from df"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.columns]


class StandardScaler(BaseEstimator, TransformerMixin):
    """Scale features"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        _, numerical_cols = get_column_names_by_type(X)
        self.columns = numerical_cols
        self.mus = X[self.columns].mean()
        self.thetas = X[self.columns].std(ddof=0)
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.columns] = X[self.columns].sub(self.mus).div(self.thetas).replace((np.inf, -np.inf, np.nan), 0)
        return X


class ColumnImputer(BaseEstimator, TransformerMixin):
    """Performs imputation for numerical and categoricals"""
    def __init__(self):
        pass

    def fit(self, X, y=None):
        categorical_cols, numerical_cols = get_column_names_by_type(X)
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.numerical_fills = X[self.numerical_cols].mean()
        self.categorical_fills = "missing"
        return self

    def transform(self, X, y=None):
        X = X.copy()
        X[self.numerical_cols] = X[self.numerical_cols].fillna(self.numerical_fills)
        X[self.categorical_cols] = X[self.categorical_cols].fillna(self.categorical_fills)
        return X


class CategoricalSelector(BaseEstimator, TransformerMixin):
    """One hot encodes"""
    def __init__(self, drop_first=True, sparse=False):
        self.drop_first = drop_first
        self.sparse = sparse

    def fit(self, X, y=None):
        categorical_cols, _ = get_column_names_by_type(X)
        self.attribute_names = categorical_cols
        cats = {}
        for column in self.attribute_names:
            cats[column] = X[column].unique().tolist()
        self.categoricals = cats
        return self

    def transform(self, X, y=None):
        df = X.copy()
        for column in self.attribute_names:
            df[column] = Categorical(df[column], categories=self.categoricals[column])
        new_df = get_dummies(df, sparse=self.sparse, drop_first=self.drop_first)
        self.columns = new_df.columns
        return new_df


class RemoveZeroValueFeatures(BaseEstimator, TransformerMixin):
    """Remove features that have no variance ie entire column is same. Allows a threshold for max sparsity."""
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        to_drop = []
        nrows = len(X)
        for column in X.columns:
            if (X[column].value_counts() / nrows).iloc[0] > self.threshold:
                to_drop.append(column)
        self.to_drop = to_drop
        return self

    def transform(self, X, y=None):
        df = X.copy()
        return df.drop(self.to_drop, axis=1)


def parquet_dataframe(path, columns=None):
    """Create dataframe from s3 directory of parquet
    Note: path must end in /*.parquet to capture full dir"""
    s3 = S3FileSystem(s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    all_paths_from_s3 = s3.glob(path + "*.parquet")
    pf = ParquetFile(all_paths_from_s3, open_with=s3.open)
    return pf.to_pandas(columns=columns)


def parquet_file(path, columns=None):
    """Create dataframe from s3 directory of parquet
    Note: path must end in /*.parquet to capture full dir"""
    s3 = S3FileSystem(s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    all_paths_from_s3 = s3.glob(path + "*.parquet")
    pf = ParquetFile(all_paths_from_s3, open_with=s3.open)
    return pf


def save_data_qa(data, path, name):
    describe = data.describe(include="all").T
    save_dataframe_csv(describe, path, name, True)


def get_column_names_by_type(dataframe):
    types = dataframe.dtypes
    categoricals = types.loc[types == "object"].index.tolist()
    numericals = types.loc[types != "object"].index.tolist()
    return categoricals, numericals


def training_data_import(input_path, output_path, response_col_name, train_task, n_lower=None, n_upper=None):
    raw_training_data = parquet_dataframe(input_path)
    save_data_qa(raw_training_data, output_path, train_task + "_training_data_qa.csv")
    return make_training_data(raw_training_data, response_col_name, drop_cols=None, n_lower=n_lower, n_upper=n_upper)


def save_pickled_model(estimator, path, name="model.sav"):
    output_path = path + name
    if is_s3_uri(path):
        joblib.dump(estimator, name)
        upload_to_s3(name, output_path)
    else:
        joblib.dump(estimator, output_path)


def load_pickled_model(path):
    if is_s3_uri(path):
        name = s3_split(path)
        download_from_s3(name, path)
        best_estimator = joblib.load(name)
    else:
        best_estimator = joblib.load(path)
    return best_estimator


def save_dataframe_csv(frame, path, name="", index=False):
    output_path = path + name
    if is_s3_uri(path):
        frame.to_csv(name, index=index)
        upload_to_s3(name, output_path)
    else:
        frame.to_csv(output_path, index=index)


def import_data(path, columns=None):
    if is_s3_uri(path):
        data = parquet_dataframe(path, columns)
    else:
        data = read_csv(path)
    return data


def prediction_data_import(input_path, index_col=None, columns=None):
    prediction_data = import_data(input_path, columns)
    if index_col is not None:
        prediction_data.set_index(index_col, inplace=True)
    return prediction_data


def save_redshift(frame, path):
    schema, table = path.split(".")
    conn = create_engine(environ["DB_CONNECTION_STRING"])
    frame.to_sql(table, conn, schema=schema, if_exists="replace", index=False)


def make_dataframe_db_schema(col_data_list: List[Tuple[str, str, bool]], tbl_name, index=None, primary_key=None):
    """define the schema for writing a df to an arbitrary db

    :param col_data_list: list-like in the format ["col_name", "col_type", nullable] where "col_name" is string
    indicating the name of the column, "col_type" is string in ["int", "float", "string", "date"] indicating type of
    column and nullable is boolean indicating if column is nullable in the db
    :param tbl_name: string indicating name of table in db, should match the name of respective csv file
    :param index: list-like of strings indicating columns to index, optional
    :param primary_key: list-like of strings indicating columns to be primary key, optional
    """
    dict_names = ("name", "type", "nullable")
    df_schema = [dict(zip(dict_names, column)) for column in col_data_list]
    manifest_schema = dict(schema=df_schema, table_name=tbl_name)
    if index is not None:
        manifest_schema["index"] = index
    if primary_key is not None:
        manifest_schema["primary_key"] = primary_key
    return manifest_schema


def save_manifest_s3(manifest, path):
    filename = manifest["table_name"] + ".schema"
    save_json_s3(manifest, filename, path)


def save_json_s3(json_obj, filename, path):
    output_path = path + filename
    if is_s3_uri(path):
        with open(filename, "w") as outfile:
            json.dump(json_obj, outfile)
        upload_to_s3(filename, output_path)
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(output_path, "w") as outfile:
            json.dump(json_obj, outfile)


def payer_to_redshift_customer(payer):
    return {
        "highmark": "Highmark Blue Shield",
        "premera": "Premera",
        "horizon": "Horizon",
    }[payer]


def row_count_checker(n_rows, n_lower, n_upper):
    """Check the num rows and enforce boundaries since no one likes surprises and return if we need to downsample"""
    need_sample = False
    if n_rows < n_lower:
        raise ValueError("MikeWiLLMadeIt but we only have {} rows".format(n_rows))
    elif n_rows > n_upper:
        need_sample = True
    return need_sample


def df_size_checker(frame, n_lower, n_upper):
    """Check the dataframe shape and enforce boundaries since no one likes surprises"""
    n_rows, _ = frame.shape
    if n_rows < n_lower:
        raise ValueError("MikeWiLLMadeIt but we only have {} rows".format(n_rows))
    elif n_rows > n_upper:
        frame = frame.sample(n=n_upper, random_state=42)
    return frame


def get_qa_yaml(path, filename):
    """Get column names from manifest"""
    if is_s3_uri(path):
        download_from_s3(filename, path)
    d = yaml.load(open(filename))
    return DataFrame(d["summary"]).T


def get_col_names_from_manifest(path, filename="manifest.json") -> List:
    """Get column names from manifest"""
    if is_s3_uri(path):
        download_from_s3(filename, path)
    json_file = open(filename)
    json_str = json_file.read()
    json_dict = json.loads(json_str)
    return [key["name"] for key in json_dict["model"]["fields"]]


def get_list_s3_files(s3_path) -> List:
    """Get list of objects in s3 bucket at s3_path"""
    s3 = S3FileSystem(s3_additional_kwargs=dict(ServerSideEncryption="AES256"))
    return s3.ls(s3_path)


def get_sampling_fraction(nrows, desired_observations):
    """Get the fraction of rows needed to sample in order to have desired_observations count

    :param nrows: int representing total num rows
    :param desired_observations: int representing number of observations we want to sample
    :return fraction: float representing fraction of rows to sample
    """
    return min(desired_observations / nrows, 1)


def check_sample_needed(num_rows, n_lower, n_upper):
    """Check if we need to sample from our dataset. If too large, we'll need to downsample if too small raise

    :param num_rows: int representing num rows in dataset
    :param n_lower: int representing lower bound allowed
    :param n_upper: int representing upper bound allowed
    :return sample_needed: bool representing whether we need to sample
    """
    sample_needed = False
    if num_rows < n_lower:
        raise ValueError("MikeWiLLMadeIt but we only have {} rows".format(num_rows))
    elif num_rows > n_upper:
        sample_needed = True
    return sample_needed


def make_sample_df(df, fraction, index=None):
    """Make a sample dataframe"""
    if index is not None:
        df = df.set_index(index)
    return df.sample(frac=fraction, random_state=42)


def get_training_data(data: ParquetFile, min_rows, max_rows, index=None):
    sample_needed = check_sample_needed(data.count, min_rows, max_rows)
    if sample_needed:
        logger.info("Sampling needed! Iterating...")
        sample_fraction = get_sampling_fraction(data.count, max_rows)
        data_list = []
        for df in data.iter_row_groups():
            df = make_sample_df(df, sample_fraction, index)
            data_list.append(df)
        data = concat(data_list)
    else:
        data = data.to_pandas(index=index)
    return data