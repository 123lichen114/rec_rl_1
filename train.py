#Dependencies
import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time
from tensorflowonspark import TFCluster
from pyspark.sql import SparkSession

from envs import OfflineEnv
from recommender import DRRAgent

import os
os.environ["PYSPARK_PYTHON"] = "/data2/lc/my_envs/rec_rl_1/bin/python3.8"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/data2/lc/my_envs/rec_rl_1/bin/python3.8"
ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'ml-1m')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10

def train_fn(argv,ctx):
    # 初始化环境
    print('Data loading...')
    #Loading datasets
    ratings_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'ratings.dat'), 'r').readlines()]
    users_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'users.dat'), 'r').readlines()]
    movies_list = [i.strip().split("::") for i in open(os.path.join(DATA_DIR,'movies.dat'),encoding='latin-1').readlines()]
    print(f"list ok")
    ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])
    movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # 电影 id 到电影标题的映射
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list}
    ratings_df = ratings_df.applymap(int)

    # 按用户整理看过的电影
    users_dict = np.load('./data/user_dict.npy', allow_pickle=True)

    # 每个用户的电影历史长度
    users_history_lens = np.load('./data/users_histroy_len.npy')

    users_num = max(ratings_df["UserID"])+1
    items_num = max(ratings_df["MovieID"])+1

    # 训练设置
    train_users_num = int(users_num * 0.8)
    train_items_num = items_num
    train_users_dict = {k:users_dict.item().get(k) for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]

    print('DONE!')
    time.sleep(2)

    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    recommender = DRRAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    recommender.actor.build_networks()
    recommender.critic.build_networks()
    recommender.train(MAX_EPISODE_NUM, load_model=False)

if __name__ == "__main__":
    # 创建 SparkSession
    spark = SparkSession.builder \
        .appName("DistributedDRR") \
        .getOrCreate()

    # 从 SparkSession 中获取 SparkContext
    sc = spark.sparkContext

    # 配置 TensorFlow on Spark 集群
    tf_args = []  # 定义 tf_args 参数
    cluster = TFCluster.run(sc, train_fn, tf_args, num_executors=4, num_ps=1, tensorboard=False)
    cluster.shutdown()
    spark.stop()