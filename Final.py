import argparse
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import os
import random
from collections import defaultdict
import sys
from tqdm import tqdm
from sklearn.decomposition import NMF

def client_compute_gradients(uid, user_emb, item_emb, model_layer, H, train_data_per_user, loss_fn, clip,
                             pseudo_item_count, num_items, delt, epochs):
    user_interactions = train_data_per_user[uid]
    if not user_interactions:
        return None
    real_interaction_count = len(user_interactions)
    sigmod = (clip * math.sqrt(2 * math.log(1.25 / delt, math.e)) / ((epochs + 1) * 100))
    sigmod = sigmod / np.sqrt(real_interaction_count)

    real_items = set(x[0] for x in user_interactions)
    candidate_items = list(set(range(num_items)) - real_items)

    if len(candidate_items) >= pseudo_item_count:
        pseudo_items = random.sample(candidate_items, pseudo_item_count)
    else:
        pseudo_items = random.choices(candidate_items, k=pseudo_item_count)

    updated_user_emb = model_layer(user_emb, H)
    pseudo_preds = predict(updated_user_emb, item_emb, [uid] * pseudo_item_count, pseudo_items)
    pseudo_preds = tf.clip_by_value(pseudo_preds, 1.0, 5.0)

    all_i_idx = [x[0] for x in user_interactions] + pseudo_items
    all_ratings = [x[1] for x in user_interactions] + pseudo_preds.numpy().tolist()

    i_idx = tf.convert_to_tensor(all_i_idx, dtype=tf.int32)
    ratings = tf.convert_to_tensor(all_ratings, dtype=tf.float32)

    with tf.GradientTape() as tape:
        updated_user_emb = model_layer(user_emb, H)
        pred = predict(updated_user_emb, item_emb, [uid] * len(i_idx), i_idx)
        loss = loss_fn(ratings, pred)

    real_grads = tape.gradient(loss, [user_emb, item_emb] + model_layer.trainable_variables)

    clipped_grads = []
    for grad in real_grads:
        if grad is not None:
            clipped_grad = tf.clip_by_value(grad, -clip, clip)
            clipped_grads.append(clipped_grad)
        else:
            clipped_grads.append(None)

    if clipped_grads[1] is not None:
        noise = tf.random.normal(shape=tf.shape(clipped_grads[1]), mean=0.0, stddev=sigmod)
        clipped_grads[1] += noise

    preds = predict(updated_user_emb, item_emb, [uid] * real_interaction_count,
                    tf.convert_to_tensor([x[0] for x in user_interactions]))
    labels = tf.convert_to_tensor([x[1] for x in user_interactions], dtype=tf.float32)
    accuracy = 1.0 - tf.reduce_mean(tf.abs(preds - labels)) / 4.0

    return clipped_grads, real_interaction_count, accuracy.numpy()


class Server:
    def __init__(self, num_users, num_items, embedding_dim, lr, participation_rate, epochs, local_epochs,
                 clip, pseudo_item_count, batch_size, delt, extra_epoch, extra_epoch_batch, mf_embeddings=None):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.participation_rate = participation_rate
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.clip = clip
        self.pseudo_item_count = pseudo_item_count
        self.batch_size = batch_size
        self.delt = delt
        self.extra_epoch = extra_epoch
        self.extra_epoch_batch = extra_epoch_batch

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        if mf_embeddings:
            self.user_embedding = tf.Variable(mf_embeddings[0], name="user_embedding", dtype=tf.float32)
            self.item_embedding = tf.Variable(mf_embeddings[1], name="item_embedding", dtype=tf.float32)
        else:
            self.user_embedding = tf.Variable(tf.random.normal([num_users, embedding_dim]), name="user_embedding")
            self.item_embedding = tf.Variable(tf.random.normal([num_items, embedding_dim]), name="item_embedding")

        self.hgc_layer = HyperGraphConv(output_dim=embedding_dim)

    def aggregate_gradients(self, gradients_with_counts):
        gradients_list = [x[0] for x in gradients_with_counts]
        weights = [x[1] * x[2] for x in gradients_with_counts]
        total_weight = sum(weights)

        agg_grads = []
        for grads in zip(*gradients_list):
            weighted = []
            for g, w in zip(grads, weights):
                if isinstance(g, tf.IndexedSlices):
                    g = tf.convert_to_tensor(g)
                weighted.append(g * w)
            agg_grads.append(tf.add_n(weighted) / total_weight)
        return agg_grads

    def train(self, H, train_data_per_user, df_val):
        val_users = df_val[0].values
        val_items = df_val[1].values
        val_ratings = df_val[2].values.astype(np.float32)

        for epoch in range(self.epochs):
            selected_users = random.sample(range(self.num_users),
                                           int(self.participation_rate * self.num_users))
            for i in tqdm(range(0, len(selected_users), self.batch_size), desc=f"Epoch {epoch + 1}/{self.epochs}",
                          unit="batch"):
                batch_users = selected_users[i:i + self.batch_size]
                gradients_all = []

                for uid in batch_users:
                    grads, count, acc = client_compute_gradients(uid, self.user_embedding, self.item_embedding,
                                                                 self.hgc_layer, H, train_data_per_user,
                                                                 self.mse_loss, self.clip,
                                                                 self.pseudo_item_count, self.num_items, self.delt,
                                                                 epoch)
                    if grads:
                        gradients_all.append((grads, count, acc))

                if gradients_all:
                    agg_grads = self.aggregate_gradients(gradients_all)
                    self.optimizer.apply_gradients(
                        zip(agg_grads, [self.user_embedding, self.item_embedding] + self.hgc_layer.trainable_variables)
                    )

            updated_user_emb = self.hgc_layer(self.user_embedding, H)
            val_preds = predict(updated_user_emb, self.item_embedding, val_users, val_items)
            val_rmse = tf.sqrt(self.mse_loss(val_ratings, val_preds))
            print(f"Federated Epoch {epoch + 1}, Val RMSE: {val_rmse:.4f}")

    def test(self, H, df_test):
        test_users = df_test[0].values
        test_items = df_test[1].values
        test_ratings = df_test[2].values.astype(np.float32)

        final_user_emb = self.hgc_layer(self.user_embedding, H)
        test_preds = predict(final_user_emb, self.item_embedding, test_users, test_items)
        test_rmse = tf.sqrt(self.mse_loss(test_ratings, test_preds))
        print(f"\nFinal Test RMSE: {test_rmse:.4f}")

def build_hypergraph_sparse(df, num_users, num_items):
    rows, cols = [], []
    for uid, user_df in df.groupby(0):
        items = user_df[1].values
        for item in items:
            rows.append(uid)
            cols.append(item)
    data = np.ones(len(rows), dtype=np.float32)
    indices = np.vstack((rows, cols)).T
    shape = (num_users, num_items)
    sparse_tensor = tf.sparse.SparseTensor(indices=indices, values=data, dense_shape=shape)
    return tf.sparse.reorder(sparse_tensor)


def preprocess_data(df_train, df_val, df_test):
    df_all = pd.concat([df_train, df_val, df_test])
    user_ids = df_all[0].unique()
    item_ids = df_all[1].unique()
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: j for j, i in enumerate(item_ids)}

    df_train[0] = df_train[0].map(user2idx)
    df_train[1] = df_train[1].map(item2idx)
    df_val[0] = df_val[0].map(user2idx)
    df_val[1] = df_val[1].map(item2idx)
    df_test[0] = df_test[0].map(user2idx)
    df_test[1] = df_test[1].map(item2idx)

    num_users = len(user2idx)
    num_items = len(item2idx)

    return df_train, df_val, df_test, num_users, num_items, user2idx, item2idx

def main():
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--participation_rate', type=float, default=0.1)
    parser.add_argument('--log_file', type=str, default='default.txt', help='log_file')
    parser.add_argument('--clip', type=float, default=0.3, help='clip')
    parser.add_argument('--pseudo_item_count', type=int, default=0, help='pseudo_item_count')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--extra_epoch', type=int, default=20, help='extra_epoch')
    parser.add_argument('--extra_epoch_batch', type=int, default=128, help='extra_epoch_batch')
    parser.add_argument('--delt', type=float, default=0.001, help='delt')
    args = parser.parse_args()

    setup_logger(args.log_file)

    current_path = "../Datasets/"
    df_train = pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/train.tsv"), sep='\t', header=None)
    df_val = pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/val.tsv"), sep='\t', header=None)
    df_test = pd.read_csv(os.path.join(current_path, "./ML-100K/tsv_data/test.tsv"), sep='\t', header=None)

    df_train, df_val, df_test, num_users, num_items, user2idx, item2idx = preprocess_data(df_train, df_val, df_test)

    user_embeddings, item_embeddings = init_mf_embeddings(df_train, num_users, num_items, args.embedding_dim)

    H = build_hypergraph_sparse(df_train, num_users, num_items)

    train_data_per_user = defaultdict(list)
    for row in df_train.itertuples(index=False):
        train_data_per_user[row[0]].append((row[1], row[2]))

    server = Server(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        participation_rate=args.participation_rate,
        epochs=args.epochs,
        local_epochs=args.local_epochs,
        clip=args.clip,
        pseudo_item_count=args.pseudo_item_count,
        batch_size=args.batch_size,
        delt=args.delt,
        extra_epoch=args.extra_epoch,
        extra_epoch_batch=args.extra_epoch_batch,
        mf_embeddings=(user_embeddings, item_embeddings)
    )

    server.train(H, train_data_per_user, df_val)
    server.test(H, df_test)