import tensorflow as tf
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import math


class LocalDensityEstimator:
    def __init__(self, k=32):
        self.k = k
        self.knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')

    def compute_density(self, points):
        self.knn.fit(points)
        distances, _ = self.knn.kneighbors(points)
        d_k = distances[:, -1].reshape(-1, 1)
        d_k = np.maximum(d_k, 1e-6)
        sphere_volume = (4/3) * math.pi * (d_k ** 3)
        density = self.k / sphere_volume
        return density.astype(np.float32)


class DensityWeightMLP(tf.keras.Model):
    def __init__(self, hidden_dims=[64, 32]):
        super().__init__()
        self.layers_list = []
        input_dim = 1
        for dim in hidden_dims:
            self.layers_list.append(tf.keras.layers.Dense(dim, activation=tf.nn.relu))
            input_dim = dim
        self.layers_list.append(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    def call(self, density):
        x = density
        for layer in self.layers_list:
            x = layer(x)
        return x


class DensityAwareFeatureAggregator(tf.keras.Model):
    def __init__(self, in_feat_dim, out_feat_dim, k=32):
        super().__init__()
        self.k = k
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.pos_enc = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.relu),
            tf.keras.layers.Dense(64)
        ])
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(out_feat_dim)
        ])
        self.density_weight_mlp = DensityWeightMLP()

    def call(self, points, features, density):
        B, N, _ = points.shape
        agg_features = []
        for b in range(B):
            batch_points = points[b].numpy()
            batch_feat = features[b]
            batch_density = density[b]
            knn = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree')
            knn.fit(batch_points)
            _, neighbor_idx = knn.kneighbors(batch_points)
            neighbor_idx = tf.convert_to_tensor(neighbor_idx, dtype=tf.int32)
            neighbor_feat = tf.gather(batch_feat, neighbor_idx, axis=0)
            center_points = tf.tile(tf.expand_dims(batch_points, axis=1), [1, self.k, 1])
            rel_pos = tf.convert_to_tensor(batch_points[neighbor_idx] - center_points, dtype=tf.float32)
            pos_enc_feat = self.pos_enc(rel_pos)
            cat_feat = tf.concat([neighbor_feat, pos_enc_feat], axis=-1)
            mlp_feat = self.mlp(cat_feat)
            center_density = tf.tile(tf.expand_dims(batch_density, axis=1), [1, self.k, 1])
            neighbor_weight = self.density_weight_mlp(center_density)
            neighbor_weight = tf.nn.softmax(neighbor_weight, axis=1)
            weighted_feat = neighbor_weight * mlp_feat
            batch_agg = tf.reduce_sum(weighted_feat, axis=1)
            agg_features.append(batch_agg)
        return tf.stack(agg_features, axis=0)


class RandLA_ADW(tf.keras.Model):
    def __init__(self, in_feat_dim=3, num_classes=6, k=32):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.density_estimator = LocalDensityEstimator(k=k)
        self.encoder = [
            tf.keras.Sequential([
                DensityAwareFeatureAggregator(in_feat_dim=in_feat_dim, out_feat_dim=64, k=k),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                DensityAwareFeatureAggregator(in_feat_dim=64, out_feat_dim=128, k=k),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                DensityAwareFeatureAggregator(in_feat_dim=128, out_feat_dim=256, k=k),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                DensityAwareFeatureAggregator(in_feat_dim=256, out_feat_dim=512, k=k),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ])
        ]
        self.decoder = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(256),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Dense(128),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Dense(64),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ]),
            tf.keras.Sequential([
                tf.keras.layers.Dense(32),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation(tf.nn.relu)
            ])
        ]
        self.classifier = tf.keras.layers.Dense(num_classes)
        self.mlp_d = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation=tf.nn.relu),
            tf.keras.layers.Dense(32)
        ])

    def downsample(self, x, ratio):
        idx = tf.random.shuffle(tf.range(tf.shape(x)[1]))[:int(tf.shape(x)[1] * ratio)]
        return tf.gather(x, idx, axis=1)

    def call(self, points, training=True):
        B, N, C = points.shape
        features = points
        density = []
        for b in range(B):
            batch_density = self.density_estimator.compute_density(points[b].numpy())
            density.append(tf.convert_to_tensor(batch_density, dtype=tf.float32))
        density = tf.stack(density, axis=0)
        enc_feats = []
        for i, block in enumerate(self.encoder):
            features = block(points, features, density, training=training)
            enc_feats.append(features)
            if i < len(self.encoder) - 1:
                points = self.downsample(points, ratio=0.25)
                features = self.downsample(features, ratio=0.25)
                density = self.downsample(density, ratio=0.25)
        dec_feat = features
        for i, block in enumerate(self.decoder):
            up_ratio = 4
            if i < len(self.decoder) - 1:
                up_size = tf.shape(enc_feats[-(i+2)])[1]
            else:
                up_size = N
            dec_feat = tf.keras.layers.UpSampling1D(size=up_ratio)(dec_feat)
            dec_feat = dec_feat[:, :up_size, :]
            if i > 0:
                dec_feat = tf.concat([dec_feat, enc_feats[-(i+1)]], axis=-1)
            dec_feat = block(dec_feat, training=training)
        logits = self.classifier(dec_feat)
        return logits, dec_feat, density


class HybridLoss:
    def __init__(self, lambda_density=0.1):
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.lambda_d = lambda_density

    def __call__(self, logits, labels, final_feat, density, mlp_d):
        B, N, C = logits.shape
        ce_loss_val = self.ce_loss(tf.reshape(labels, [-1]), tf.reshape(logits, [-1, C]))
        density_feat = mlp_d(density)
        density_loss_val = self.mse_loss(final_feat, density_feat)
        total_loss = ce_loss_val + self.lambda_d * density_loss_val
        return total_loss, ce_loss_val, density_loss_val


class DynamicParamDBSCAN:
    def __init__(self, eps0=5.0, min_pts0=10, k=32):
        self.eps0 = eps0
        self.min_pts0 = min_pts0
        self.density_estimator = LocalDensityEstimator(k=k)

    def compute_global_mean_density(self, points):
        density = self.density_estimator.compute_density(points)
        return np.mean(density)

    def cluster(self, points):
        density = self.density_estimator.compute_density(points)
        rho_mean = self.compute_global_mean_density(points)
        eps = np.where(
            density < rho_mean,
            self.eps0 * (rho_mean / density),
            self.eps0 * (rho_mean / density)
        ).mean()
        min_pts = np.where(
            density < rho_mean,
            self.min_pts0 * (density / rho_mean),
            self.min_pts0 * (density / rho_mean)
        ).mean().astype(int)
        dbscan = DBSCAN(eps=eps, min_samples=min_pts, metric='euclidean')
        labels = dbscan.fit_predict(points)
        return labels, eps, min_pts


def generate_synthetic_sum_data(B=2, N=4096):
    points = np.random.uniform(0, 100, size=(B, N, 3)).astype(np.float32)
    labels = np.random.randint(0, 6, size=(B, N)).astype(np.int64)
    points = tf.convert_to_tensor(points)
    labels = tf.convert_to_tensor(labels)
    return points, labels


def train_randla_adw():
    epochs = 50
    batch_size = 2
    lr = 0.01
    num_classes = 6
    lambda_density = 0.1
    train_points, train_labels = generate_synthetic_sum_data(B=100, N=4096)
    dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels)).batch(batch_size).shuffle(100)
    model = RandLA_ADW(in_feat_dim=3, num_classes=num_classes, k=32)
    loss_fn = HybridLoss(lambda_density=lambda_density)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    lr_scheduler = tf.keras.optimizers.schedules.StepDecay(
        initial_learning_rate=lr,
        step_size=10 * len(dataset),
        decay_rate=0.5
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)

    for epoch in range(epochs):
        total_loss = 0.0
        ce_total = 0.0
        density_total = 0.0
        step = 0
        for batch_points, batch_labels in dataset:
            with tf.GradientTape() as tape:
                logits, final_feat, density = model(batch_points, training=True)
                loss, ce_loss, density_loss = loss_fn(logits, batch_labels, final_feat, density, model.mlp_d)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy() * batch_size
            ce_total += ce_loss.numpy()
            density_total += density_loss.numpy()
            step += 1
        avg_loss = total_loss / (len(dataset) * batch_size)
        avg_ce = ce_total / step
        avg_density = density_total / step
        print(f"Epoch [{epoch+1}/{epochs}], Avg Loss: {avg_loss:.4f}, CE Loss: {avg_ce:.4f}, Density Loss: {avg_density:.4f}")
    model.save_weights("randla_adw_weights.h5")
    print("Model weights saved as randla_adw_weights.h5")
    return model


def detect_objects_with_dynamic_dbscan(model, test_points):
    model.trainable = False
    dynamic_dbscan = DynamicParamDBSCAN(eps0=5.0, min_pts0=10, k=32)
    object_labels = []
    B = test_points.shape[0]
    for b in range(B):
        batch_points = test_points[b:b+1]
        logits, _, _ = model(batch_points, training=False)
        sem_labels = tf.argmax(logits, axis=-1).numpy().squeeze(0)
        target_mask = np.isin(sem_labels, [4, 5])
        target_points = batch_points.numpy()[0][target_mask]
        if target_points.shape[0] < 10:
            batch_object_labels = np.full(sem_labels.shape, -1)
        else:
            cluster_labels, _, _ = dynamic_dbscan.cluster(target_points)
            batch_object_labels = np.full(sem_labels.shape, -1)
            batch_object_labels[target_mask] = cluster_labels + 1
        object_labels.append(batch_object_labels)
    return np.stack(object_labels, axis=0)

