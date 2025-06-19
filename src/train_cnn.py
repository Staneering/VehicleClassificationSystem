# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.utils.class_weight import compute_class_weight

# # Set directories
# TRAIN_DIR = '/teamspace/studios/this_studio/car_model_detection/data/processed/train'
# VAL_DIR = '/teamspace/studios/this_studio/car_model_detection/data/processed/val'
# MODEL_PATH = '/teamspace/studios/this_studio/car_model_detection/models/car_model_efficientnetb0.keras'

# # Parameters
# IMG_SIZE = (224, 224)
# BATCH_SIZE = 32
# EPOCHS = 10

# # Set number of classes
# NUM_CLASSES = len(os.listdir(TRAIN_DIR))

# # Augmentation Strategy
# class BalancedDataGen(ImageDataGenerator):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

# train_datagen = BalancedDataGen(
#     rescale=1./255,
#     rotation_range=30,
#     zoom_range=0.3,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     brightness_range=[0.5, 1.5],
#     shear_range=0.15,
#     channel_shift_range=50.0,
#     horizontal_flip=True,
#     fill_mode='constant',
#     preprocessing_function=lambda x: x + tf.random.normal(x.shape, stddev=0.1)
# )

# val_datagen = BalancedDataGen(rescale=1./255)

# # Data Generators
# train_gen = train_datagen.flow_from_directory(
#     TRAIN_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=True,
#     seed=42
# )

# val_gen = val_datagen.flow_from_directory(
#     VAL_DIR,
#     target_size=IMG_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     shuffle=False
# )

# # Compute class weights
# class_weights = compute_class_weight(
#     class_weight='balanced',
#     classes=np.unique(train_gen.classes),
#     y=train_gen.classes
# )
# class_weights = dict(enumerate(class_weights))

# # Load EfficientNetB0
# base_model = tf.keras.applications.EfficientNetB0(
#     input_shape=IMG_SIZE + (3,),
#     include_top=False,
#     weights='imagenet',
#     pooling='avg'
# )
# base_model.trainable = False

# # Build Model
# model = models.Sequential([
#     base_model,
#     layers.BatchNormalization(),
#     layers.Dropout(0.5),
#     layers.Dense(512, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dropout(0.4),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

# # Compile with AdamW optimizer
# optimizer = tf.keras.optimizers.AdamW(
#     learning_rate=0.001,
#     weight_decay=0.002,
#     global_clipnorm=1.0
# )

# model.compile(
#     optimizer=optimizer,
#     loss='categorical_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.Precision(name='precision'),
#         tf.keras.metrics.Recall(name='recall'),
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )

# # Callbacks
# callbacks = [
#     tf.keras.callbacks.ModelCheckpoint(
#         MODEL_PATH,
#         save_best_only=True,
#         monitor='val_auc',
#         mode='max'
#     ),
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_recall',
#         patience=8,
#         min_delta=0.01,
#         restore_best_weights=True
#     ),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss',
#         factor=0.5,
#         patience=3,
#         cooldown=1,
#         min_lr=1e-6
#     ),
#     tf.keras.callbacks.CSVLogger('training_log.csv'),
#     tf.keras.callbacks.TensorBoard(log_dir='./logs'),
#     tf.keras.callbacks.LambdaCallback(
#         on_epoch_end=lambda epoch, logs:
#         print(f"\nCurrent LR: {tf.keras.backend.get_value(model.optimizer.learning_rate):.6f}")
#     )
# ]

# # 🔵 Phase 1: Train top layers with frozen EfficientNetB0
# print("🔵 Phase 1: Training with Frozen EfficientNetB0")
# history = model.fit(
#     train_gen,
#     epochs=EPOCHS,
#     validation_data=val_gen,
#     callbacks=callbacks,
#     class_weight=class_weights
# )

# # 🔵 Phase 2: Fine-tune EfficientNetB0 (Unfreeze last 30 layers)
# print("\n🔵 Phase 2: Fine-Tuning EfficientNetB0")
# base_model.trainable = True
# for layer in base_model.layers[:-30]:
#     layer.trainable = False

# model.compile(
#     optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001),
#     loss='categorical_crossentropy',
#     metrics=[
#         'accuracy',
#         tf.keras.metrics.Precision(name='precision'),
#         tf.keras.metrics.Recall(name='recall'),
#         tf.keras.metrics.AUC(name='auc')
#     ]
# )

# history = model.fit(
#     train_gen,
#     epochs=EPOCHS + 20,
#     initial_epoch=history.epoch[-1] + 1,
#     validation_data=val_gen,
#     callbacks=callbacks,
#     class_weight=class_weights
# )

# # ✅ Save final model
# model.save(MODEL_PATH)
# print(f"\n✅ Best model saved to {MODEL_PATH}")
# print(f"🔍 Best Validation Recall: {max(history.history['val_recall']):.2%}")
# print(f"🔍 Best Validation AUC: {max(history.history['val_auc']):.2%}")


# # Load the best saved model
# model = tf.keras.models.load_model(MODEL_PATH)

# # Evaluate on validation set
# val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(val_gen)

# print(f"\n🔍 Final Evaluation on Validation Set:")
# print(f"📌 Accuracy:  {val_acc:.2%}")
# print(f"📌 Precision: {val_precision:.2%}")
# print(f"📌 Recall:    {val_recall:.2%}")
# print(f"📌 AUC:       {val_auc:.2%}")


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision, backend as K
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Mixed precision policy
mixed_precision.set_global_policy('mixed_float16')

# ================== PATHS & PARAMS ==================
TRAIN_DIR = '/teamspace/studios/this_studio/car_model_detection/data/processed/train'
VAL_DIR = '/teamspace/studios/this_studio/car_model_detection/data/processed/val'
MODEL_PATH = '/teamspace/studios/this_studio/car_model_detection/models/efficientnetv2s_car_model.keras'
IMG_SIZE = (384, 384)
BATCH_SIZE = 32
EPOCHS_PHASE1 = 20
EPOCHS_PHASE2 = 40

# ================== CUSTOM LAYERS ==================
class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')
        super().build(input_shape)
        
    def call(self, x):
        avg_pool = K.expand_dims(K.mean(x, axis=-1), axis=-1)
        max_pool = K.expand_dims(K.max(x, axis=-1), axis=-1)
        concat = K.concatenate([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return layers.multiply([x, attention])
        # ================== FIXED LOSS FUNCTION ==================
class WeightedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, class_weights, gamma=2.0, alpha=0.25, name='weighted_focal_loss'):
        super().__init__(name=name)
        self.class_weights = tf.constant(class_weights, dtype=tf.float32)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        # Compute per-sample cross-entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1)  # shape: (batch_size,)

        pt = tf.exp(-ce)
        alpha_t = self.alpha * tf.reduce_sum(y_true, axis=-1) + (1 - self.alpha) * (1 - tf.reduce_sum(y_true, axis=-1))
        
        sample_weights = tf.gather(self.class_weights, tf.argmax(y_true, axis=-1))  # shape: (batch_size,)
        focal_loss = sample_weights * alpha_t * tf.pow(1 - pt, self.gamma) * ce
        return tf.reduce_mean(focal_loss)

        # ================== MODEL ARCHITECTURE ==================
def build_model(num_classes):
    base_model = EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        include_preprocessing=True
    )

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs)
    x = SpatialAttention()(x)
    
    gap = layers.GlobalAvgPool2D()(x)
    gmp = layers.GlobalMaxPool2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    x = layers.Dense(192, activation='swish', kernel_regularizer='l2')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    return models.Model(inputs, outputs)

    # ================== MAIN TRAINING FUNCTION ==================
def main():
    # Data pipeline
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.9, 1.1],
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    print("Class weights:", class_weights)  # Debug print

    # Build model
    model = build_model(len(train_gen.class_indices))

    # Phase 1: Frozen base
    model.compile(
        optimizer=AdamW(learning_rate=3e-4),
        loss=WeightedFocalLoss(class_weights),
        metrics=['accuracy']
    )

    print("🚀 Phase 1: Training head (frozen backbone)")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5),
            tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True)
        ]
    )

    # Phase 2: Gradual unfreeze
    print("🚀 Phase 2: Fine-tuning")
    for layer in model.layers[-30:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer=AdamW(learning_rate=1e-5),
        loss=WeightedFocalLoss(class_weights),
        metrics=['accuracy']
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_PHASE1 + EPOCHS_PHASE2,
        initial_epoch=EPOCHS_PHASE1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True)
        ]
    )

    # Evaluation
    val_preds = model.predict(val_gen)
    print("\n📊 Classification Report:")
    print(classification_report(
        val_gen.classes,
        np.argmax(val_preds, axis=1),
        target_names=val_gen.class_indices.keys()
    ))
    
    # Confusion matrix
    cm = confusion_matrix(val_gen.classes, np.argmax(val_preds, axis=1))
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=val_gen.class_indices.keys(), 
                yticklabels=val_gen.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.show()
    
    model.save(MODEL_PATH)
    print(f"✅ Best validation accuracy: {max(history.history['val_accuracy']):.2%}")

if __name__ == "__main__":
    main()