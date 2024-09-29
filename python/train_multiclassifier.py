from multiclassifier import *


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)


if __name__ == '__main__':
    output_path = os.path.join("saved_models","final_softmax")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    patches_path = os.path.join(".","dataset","patches")
    MyClassifier = MulticlassClassifier(data_dir=patches_path, 
                                patch_size_ratio = 1,
                                remove_classes=["Mobile invertebrate", "Unscorable", "Sessile invertebrate community"],
                                to_be_combined=["Bare rock", "Turf", "Encrusting algae","Filamentous algae", "grazed rock"],
                                outlier_classes=[],
                                sample_ratios={},
                                batch_size=128,
                                model_name="inception",
                                epochs=1,
                                initial_learning_rate=0.001,
                                decay_steps=1236,
                                decay_rate=0.9,
                                lr_schedule_flag=True,
                                full_training_flag=True,
                                dense_number=1024,
                                dropout_ratio=0.3,
                                loss_function='categorical_crossentropy',
                                last_activation='softmax',
                                patience=10,
                                model_save_path=output_path,
                                verbose=False)   
MyClassifier.fit()