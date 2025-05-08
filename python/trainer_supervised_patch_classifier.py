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


def backbone_finder (model_names,patches_path, output_path):
  if not os.path.exists(output_path):
     os.makedirs(output_path)
  for model_name in model_names:
    output_path_ = os.path.join(output_path, model_name)
    if not os.path.exists(output_path_):
          os.makedirs(output_path_)
    try:
      MyClassifier = MulticlassClassifier(data_dir=patches_path, 
                                    patch_size_ratio = 1,
                                    remove_classes=["Mobile invertebrate", "Unscorable", "Sessile invertebrate community"],
                                    to_be_combined=["Bare rock", "Turf", "Encrusting algae","Filamentous algae", "grazed rock"],
                                    outlier_classes=[],
                                    sample_ratios={},
                                    batch_size=64,
                                    model_name=model_name,
                                    epochs=100,
                                    initial_learning_rate=0.001,
                                    decay_steps=1236,
                                    decay_rate=0.9,
                                    lr_schedule_flag=False,
                                    full_training_flag=True,
                                    dense_number=512,
                                    dropout_ratio=0.3,
                                    loss_function='categorical_crossentropy',
                                    last_activation='softmax',
                                    patience=15,
                                    model_save_path=output_path_,
                                    verbose=False)   
      MyClassifier.fit()
    except:
       print (model_name, ": issues for training, check it out")

if __name__ == '__main__':
    output_path = os.path.join("backbone_selection_results_patch","best_backbone_full_training")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    patches_path = os.path.join(".","dataset","patches")

    model_names = ["inception","efficient", "efficientL", "resnet","convnextB","convnextS","xception","densenet","inception_resnet"]
    
    model_names = "convnextB"
    backbone_finder (model_names,patches_path, output_path)

    
    