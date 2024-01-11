import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc, f1_score



dir_path = 'test'
models_dir='models'
model_files = os.listdir(models_dir)
image_files = os.listdir(dir_path)
images=[]
true_labels = []
names=[]
predicted_classes_list = []
accuracies = []
f1_scores=[]
fprs=[]
tprs = []
roc_aucs = []
precisions = []
recalls = []

for file_name in image_files:
    # Load the image and normalize its pixel values
    img = image.load_img( os.path.join( dir_path, file_name ), target_size=(224, 224) )
    img_array = image.img_to_array( img )
    img_array = np.expand_dims( img_array, axis=0 )
    img_array = img_array / 255.0
    images.append( img_array )
    names.append(file_name)


    if 'normal' in file_name:

        true_label = 1
        true_labels.append( true_label )
    else:
        true_label = 0
        true_labels.append( true_label )

#Load the image and normalize its pixel values
images = np.concatenate(images, axis=0)

for i, model_file in enumerate(model_files):
    # Load the model
    predicted_classes = []
    model_path = os.path.join(models_dir, model_file)
    model = keras.models.load_model(model_path)
    print(f"\n\n TESTING MODEL : {i+1} out of {len(model_files)} : {model_file}")
    # Make a prediction on the input data
    predictions = model.predict( images)
    # Get the predicted class for the image
    predicted_class = np.argmax( predictions, axis=1 )
    # Append the predicted class to the list
    predicted_classes.append( predicted_class )

    print("Probabilty of each class \n [  Abnormal , Normal  ] \n",predictions)
    accuracy = accuracy_score(true_labels, predicted_class)
    f1 = f1_score(true_labels, predicted_class)
    fpr, tpr, _ = roc_curve(true_labels, predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(true_labels, predictions[:, 1])

    accuracies.append(accuracy)
    fprs.append(fpr)
    tprs.append(tpr)
    f1_scores.append(f1)
    roc_aucs.append(roc_auc)
    precisions.append(precision)
    recalls.append(recall)


    for i,name in enumerate(names):

            if predicted_class[i] == 0 :
                label="Abnormal"
            else:
                label="Normal"

            print( f"{model_file} model :Predicted class for {name}: {predicted_class[i]} : {label}" )


# Plot ROC curve for each model

num_models = len(model_files)
num_cols = 3
num_rows = (num_models + num_cols - 1) // num_cols  # Round up to the nearest integer

fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
axs = axs.flatten()  # Flatten the axs array to simplify indexing

for i, model_file in enumerate(model_files):
    axs[i].plot( fprs[i], tprs[i], label=f'{model_file} (AUC = {roc_aucs[i]:.2f})' )
    # axs[i].plot( [0, 1], [0, 1], color='navy', linestyle='--' )
    axs[i].set_xlim( [0.0, 1.0] )
    axs[i].set_ylim( [0.0, 1.05] )
    axs[i].set_xlabel( 'False Positive Rate', fontsize=8 )
    axs[i].set_ylabel( 'True Positive Rate', fontsize=8 )
    axs[i].set_title( f'ROC - {model_file}', fontsize=8 )
    axs[i].legend( loc="lower right", fontsize=5 )
# Hide empty subplots if there are fewer models than subplots
for j in range(num_models, num_rows * num_cols):
    axs[j].axis('off')

plt.tight_layout()
plt.show()

#plot the precision recall

nrows = (len(model_files) + 2) // 3  # Calculate the number of rows
fig, axs = plt.subplots(nrows, 3, figsize=(18, 6 * nrows), sharex=True, sharey=True)

plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust spacing between subplots

for i, model_file in enumerate(model_files):
    row = i // 3  # Calculate the row index
    col = i % 3  # Calculate the column index
    axs[row, col].plot(recalls[i], precisions[i], label=f'{model_file}')
    axs[row, col].set_xlim([0.0, 1.0])
    axs[row, col].set_ylim([0.0, 1.05])
    axs[row, col].set_xlabel('Recall',fontsize=5)
    axs[row, col].set_ylabel('Precision',fontsize=5)
    axs[row, col].set_title(f'Precision-Recall Curve - {model_file}',fontsize=5)
    axs[row, col].legend(loc="lower right")

# Remove any unused subplots
for i in range(len(model_files), nrows * 3):
    axs.flat[i].axis('off')

plt.tight_layout()
plt.show()

# plt.figure(figsize=(10, 6))
# for i, model_file in enumerate(model_files):
#     plt.plot(recalls[i], precisions[i], label=f'{model_file}')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower right")
# plt.show()


# Plot accuracy curve for each model
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model_files) + 1), accuracies, marker='o')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.xticks(range(1, len(model_files) + 1), model_files, rotation=45)
plt.show()

#plot the f1 score
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model_files) + 1), f1_scores, marker='o')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.title('F1 Score Curve')
plt.xticks(range(1, len(model_files) + 1), model_files, rotation=45)
plt.show()