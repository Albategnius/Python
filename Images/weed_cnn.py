# Import packages to for DL Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dropout, Flatten, Dense, MaxPool1D
from keras import applications
from sklearn.metrics import confusion_matrix
import cv2
import time
import datetime
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import os, glob, sys
import shutil
from keras.optimizers import SGD
from sklearn import metrics
from keras.callbacks import TensorBoard
from time import time
from keras import backend
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import scipy.ndimage as ndimage
import seaborn as sns
import argparse
import warnings
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "0" ##sck

# Import packages to create output Shape file
from osgeo import ogr, osr, gdal
from pyproj import Proj, transform

plt.switch_backend('agg')  
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--home", required=True, help="path to home directory. To ensure folder structure is maintained.")
ap.add_argument("-i", "--src", default=[], nargs="+", help="tif image file name e.g. 03_201806271352_RI_GLB_GLBC288D01_RGB")
ap.add_argument("-t", "--train", action='store_false', help="Set -t if for test scoring only i.e. no model will be trained") # store_false == default always true
ap.add_argument("-m", "--modelname", help="Set model name if Train -t is set")
ap.add_argument("-c", "--multiclass", action='store_true', help="Set -c if multiclass model i.e. > 2 classes") # store_true == default always false
ap.add_argument("-p", "--inputfolder", required=True, help="path to input folder")
args = ap.parse_args()

home = args.home
train = args.train
src = args.src
modelname = args.modelname
multiclass = args.multiclass
inputfolder = args.inputfolder

# source activate app_proto_py36
print("The home address is: ", home)

# Specify True to train model
print("Model is set to: ", train)

# Test petak filenames
print("The test petaks are: ", src)

print("Model multiclass: ", multiclass)

print("Input Folder: ", inputfolder)

# Test Image Path & ID
#src = "03_201806271352_RI_GLB_GLBC283D01_RGB.tif
#testid = src.split("_")[4]
#print("The test petak id is: ", testid)

print("The modelname is:", modelname)

# Model and data directories
train_data_dir = home + inputfolder
test_data_dir = home + 'Blindtest_Multiclass_NEW/'
saveDir = home + 'Output/'
raw_dir = home + 'RawData/'
archive_dir = home + 'SMF_DL_weed_models_archive/'
trainAnalysis = archive_dir + 'trainAnalysis/'
modelArchive = archive_dir + 'modelArchive/'


dir_path= home + 'Scripts/'
os.chdir(dir_path)
sys.path.append(os.path.abspath(dir_path))

# Dimensions of images
img_width, img_height = 75, 75 # 75  #ngitung 75 berdasarkan limit di blindtest?

# Model information
train_batch_size = 10 # 256
val_batch_size = 10 # 256
epochs = 2

# Write log file
def DL_log_file(name, txt):
    with open(name + '.txt', 'a') as file: #'file.txt'
        file.write(txt+'\n')

def DL_log_file_model(name, model):
    with open(name + '.txt', 'a') as fh: #'file.txt'
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

# convert pixel to lat/lon
def pixelToLatLon(TL_lon , TL_lat, radPerPixX, radPerPixY, x, y):
    # X-axis assumed to point East and Y-axis to South
    lon = TL_lon + x * radPerPixX
    lat = TL_lat + y * radPerPixY
    return lon, lat

	
# create polygons for each grid
def create_polygon(coords_cnt, TLx2, TLy2, radPerPixX, radPerPixY):
    ring = ogr.Geometry(ogr.wkbLinearRing)

    for arr in coords_cnt:
        lon, lat = pixelToLatLon(TL_lon=TLx2, TL_lat=TLy2,
                                 radPerPixX=radPerPixX, radPerPixY=radPerPixY,
                                 x=int(arr[0]), y=int(arr[1]))
        ring.AddPoint(lon, lat)

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly.ExportToWkt()
	
# write shape file
def write_shapefile(tmp, coords_cnt, out_shp, TLx2, TLy2, radPerPixX, radPerPixY):
    t_srs = osr.SpatialReference() ###
    t_srs.SetFromUserInput('EPSG:4326') ###

    # Now convert it to a shapefile with OGR
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.CreateDataSource(out_shp)
    layer = ds.CreateLayer('', srs=t_srs, geom_type=ogr.wkbPolygon)

    # Add one attribute
    layer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))

    for i in range(0, len(tmp)):
        poly = create_polygon(coords_cnt[i], TLx2, TLy2, radPerPixX, radPerPixY)
        defn = layer.GetLayerDefn()

        #  Create a new feature (attribute and geometry)
        feat = ogr.Feature(defn)
        feat.SetField('id', i + 1)

        # Make a geometry, from Shapely object
        geom = ogr.CreateGeometryFromWkt(poly)
        feat.SetGeometry(geom)

        layer.CreateFeature(feat)

        #feat = geom = None  # destroy these

    # Save and close everything
    #ds = layer = None
	
# obtain tif georef info
def load_georef(raw_dir, src):
    raster = gdal.Open(raw_dir + src)

    xsize = raster.RasterXSize
    ysize = raster.RasterYSize
    print(xsize, ysize)

    k = raster.GetGeoTransform() #jgw files

    x = k[0] + (xsize-1) * k[1] + (ysize-1) * k[2]
    y = k[3] + (xsize-1) * k[4] + (ysize-1) * k[5]

    # Extract CRS
    prj = osr.SpatialReference(raster.GetProjection())
    init_crs = 'epsg:' + prj.GetAttrValue('AUTHORITY',1)
    
    print("This is the init_crs ", init_crs)
    # Conversion to lat lon
    inProj = Proj(init=init_crs)
    outProj = Proj(init='epsg:4326')

    TLx1, TLy1 = k[0], k[3]
    TLx2, TLy2 = transform(inProj, outProj, TLx1, TLy1)

    BRx1, BRy1 = x, y
    BRx2, BRy2 = transform(inProj, outProj, BRx1, BRy1)

    radPerPixX = float(BRx2 - TLx2)/float(xsize-1)
    radPerPixY = float(BRy2 - TLy2)/float(ysize-1)

    return TLx2, TLy2, radPerPixX, radPerPixY

# Acc and Loss plot for the trained model
def performance_results(results, trainAnalysis, modelname):
    ## plot the history of the CNN model
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    ax[0].plot(results.history['acc'])
    ax[0].plot(results.history['val_acc'])
    # ax[0].set_xlim(0, 50)
    # xinterval = np.arange(0, 50, 2)
    # ax[0].set_xticks(xinterval)
    # ax[0].set_ylim(0.4, 1)
    # yinterval = np.arange(0.4, 1, 0.025)
    # ax[0].set_yticks(yinterval)
    ax[0].set_title('model accuracy')
    ax[0].set_ylabel('accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].legend(['train', 'val'], loc='upper left')


    ax[1].plot(results.history['loss'])
    ax[1].plot(results.history['val_loss'])
    # ax[1].set_xlim(0, 50)
    # xinterval = np.arange(0, 50, 2)
    # ax[1].set_xticks(xinterval)
    # ax[1].set_ylim(0, 1)
    # yinterval = np.arange(0, 1, 0.05)
    # ax[1].set_yticks(yinterval)
    ax[1].set_title('model loss')
    ax[1].set_ylabel('loss')
    ax[1].set_xlabel('epoch')
    ax[1].legend(['train', 'val'], loc='upper left')

    #plt.show()
    fig.savefig(trainAnalysis+modelname+"_Acc and Loss.jpeg")
    plt.close()


# Optimal Cutoff for binary case
def get_best_cutoff(fpr, tpr, thresholds):

    # cutoff based on min absolute difference of sen and spec
    diff = np.abs(tpr - (1-fpr))
    ind = np.where(diff == min(diff))[0][0]
    cut_min = thresholds[ind]

    # cutoff based on youden index
    diff = tpr + (1-fpr) - 1
    ind = np.where(diff == max(diff))[0][0]
    cut_youden = thresholds[ind]

    # cutoff based on highest avg of sen and spec
    diff = (tpr + (1-fpr)) / 2
    ind = np.where(diff == max(diff))[0][0]
    cut_avg = thresholds[ind]

    # cutoff based on min euclidean dist in ROC curve
    diff = np.sqrt(fpr**2 + (1-tpr)**2)
    ind = np.where(diff == min(diff))[0][0]
    cut_euclid = thresholds[ind]

    return cut_min, cut_youden, cut_avg, cut_euclid

# Build Model
def build_model(multiclass):
    # Specify and build CNN Model
    model = Sequential()
	
    # Block 1
    model.add(Conv2D(32, (1, 1), padding='same', input_shape=(img_width, img_height, 3), activation= 'relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(64, (1, 1), padding='same', activation= 'relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation= 'relu'))

    # Block 3
    model.add(Conv2D(128, (1, 1), padding='same', activation= 'relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation= 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    # model.add(Dense(4096, activation='relu', name='fc1'))
    # model.add(Dropout(0.5))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    if multiclass:
        model.add(Dense(3))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

    # Specify optimization function
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)

    # Compile model
    if multiclass:
        model.compile(loss='categorical_crossentropy', optimizer=sgd , metrics=['accuracy'])
    else:
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
    return model

# Predict scores
def predict_scores(data_generator, multiclass):
    v_score = model.evaluate_generator(data_generator)
    print("Loss: ", v_score[0], "Accuracy: ", v_score[1])

    # Predictions on Validation Set
    ActualLabels = []
    ScoreAssigned = []

    ActualLabelsMul = []
    ScoreAssignedMul = []
    
    if multiclass:
        data_generator.reset()
        print(data_generator.batch_index)
        for i in range(int(data_generator.n / data_generator.batch_size) + 1):
            batch = data_generator.next()
            for j in range(len(batch[0])):
                img = batch[0][j]
                img = img.reshape((1,) + img.shape)
                pred = model.predict(img).tolist().pop()
                labels = batch[1][j]
                
                ScoreAssignedMul.append(pred)
                ActualLabelsMul.append(labels)

                pred = np.where(i==np.max(i))[0]
                labels = np.where(i==np.max(i))[0]

                ScoreAssigned.append(pred)
                ActualLabels.append(labels)

        data_generator.reset()
        print(data_generator.batch_index)
    else:
        data_generator.reset()
        print(data_generator.batch_index)

        for i in range(int(data_generator.n / data_generator.batch_size) + 1):
            batch = data_generator.next()
            for j in range(len(batch[0])):
                img = batch[0][j]
                img = img.reshape((1,) + img.shape)
                pred = model.predict(img).tolist().pop()
                labels = batch[1][j]
                ScoreAssigned.append(pred)
                ActualLabels.append(labels)

        data_generator.reset()
        print(data_generator.batch_index)

    return ActualLabels, ScoreAssigned

# Probability Density Plots
def prob_density_plt(ScoreAssigned, ActualLabels, trainAnalysis, modelname, multiclass):
    # Probability density plots
    sco = np.array(ScoreAssigned)
    lab = np.array(ActualLabels)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    
    sns.distplot(sco[lab == 0], ax=ax, hist=True, kde=True, bins=10, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
    sns.distplot(sco[lab == 1], ax=ax, hist=True, kde=True, bins=10, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
    if multiclass:
        sns.distplot(sco[lab == 2], ax=ax, hist=True, kde=True, bins=10, hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
    
    fig.savefig(trainAnalysis+modelname+"_prob density plots.jpeg")  
    plt.close()
    
# ROC metric and plots
def roc_plt(ScoreAssigned, ActualLabels, trainAnalysis, modelname):
    sco = np.array(ScoreAssigned)
    lab = np.array(ActualLabels)
    roc = metrics.roc_auc_score(lab, sco)

    fpr, tpr, thresholds = metrics.roc_curve(ActualLabels, ScoreAssigned)
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(fpr, tpr)
    
    fig.savefig(trainAnalysis+modelname+"_ROC plots.jpeg")
    plt.close()

    return roc, fpr, tpr, thresholds

# Load Trained model
def loading_model(model_path, model_weights_path):
    model = load_model(model_path)
    model.load_weights(model_weights_path)
    return model

# Load data
datagen_train = ImageDataGenerator(rescale=1. / 255, rotation_range=180,
                                   width_shift_range=0.2, height_shift_range=0.2,
                                   horizontal_flip=True, vertical_flip=True,
                                   fill_mode='wrap',
                                   shear_range=0.05, zoom_range=0.05,
                                   brightness_range=(0.6, 1.5))
datagen_val = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
datagenTest = ImageDataGenerator(rescale=1. / 255)

# datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2, rotation_range=8)
# datagenTest = ImageDataGenerator(rescale=1./255)

if multiclass:
    classmode = 'categorical'
else:
    classmode = 'binary'

train_generator = datagen_train.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=train_batch_size,
    class_mode=classmode,
    shuffle=True,
    #save_to_dir = saveAugImages,
    subset="training")

val_generator = datagen_val.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=val_batch_size,
    class_mode=classmode,
    shuffle=False,
    #save_to_dir = saveAugImages,
    subset="validation")

#backend.set_learning_phase(1)

#########################################################################################
###################################### Train Model ######################################
#########################################################################################

if train:
    # Specify and build CNN Model
    model = build_model(multiclass)

    '''
    x_val_lst = []
    y_val_lst = []
    imagesRead = 0
    for name in val_generator.filenames:
        img = cv2.imread(train_data_dir+'\\'+name)
        imagesRead += 1 #imagesRead + 1
        print("Images read: "+str(imagesRead))
        x_val_lst.append((img))
        if name.startswith("not_tree"):
            y_val_lst.append(0)
        elif name.startswith("tree"):
            y_val_lst.append(1)
        del(img)
    
    x_val = np.array(x_val_lst)
    y_val = np.array(y_val_lst)'''

    '''os.chdir(tensor_log_dir)
    print(os.getcwd())
    
    # Setup Tensorboard
    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time()),
        histogram_freq=10,
        write_graph=True,
        write_grads=True,
        batch_size=batch_size,
        write_images=True)'''

    # Train Model
    counter = Counter(train_generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}
    
    print(class_weights, max_val, counter)

    results = model.fit_generator(
            train_generator,
            steps_per_epoch=int(len(train_generator.filenames)//train_batch_size),
            epochs=epochs,
            validation_data=val_generator,#, #(x_val,y_val),#val_generator,
            validation_steps=int(len(val_generator.filenames)// val_batch_size),
            class_weight=class_weights,
            workers=2)
            #callbacks=[tensorboard])

    # Save training graphs and model files in desired folders
    id = str(datetime.datetime.now())
    id = id.replace(':', '_')
    id = id.replace(' ', '_t_')

    todayDate = datetime.datetime.today().strftime('%Y-%m-%d')

    runNo = int(len([name for name in os.listdir(modelArchive) if os.path.isfile(modelArchive+name)])/3+1)

    # Save weights after training
    modelname = 'run_' + str(runNo) + '_' + todayDate + '_' + classmode + '_' + inputfolder
    model.save_weights(modelArchive+modelname+'_model_weights.h5')
    model.save(modelArchive+modelname+'_model.h5')

    # Training Results Accuracy and Loss plots
    performance_results(results, trainAnalysis, modelname)
    print("Mean Acc: "+str(sum(results.history['acc'])/len(results.history['acc'])))
    print("Mean loss: "+str(sum(results.history['loss'])/len(results.history['loss'])))
    print("Mean val_acc: "+str(sum(results.history['val_acc'])/len(results.history['val_acc'])))
    print("Mean val_loss: "+str(sum(results.history['val_loss'])/len(results.history['val_loss'])))

    ### Log File
    # Run Configuration
    DL_log_file(modelArchive + '/' + modelname, " CONFIGURATION ") 
    DL_log_file(modelArchive + '/' + modelname, "The model id is:" + modelname)
    DL_log_file(modelArchive + '/' + modelname, "The training input folder is:" + inputfolder)
    DL_log_file(modelArchive + '/' + modelname, "Model Training is set to: " + str(train))
    DL_log_file(modelArchive + '/' + modelname, "Model Class is set to: " + classmode)        
    DL_log_file(modelArchive + '/' + modelname, "The species and age is:" + "EPEL 6M")
    DL_log_file(modelArchive + '/' + modelname, "")
            
    # Model Info
    DL_log_file(modelArchive + '/' + modelname, " MODEL INFO ")    
    DL_log_file(modelArchive + '/' + modelname, "Image width: " + str(img_width) + "  Image height: " + str(img_height))
        
    DL_log_file(modelArchive + '/' + modelname, "Train Batch Size: " + str(train_batch_size))
    DL_log_file(modelArchive + '/' + modelname, "Validation Batch Size: " + str(val_batch_size))
    DL_log_file(modelArchive + '/' + modelname, "Epochs: " + str(epochs))
    DL_log_file(modelArchive + '/' + modelname, "")

    DL_log_file(modelArchive + '/' + modelname, "Class Weights " + str(class_weights))
    DL_log_file(modelArchive + '/' + modelname, "Max Val " + str(max_val))
    DL_log_file(modelArchive + '/' + modelname, "Counter " + str(counter))
    DL_log_file(modelArchive + '/' + modelname, "")

    DL_log_file(modelArchive + '/' + modelname, " MODEL ARCHITECTURE ") 
    DL_log_file_model(modelArchive + '/' + modelname, model)
    DL_log_file(modelArchive + '/' + modelname, "")
                                                                                                                                                                                                                                            
    # Write metrics to log file
    DL_log_file(modelArchive + '/' + modelname, " TRAIN MODEL METRICS ") 
    DL_log_file(modelArchive + '/' + modelname, "Mean loss: "+str(sum(results.history['loss'])/len(results.history['loss'])))
    DL_log_file(modelArchive + '/' + modelname, "Mean Acc: "+str(sum(results.history['acc'])/len(results.history['acc'])))
    DL_log_file(modelArchive + '/' + modelname, "Mean val_acc: "+str(sum(results.history['val_acc'])/len(results.history['val_acc'])))
    DL_log_file(modelArchive + '/' + modelname, "Mean val_loss: "+str(sum(results.history['val_loss'])/len(results.history['val_loss'])))
    DL_log_file(modelArchive + '/' + modelname, "")

#########################################################################################
###################################### Load Model #######################################
#########################################################################################

if not train:
    model_path = modelArchive + modelname + '_model.h5'#'run1_2018-09-03_model.h5'
    model_weights_path = modelArchive + modelname + '_model_weights.h5' #/run1_2018-09-03model_weights.h5'
    print(model_path)
    print(model_weights_path)
    model = loading_model(model_path, model_weights_path)
    
    ### Log File
    # Run Configuration
    DL_log_file(modelArchive + '/' + modelname, " CONFIGURATION ") 
    DL_log_file(modelArchive + '/' + modelname, "The model id is:" + modelname)
    DL_log_file(modelArchive + '/' + modelname, "Model Training is set to: " + str(train)) 
    DL_log_file(modelArchive + '/' + modelname, "Model Class is set to: " + classmode)     
    DL_log_file(modelArchive + '/' + modelname, "The species and age is:" + "EPEL 2M")
    DL_log_file(modelArchive + '/' + modelname, "")
            
    # Model Info
    DL_log_file(modelArchive + '/' + modelname, " MODEL INFO ")    
    DL_log_file(modelArchive + '/' + modelname, "Image width: " + str(img_width) + "  Image height: " + str(img_height))
        
    DL_log_file(modelArchive + '/' + modelname, "Train Batch Size: " + str(train_batch_size))
    DL_log_file(modelArchive + '/' + modelname, "Validation Batch Size: " + str(val_batch_size))
    DL_log_file(modelArchive + '/' + modelname, "Epochs: " + str(epochs))
    DL_log_file(modelArchive + '/' + modelname, "")
    
#########################################################################################
################################# Score Validation Data #################################
#########################################################################################
# Model metrics for Validation set - loss and accuracy
v_score = model.evaluate_generator(val_generator)
print("Loss: ", v_score[0], "Accuracy: ", v_score[1])

DL_log_file(modelArchive + '/' + modelname, " VALIDATION MODEL METRICS ") 
DL_log_file(modelArchive + '/' + modelname, "Loss: " + str(v_score[0]) + "  Accuracy: " + str(v_score[1]))

# Predictions on Validation Set
ActualLabels = []
ScoreAssigned = []

val_generator.reset()
print("The validation batch index prior to extraction is: ", val_generator.batch_index)

if multiclass:
    ActualLabelsMul = []
    ScoreAssignedMul = []
    
    val_generator.reset()
    print(val_generator.batch_index)
    for i in range(int(val_generator.n / val_generator.batch_size) + 1):
        batch = val_generator.next()
        for j in range(len(batch[0])):
            img = batch[0][j]
            img = img.reshape((1,) + img.shape)
            pred = model.predict(img).tolist().pop()
            labels = batch[1][j]

            ScoreAssignedMul.append(pred)
            ActualLabelsMul.append(labels)

    ScoreAssigned = [np.max(i) for i in ScoreAssignedMul]
    PredLabels = [np.where(i==np.max(i))[0][0] for i in ScoreAssignedMul]
    ActualLabels = [np.where(i==np.max(i))[0][0] for i in ActualLabelsMul]

    # TODO: if max index is treeweed -- consider assigning to weed if treeweed score>0.5
    # print("LENGTH, ELEMENT[1], ScoreAssignedMul[1]", len(PredLabels), " ", PredLabels[1], " " , ScoreAssignedMul[1][1])
    # for i in range(0, len(PredLabels)):
    #     if PredLabels[i] == 1 and ScoreAssignedMul[i][1] > np.float(0.5):
    #         PredLabels[i] = 2 

    val_generator.reset()
    print(val_generator.batch_index)
else:
    val_generator.reset()
    print(val_generator.batch_index)
    # Extract actual labels
    for i in range(int(val_generator.n/val_generator.batch_size)+1):
        batch = val_generator.next()
        for j in range(len(batch[0])):
            img = batch[0][j]
            img = img.reshape((1,) + img.shape)
            
            ScoreAssigned.append(model.predict(img).tolist().pop())
            ActualLabels.append(batch[1][j])

val_generator.reset()
print("The validation batch index prior to extraction is: ", val_generator.batch_index)

# Probability density plots
prob_density_plt(ScoreAssigned, ActualLabels, trainAnalysis, modelname, multiclass)

# ROC metric and plots for binary case only
if not multiclass:
    roc, fpr, tpr, thresholds = roc_plt(ScoreAssigned, ActualLabels, trainAnalysis, modelname)
    print("The validation ROC Score is: ", roc)
    DL_log_file(modelArchive + '/' + modelname, "The validation ROC Score is: " + str(roc))

    # Optimal Cutoff Score is the average of the four best cutoff scores
    cut_min, cut_youden, cut_avg, cut_euclid = get_best_cutoff(fpr, tpr, thresholds)
    print("The cutoff scores are as follows: ", cut_min, cut_youden, cut_avg, cut_euclid)
    DL_log_file(modelArchive + '/' + modelname, "The cutoff scores are as follows: " + str([cut_min, cut_youden, cut_avg, cut_euclid]))

    opt_cutoff = np.mean([cut_min, cut_youden, cut_avg, cut_euclid])
    print("The optimal cutoff scores is: ", opt_cutoff)
    DL_log_file(modelArchive + '/' + modelname, "The optimal cutoff scores is: " + str(opt_cutoff))
    DL_log_file(modelArchive + '/' + modelname, "")

#backend.set_learning_phase(0)

#########################################################################################
################################# Load Test Data ########################################
#########################################################################################

for k in src:
    # Test Image Path & ID
    tmp = k.split("_")
    testid = "_".join(tmp[2:5]) 
    print("The test petak id is: ", testid)
    
    test_dir = test_data_dir + testid + '/'
  
    test_generator = datagenTest.flow_from_directory(
        test_dir,
        target_size=(img_width, img_height),
        batch_size=train_batch_size,
        class_mode=classmode,
        shuffle=False)

    #########################################################################################
    #################################### Score Test Data ####################################
    #########################################################################################
    # Model metrics for Test set - loss and accuracy
    score = model.evaluate_generator(test_generator)#, int(len(test_generator.filenames)/batch_size), workers=12)
    print("Loss: ", score[0], "Accuracy: ", score[1])
    
    
    DL_log_file(modelArchive + '/' + modelname, " TEST MODEL METRICS ") 
    DL_log_file(modelArchive + '/' + modelname, "The test petak id is: " + testid)
    DL_log_file(modelArchive + '/' + modelname, "Loss: " + str(score[0]) + "  Accuracy: " + str(score[1]))
    
    # Predictions on Test Set
    ActualLabelsTest = []
    ScoreAssignedTest = []
        
    # Extract actual labels
    if multiclass:
        ActualLabelsMul = []
        ScoreAssignedMul = []
        
        test_generator.reset()
        print("The test batch index prior to extraction is: ", test_generator.batch_index)

        for i in range(int(test_generator.n / test_generator.batch_size) + 1):
            batch = test_generator.next()
            for j in range(len(batch[0])):
                img = batch[0][j]
                img = img.reshape((1,) + img.shape)
                pred = model.predict(img).tolist().pop()
                labels = batch[1][j]

                ScoreAssignedMul.append(pred)
                ActualLabelsMul.append(labels)

        ScoreAssignedTest = [np.max(i) for i in ScoreAssignedMul]
        PredLabelsTest = [np.where(i==np.max(i))[0][0] for i in ScoreAssignedMul]
        ActualLabelsTest = [np.where(i==np.max(i))[0][0] for i in ActualLabelsMul]

        # TODO: if max index is treeweed -- consider assigning to weed if treeweed score>0.5
        # for i in range(0, len(PredLabelsTest)):
        #     if PredLabelsTest[i] == 1 and ScoreAssignedMul[i][1] > np.float(0.5):
        #         PredLabelsTest[i] = 2 

        test_generator.reset()
        print(test_generator.batch_index)
    else:
        test_generator.reset()
        print("The test batch index prior to extraction is: ", test_generator.batch_index)

        for i in range(int(test_generator.n/test_generator.batch_size)+1):
            batch = test_generator.next()
            #print(test_generator.batch_index)
            for j in range(len(batch[0])):
                img = batch[0][j]
                img = img.reshape((1,) + img.shape)
                ScoreAssignedTest.append(model.predict(img).tolist().pop())
                ActualLabelsTest.append(batch[1][j])
        
        test_generator.reset()
        print("The test batch index prior to extraction is: ", test_generator.batch_index)
    
    # Crosscheck the score lengths are the same
    print("Length of Test data file names: ", len(test_generator.filenames))
    print("Length of Actual labels: ", len(ActualLabelsTest))
    print("Length of Predicted Scores: ", len(ScoreAssignedTest))
    
    # Probability density plots
    prob_density_plt(ScoreAssignedTest, ActualLabelsTest, trainAnalysis, modelname+'_'+testid, multiclass)
    
    # Accuracy from Confusion Matrix
    if multiclass:
        CM = metrics.confusion_matrix(ActualLabelsTest, PredLabelsTest) # actual by predicted
        print(CM)
        if len(CM) > 2:
            #DL_log_file(modelArchive + '/' + modelname, str(CM))

            DL_log_file(modelArchive + '/' + modelname, 'Confusion Matrix: ')
            DL_log_file(modelArchive + '/' + modelname, str([CM[0][0], CM[0][1], CM[0][2]]))
            DL_log_file(modelArchive + '/' + modelname, str([CM[1][0], CM[1][1], CM[1][2]]))
            DL_log_file(modelArchive + '/' + modelname, str([CM[2][0], CM[2][1], CM[2][2]]))
            DL_log_file(modelArchive + '/' + modelname, "")
            
            acc = (CM[0][0]+CM[1][1]+CM[2][2])/np.sum(CM)*100
            print('The Test accuracy is:', acc, '%')
            DL_log_file(modelArchive + '/' + modelname, 'The Test accuracy is: '+ str(acc) + '%')
            DL_log_file(modelArchive + '/' + modelname, "######################################")
        else:
            DL_log_file(modelArchive + '/' + modelname, 'Confusion Matrix: ')
            DL_log_file(modelArchive + '/' + modelname, str([CM[0][0], CM[0][1]]))
            DL_log_file(modelArchive + '/' + modelname, str([CM[1][0], CM[1][1]]))
            DL_log_file(modelArchive + '/' + modelname, "")
        
            acc = (CM[0][0]+CM[1][1])/np.sum(CM)*100
            print('The Test accuracy is:', acc, '%')
            DL_log_file(modelArchive + '/' + modelname, 'The Test accuracy is: '+ str(acc) + '%')
            DL_log_file(modelArchive + '/' + modelname, "######################################")

    else:
        # ROC metric and plots
        if len(np.unique(ActualLabelsTest))>1:
            roc, fpr, tpr, thresholds = roc_plt(ScoreAssignedTest, ActualLabelsTest, trainAnalysis, modelname+'_'+testid)
            print("The Test Set ROC Score is: ", roc)
            DL_log_file(modelArchive + '/' + modelname, "The Test ROC Score is: " + str(roc))
            DL_log_file(modelArchive + '/' + modelname, "")
        else:
            print("The Test Set has only 1 level so there is no ROC Score ", np.unique(ActualLabelsTest))
            DL_log_file(modelArchive + '/' + modelname, "The Test Set has only 1 level so there is no ROC Score")
            DL_log_file(modelArchive + '/' + modelname, "")

        # Accuracy based on optimal cutoff which is derived from Validation data ROC curve
        ScoreAssignedTestLabel = []
        for i in ScoreAssignedTest:
            if i <= opt_cutoff:
                ScoreAssignedTestLabel.append(0)
            else:
                ScoreAssignedTestLabel.append(1)
        
        CM = metrics.confusion_matrix(ActualLabelsTest, ScoreAssignedTestLabel) # actual by predicted

        #DL_log_file(modelArchive + '/' + modelname, str(CM))
    
        DL_log_file(modelArchive + '/' + modelname, 'Confusion Matrix: ')
        DL_log_file(modelArchive + '/' + modelname, str([CM[0][0], CM[0][1]]))
        DL_log_file(modelArchive + '/' + modelname, str([CM[1][0], CM[1][1]]))
        DL_log_file(modelArchive + '/' + modelname, "")
        
        acc = (CM[0][0]+CM[1][1])/np.sum(CM)*100
        print('The Test accuracy is:', acc, '%')
        DL_log_file(modelArchive + '/' + modelname, 'The Test accuracy is: '+ str(acc) + '%')
        DL_log_file(modelArchive + '/' + modelname, "######################################")

    
    #########################################################################################
    #################################### Label Test Data ####################################
    #########################################################################################
    
    # Make Directories
    os.chdir(saveDir)
    
    if multiclass:
        filenames = ['mul_pred_weed', 'mul_pred_notweed', 'mul_pred_treeweed']

        for i in filenames:
            if os.path.exists(i):
                # Remove and make directories
                # shutil.rmtree(i)
                # os.mkdir(i)
                continue
            else:
                # Make directories
                os.mkdir(i)
        
        # Check if directory exists
        try:
            os.mkdir('mul_pred_weed')
            os.mkdir('mul_pred_notweed')
            os.mkdir('mul_pred_treeweed')
        except:
            print('Directory exists.')

        # Change to Test directory
        os.chdir(test_dir)
        
        # Specify list for variables
        filenames_mul_pred_weed = []
        filenames_mul_pred_notweed = []
        filenames_mul_pred_treeweed = []

        tmp_idx = []
        for i in range(int(test_generator.n/test_generator.batch_size)+1):
            # batch is a tuple (img=>3Darray, label=>1Darray)
            batch = test_generator.next()
            
            # Obtain index of batch and get batch filenames - only works if test_gen shuffle is set to False
            print(test_generator.batch_index)
            idx = (test_generator.batch_index - 1) * test_generator.batch_size
            tmp_idx.append(test_generator.batch_index) 
            if test_generator.batch_index == 0:
                print(test_generator.batch_size)
                print(max(tmp_idx))
                idx = max(tmp_idx) * test_generator.batch_size
            print("idx start : ", idx, "idx end: ", idx+len(batch[0]))
            batch_names = test_generator.filenames[idx: idx+len(batch[0])] #test_generator.filenames[idx:idx+test_generator.batch_size]
            
            print("The length of batch_names is: ", len(batch_names))
            print("The length of batch is: ", len(batch[0]))
            for j in range(len(batch[0])):    
                #count+=1
                img = batch[0][j]
                #PP Green Masking
                tmp = img
                img_hsv = cv2.cvtColor(tmp, cv2.COLOR_BGR2HSV) # since input image is BGR
                lower_green = np.array([30, 10, 10])
                upper_green = np.array([90, 255, 255])
                green_thres = cv2.inRange(img_hsv, lower_green, upper_green)
                imx = cv2.bitwise_and(tmp, tmp, mask=green_thres)
                #img = img/255
                imgPred = img.reshape((1,) + img.shape)
                img = img*255
                sc = model.predict(imgPred).tolist().pop()
                #lb = batch[1][j]
 
                #print("This prior SC is: ", sc, " ", sc[0])
                #print("This SC is: ", sc , " of type ", type(sc))
            
                #lb = np.where(i==np.max(i))[0]

                predlb = np.where(sc==np.max(sc))[0][0]
                ### Label weed if predicted as weed and not green is less than thresh value
                not_green_thres_val = np.float(0.5) #UPDATE FROM 0.6
                imx = cv2.cvtColor(imx, cv2.COLOR_BGR2GRAY)
                green_percentage =  float(np.sum(imx == 0))/(img_hsv.shape[0]*img_hsv.shape[1])

                # reset label for treeweed to weed if its scores is > 0.5            
                #if predlb == 1 and sc[1] > np.float(0.5):
                #    predlb = 2


                if predlb == 0:
                    filenames_mul_pred_notweed.append(batch_names[j])
                elif predlb == 1:
                    filenames_mul_pred_treeweed.append(batch_names[j])
                elif predlb == 2 and green_percentage < not_green_thres_val: 
                    filenames_mul_pred_weed.append(batch_names[j])

        #1: Create filenames dict
        #  test_generator.filenames
        filenames = {'mul_pred_notweed': filenames_mul_pred_notweed,
                    'mul_pred_treeweed': filenames_mul_pred_treeweed,
                    'mul_pred_weed': filenames_mul_pred_weed}
    else:
        filenames = ['correct_notweed', 'wrong_notweed', 'correct_weed', 'wrong_weed', 'pred_weed']
        
        for i in filenames:
            if os.path.exists(i):
                # Remove and make directories
                # shutil.rmtree(i)
                # os.mkdir(i)
                continue
            else:
                # Make directories
                os.mkdir(i)
        
        # Check if directory exists
        try:
            os.mkdir('correct_notweed')
            os.mkdir('wrong_notweed')
            os.mkdir('correct_weed')
            os.mkdir('wrong_weed')
            os.mkdir('pred_weed')
        except:
            print('Directory exists.')
        
        # Change to Test directory
        os.chdir(test_dir)
        
        ClassAssigned = []
        
        # Specify list for variables
        filenames_correct_notweed = []
        filenames_wrong_notweed = []
        filenames_correct_weed = []
        filenames_wrong_weed = []
        
        tmp_idx = []
        for i in range(int(test_generator.n/test_generator.batch_size)+1):
            # batch is a tuple (img=>3Darray, label=>1Darray)
            batch = test_generator.next()
            
            # Obtain index of batch and get batch filenames - only works if test_gen shuffle is set to False
            print(test_generator.batch_index)
            idx = (test_generator.batch_index - 1) * test_generator.batch_size
            tmp_idx.append(test_generator.batch_index) 
            if test_generator.batch_index == 0:
                print(test_generator.batch_size)
                print(max(tmp_idx))
                idx = max(tmp_idx) * test_generator.batch_size
            print("idx start : ", idx, "idx end: ", idx+len(batch[0]))
            batch_names = test_generator.filenames[idx: idx+len(batch[0])] #test_generator.filenames[idx:idx+test_generator.batch_size]
            
            print("The length of batch_names is: ", len(batch_names))
            print("The length of batch is: ", len(batch[0]))
            for j in range(len(batch[0])):    
                #count+=1
                img = batch[0][j]
                imgPred = img.reshape((1,) + img.shape)
                img = img*255
                sc = model.predict(imgPred).tolist().pop().pop()
                lb = batch[1][j]
                if lb == 0 and sc <= opt_cutoff:
                    ClassAssigned.append(0)
                    filenames_correct_notweed.append(batch_names[j])
                elif lb == 0 and sc > opt_cutoff:
                    ClassAssigned.append(1)
                    filenames_wrong_notweed.append(batch_names[j])
                elif lb == 1 and sc > opt_cutoff:
                    ClassAssigned.append(1)
                    filenames_correct_weed.append(batch_names[j])
                elif lb == 1 and sc <= opt_cutoff:
                    ClassAssigned.append(0)
                    filenames_wrong_weed.append(batch_names[j])

        #1: Create filenames dict
        #  test_generator.filenames
        filenames = {'correct_notweed': filenames_correct_notweed,
                    'wrong_notweed': filenames_wrong_notweed,
                    'correct_weed': filenames_correct_weed,
                    'wrong_weed': filenames_wrong_weed,
                    'pred_weed': filenames_correct_weed + filenames_wrong_notweed}

    # del model

    #########################################################################################
    ################################### Create Shape File ###################################
    #########################################################################################
    

    #2: Find Geo info  lon - 100.xx  lat - 0.8
    print("Thi is the directory", raw_dir+k)
    TLx2, TLy2, radPerPixX, radPerPixY = load_georef(raw_dir, k)
    
    os.chdir(saveDir)
    
    #3: Create shape file for each item
    for key, tmp in filenames.items():
        #tmp = filenames[i]
    
        #3.1: Recover x and y index
        coords_cnt = []
        petak = []
        for i in tmp:
            print(i)
            ind = i.split("_")
    
            ind_petak = ind[2]
            petak.append(ind_petak)
    
            #7242:7455,5538:5751 or 7242:7455:5538,5751
            ind_coord = ind[4]
            ind_coord = ind_coord.replace(":", ",")
            x_start, x_end, y_start, y_end = ind_coord.split(",")
    
    
            cnt = np.array([[int(y_start), int(x_start)],
                            [int(y_end), int(x_start)],
                            [int(y_end), int(x_end)],
                            [int(y_start), int(x_end)]])
            #print(cnt)
    
            coords_cnt.append(cnt)
        
        if(len(coords_cnt)>1):
            #3.2: Create the shape file
            out_shp = key + '/' + key + "_" + modelname + "_" + ind_petak + "_Wtreeweed.shp" # name of shape file
            write_shapefile(tmp, coords_cnt, out_shp, TLx2, TLy2, radPerPixX, radPerPixY)

        
