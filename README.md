# Python Real Estate

This project shows how to create a model for predicting house prices and exporting some data for later use inside a Tensorflow.js application

You can read/see more about this in:

* [House price prediction 1/4: Using Keras/Tensorflow and python - Youtube Video](https://youtu.be/HSxFWWbt9S0)
* [House price prediction 1/4: Using Keras/Tensorflow and python - Blog Post](https://www.dlighthouse.co/2019/04/tensorflow-keras-predict-house-prices-1-4.html)

## Important files to run

* predict.py
* train_model.py
* share_preprocessing_tfjs.py

## Tensorflowjs Converter

### Building the image using docker
```
    docker build -t tf-converter .
```

### Running the converter using docker
```
    docker run -it --rm --name tf-converter -v "$(pwd)":/workdir tf-converter --input_format keras ./model/-inputsscaled-outputsscaled-categorical/model.h5 ./shared/model
```

### Install the converter and run it without docker
```
    pip install tensorflowjs

    tensorflowjs_converter --input_format keras \
                        ./model/-inputsscaled-outputsscaled-categorical/model.h5 \
                        ./shared/model
```
## More resources
* [Tensorflow.js](https://js.tensorflow.org/)
* [TensorFlow Docker Images](https://hub.docker.com/r/tensorflow/tensorflow/)
* [Tensorflow.js Converter](https://github.com/tensorflow/tfjs-converter)
* [Keras Loss Functions](https://keras.io/losses/)
* [Keras Metrics Functions](https://keras.io/metrics/)
* [Display Deep Learning Model Training History in Keras](https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/)
* [Keras Model Checkpoints](https://keras.io/callbacks/#modelcheckpoint)
* [Keras Early Stopping](https://keras.io/callbacks/#earlystopping)
