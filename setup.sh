#!/bin/bash

# Install Python packages
pip install absl-py==1.3.0
pip install aiohttp==3.8.1
pip install aiosignal==1.2.0
pip install astunparse==1.6.3
pip install async-timeout==4.0.2
pip install attrs==21.4.0
pip install blinker==1.4
pip install brotlipy==0.7.0
pip install cachetools==4.2.2
pip install certifi==2022.9.24
pip install cffi==1.15.1
pip install charset-normalizer==2.0.4
pip install click==8.0.4
pip install cryptography==38.0.1
pip install dataclasses==0.8
pip install ffmpeg==4.3
pip install flatbuffers==2.0.0
pip install frozenlist==1.2.0
pip install gast==0.5.3
pip install google-auth==2.6.0
pip install google-auth-oauthlib==0.4.4
pip install google-pasta==0.2.0
pip install grpcio==1.42.0
pip install h5py==3.7.0
pip install idna==3.4
pip install importlib-metadata==4.11.3
pip install joblib==1.1.1
pip install keras==2.9.0
pip install keras-preprocessing==1.1.2
pip install markdown==3.3.4
pip install mkl-service==2.4.0
pip install mkl_fft==1.3.1
pip install mkl_random==1.2.2
pip install multidict==6.0.2
pip install numpy==1.21.5
pip install oauthlib==3.2.1
pip install opt_einsum==3.3.0
pip install packaging==21.3
pip install pillow==9.2.0
pip install protobuf==3.20.1
pip install pyasn1==0.4.8
pip install pyasn1-modules==0.2.8
pip install pycparser==2.21
pip install pyjwt==2.4.0
pip install pyopenssl==22.0.0
pip install pyparsing==3.0.9
pip install pysocks==1.7.1
pip install python-flatbuffers==2.0
pip install pytorch==1.13.0+cu117
pip install pytorch-cuda==11.7
pip install pytorch-mutex==1.0
pip install pyyaml==6.0
pip install requests==2.28.1
pip install requests-oauthlib==1.3.0
pip install rsa==4.7.2
pip install scikit-learn==1.1.3
pip install scipy==1.7.3
pip install setuptools==65.5.0
pip install six==1.16.0
pip install tensorboard==2.9.0
pip install tensorboard-data-server==0.6.0
pip install tensorboard-plugin-wit==1.8.1
pip install tensorflow==2.9.1
pip install tensorflow-estimator==2.9.0
pip install termcolor==1.1.0
pip install threadpoolctl==2.2.0
pip install torchvision==0.14.0+cu117
pip install typing_extensions==4.3.0
pip install urllib3==1.26.12
pip install werkzeug==2.0.3
pip install wheel==0.37.1
pip install wrapt==1.14.1
pip install yarl==1.8.1
pip install zipp==3.8.0
pip install openslide-python
pip install progress

# Install system packages
apt-get update
apt-get install -y openslide-tools
apt-get install -y python-openslide
sudo apt-get install expect

