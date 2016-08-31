from subprocess import call
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import settings


data_zip_path = os.path.join(settings.model_root, "vgg16.pkl")
data_url = "https://s3.amazonaws.com/lasagne/recipes/pretrained/imagenet/vgg16.pkl"

# Downloading the data zip and extracting it
call(["wget",
      "--continue",  # do not download things again
      "--tries=0",  # try many times to finish the download
      "--output-document=%s" % data_zip_path,  # save it to the appropriate place
      data_url])
