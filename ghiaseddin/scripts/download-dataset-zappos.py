from subprocess import call
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import settings


data_zip_path = os.path.join(settings.zappos_root, "ut-zap50k-data.zip")
data_url = "http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-data.zip"

imgs_zip_path = os.path.join(settings.zappos_root, "ut-zap50k-images.zip")
imgs_url = "http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip"

# Downloading the data zip and extracting it
call(["wget",
      "--continue",  # do not download things again
      "--tries=0",  # try many times to finish the download
      "--output-document=%s" % data_zip_path,  # save it to the appropriate place
      data_url])

call(["unzip -d %s %s" % (settings.zappos_root, data_zip_path)], shell=True)

# Downloading the images zip and extracting it
call(["wget",
      "--continue",
      "--tries=0",
      "--output-document=%s" % imgs_zip_path,
      imgs_url])

call(["unzip -d %s %s" % (settings.zappos_root, imgs_zip_path)], shell=True)
