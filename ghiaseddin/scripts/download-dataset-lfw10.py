from subprocess import call
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import settings


data_zip_path = os.path.join(settings.lfw10_root, "LFW10.zip")
data_url = "http://cvit.iiit.ac.in/images/Projects/relativeParts/LFW10.zip"

# Downloading the data zip and extracting it
call(["wget",
      "--continue",  # do not download things again
      "--tries=0",  # try many times to finish the download
      "--output-document=%s" % data_zip_path,  # save it to the appropriate place
      data_url])

call(["unzip -d %s %s" % (settings.lfw10_root, data_zip_path)], shell=True)
