from subprocess import call
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import settings


data_zip_path = os.path.join(settings._osr_pubfig_root, "relative_attributes_v2.zip")
data_url = "https://filebox.ece.vt.edu/~parikh/relative_attributes/relative_attributes_v2.zip"

# Downloading the data zip and extracting it
call(["wget",
      "--continue",  # do not download things again
      "--tries=0",  # try many times to finish the download
      "--output-document=%s" % data_zip_path,  # save it to the appropriate place
      data_url])

call(["unzip -d %s %s" % (settings._osr_pubfig_root, data_zip_path)], shell=True)

osr_images_root_path = os.path.join(settings._osr_pubfig_root, 'relative_attributes', 'osr')
osr_images_zip_path = os.path.join(osr_images_root_path, 'spatial_envelope_256x256_static_8outdoorcategories.zip')
osr_images_url = "http://people.csail.mit.edu/torralba/code/spatialenvelope/spatial_envelope_256x256_static_8outdoorcategories.zip"

# Downloading the data zip and extracting it
call(["wget",
      "--continue",  # do not download things again
      "--tries=0",  # try many times to finish the download
      "--output-document=%s" % osr_images_zip_path,  # save it to the appropriate place
      osr_images_url])

# TODO: Rename folder to images
call(["unzip -d %s %s" % (osr_images_root_path, osr_images_zip_path)], shell=True)
