#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:39:54 2016

@author: Rasmus Handberg <rasmush@phys.au.dk>
"""

from __future__ import division, with_statement, print_function, absolute_import
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import xml.etree.ElementTree as ET
from lxml import etree
import logging
import gzip
from polynomial import polynomial

# Packages generated from protobufs:
from twodblack_pb2 import TwoDBlackModel
from flatfield_pb2 import FlatFieldModel
from CadencePixelData_pb2 import PixelData, PixelHeader

class CadenceImage:
	def __init__(self, protobuf):

		# Load the protobuf file:
		data = PixelData()
		self.PixelHeader = PixelHeader()
		#with gzip.open(protobuf, 'rb') as fid:
		#	data = fid.read()
		#	PixelData.ParseFromString(data)
		#	PixelHeader.ParseFromString(data)

		#
		self.camera = int(self.PixelHeader.camera_number)
		self.ccd = int(self.PixelHeader.ccd_number)
		self.output = None

		# Store pixel data as 1D arrays:
		self.target_data = np.array(data.target_data, dtype='int32')
		self.collateral_data = np.array(data.collateral_data, dtype='int32')

		# Find rows and columns on the 2D CCD matching the 1D pixel data:
		target_pixel_table_id = int(self.PixelHeader.target_pixel_table_id)
		target_pixel_table = etree.parse('data/target-pixel-table.xml' ).getroot() # % (target_pixel_table_id, )

		self.rows = np.zeros(len(self.target_data), dtype='int32')
		self.columns = np.zeros(len(self.target_data), dtype='int32')
		pixels = target_pixel_table.xpath('./ccd[@cameraNumber="%d"][@ccdNumber="%d"]/pixel' % (self.camera, self.ccd))
		for pixel in pixels:
			index = int(pixel.get('index'))
			self.rows[index] = int(pixel.get('row'))
			self.columns[index] = int(pixel.get('column'))


	def save_to_lib(self):
		pass

	def seperate_to_targets(self):

		target_list = ET.parse('data/target-list.xml').getroot()
		for target in target_list.findall('./ccd/target'):
			tic = int(target.get('catId'))

			pixel_index = np.array([int(p.get('index')) for p in target.findall('./pixels/pixel')], dtype='int64')

			img = self.target_data[pixel_index]
			rows = self.rows[pixel_index]
			columns = self.columns[pixel_index]

			img_reshaped = np.empty((len(rows), len(columns))) + np.NaN
			for i in np.arange(rows.min(), rows.max()):
				for j in np.arange(columns.min(), columns.max()):
					indx = (rows == i) & (columns == j)
					if indx:
						img_reshaped[i, j] = img[indx]



class TESS_CAL:

	def __init__(self, coadds=10):

		self.logger = logging.getLogger(__name__)
		self.logger.info("Starting calibration module")

		self.coadds = coadds

		# Load 2D black images from protobufs:
		self.logger.info("Loading 2D Black model...")
		twodblack_model = TwoDBlackModel()
		"""
		with gzip.open('data/twodblack.pb.gz', 'rb') as fid:
			twodblack_model.ParseFromString(fid.read())
		"""

		self.twodblack_image = {}
		for ff in twodblack_model.images:
			self.twodblack_image[ff.camera_number] = self.coadds * np.array(ff.image_data)

		# Load the gain- and linearity models from XML files:
		self.logger.info("Loading gain and linearity models...")
		gain_model = ET.parse('data/gain.xml').getroot()
		linearity_model = ET.parse('data/linearity.xml').getroot()

		#
		self.linearity_gain_model = {}
		for channel in linearity_model.findall('./channel'):
			camera = int(channel.get('cameraNumber'))
			ccd = int(channel.get('ccdNumber'))
			output = channel.get('ccdOutput')

			gain = gain_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd}'][@ccdOutput='{output}']".format(
				camera=camera,
				ccd=ccd,
				output=output
			)).get('gainElectronsPerDN')

			self.linearity_gain_model[camera, ccd, output] = {
				'gain': float(gain),
				'linpoly': polynomial(channel.find("./linearityPoly"))
			}

		# Load flatfields from protobuf files:
		flat_field_model = FlatFieldModel()
		#with gzip.open('data/flatfield.pb.gz', 'rb') as fid:
		#	flat_field_model.ParseFromString(fid.read())

		self.flatfield = {}
		for ff in flat_field_model.images:
			self.flatfield[ff.camera_number] = np.array(ff.image_data)

	def twodblack(self, img):
		"""2D black-level correction."""
		img.target_data -= self.twodblack_image[img.camera][img.rows, img.columns]
		return img


	def linearity_gain(self, img):
		"""CCD Linearity/gain correction."""

		# Load gain model
		gain_linearity_model = self.linearity_gain_model[img.camera, img.ccd, img.output]
		gain = gain_linearity_model['gain']
		linpoly = gain_linearity_model['linpoly']

		# Evaluate the polynomial and multiply the image values by it:
		img.target_data *= gain * linpoly(img.target_data/self.coadds)

		return img


	def smear(self, img):
		"""CCD smear correction."""
		# TODO: Find collateral data
		# TODO: Cosmic Ray removal in collateral data
		# TODO: Perform smear!!!
		return img

	def flatfield(self, img):
		"""CCD flat-field correction."""
		img.target_data /= self.flatfield[img.camera][img.rows, img.columns]
		return img

	def calibrate(self, img):
		"""Perform all calibration steps in sequence."""
		#img = self.twodblack(img)
		img = self.linearity_gain(img)
		img = self.smear(img)
		img = self.flatfield(img)
		return img

	def test(self):
		pass


if __name__ == '__main__':

	logging_level = logging.INFO

	# Configure the standard console logger
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	console = logging.StreamHandler()
	console.setFormatter(formatter)

	# Configure this logger
	logger = logging.getLogger(__name__)
	logger.addHandler(console)
	logger.setLevel(logging_level)

	logger.info("What are we doing?")
	files = ('data/pixel-data.pb.gz', )
	cal = TESS_CAL()
	for img_file in files: # TODO: Should be run in parallel
		img = CadenceImage(img_file)
		img_cal = cal.calibrate(img)
		img_cal.seperate_to_targets()
		# TODO: Store the calibrated data somewhere?!

	# TODO: Put all the calibrated data pertaining to one file together in one FITS file
