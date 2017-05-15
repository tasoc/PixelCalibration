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
import itertools
from polynomial import polynomial
from bottleneck import replace, nanmedian

# Packages generated from protobufs:
from twodblack_pb2 import TwoDBlackModel
from flatfield_pb2 import FlatFieldModel
from CadencePixelData_pb2 import PixelData, PixelHeader

class CadenceImage:
	def __init__(self, protobuf):

		# Load the protobuf file:
		pd = PixelData()
		ph = PixelHeader()
		with gzip.open(protobuf, 'rb') as fid:
			d = fid.read()
			pd.ParseFromString(d)
			ph.ParseFromString(d)

		#
		self.PixelHeader = ph
		self.camera = int(self.PixelHeader.camera_number)
		self.ccd = int(self.PixelHeader.ccd_number)

		#print(ph)
		#print(pd.target_data)
		#print(pd.collateral_data)

		# Store pixel data as 1D arrays:
		self.target_data = np.array(pd.target_data[1:], dtype='float64') # FIXME: Why is there a leading zero?!
		self.collateral_data = np.array(pd.collateral_data[1:], dtype='float64') # FIXME: Why is there a leading one?!

		# Replace missing data with NaN:
		replace(self.target_data, 0xFFFFFFFF, np.nan)
		replace(self.collateral_data, 0xFFFFFFFF, np.nan)

		# Properties which will be filled out later:
		self.dark = None

		# TODO: All the following is actually common to all dataset with same target_pixel_table_id
		# Find rows and columns on the 2D CCD matching the 1D pixel data:
		target_pixel_table_id = int(self.PixelHeader.target_pixel_table_id)
		target_pixel_table = etree.parse('test_data/%04d-target-pixel-table.xml' % (target_pixel_table_id, )).getroot()

		Npixels = len(self.target_data)
		self.rows = np.zeros(Npixels, dtype='int32')
		self.columns = np.zeros(Npixels, dtype='int32')
		self.outputs = np.zeros(Npixels, dtype='str')
		for pixel in target_pixel_table.xpath('./ccd[@cameraNumber="%d"][@ccdNumber="%d"]/pixel' % (self.camera, self.ccd)):
			index = int(pixel.get('index')) - 1
			column = int(pixel.get('column'))
			self.rows[index] = int(pixel.get('row'))
			self.columns[index] = column
			# Figure out what CCD outputs each column corresponds to:
			if column >= 1581:
				self.outputs[index] = 'D'
			elif column >= 1069:
				self.outputs[index] = 'C'
			elif column >= 557:
				self.outputs[index] = 'B'
			elif column >= 45:
				self.outputs[index] = 'A'

		# Convert the row and column addresses to indicies in the flatfield and 2d black images:
		self.index_columns = self.columns - 1
		self.index_rows = 512 - self.rows # FIXME: 2078 instead of 512

		#print(self.outputs)
		#print(self.rows, self.columns)
		#print(self.index_rows, self.index_columns)

		Ncollateral = len(self.collateral_data)
		collateral_rows = np.zeros(Ncollateral, dtype='int32')
		collateral_columns = np.zeros(Ncollateral, dtype='int32')

		collateral_pixel_table_id = int(self.PixelHeader.collateral_pixel_table_id)
		collateral_pixel_table = etree.parse('test_data/%04d-collateral-pixel-table.xml' % (collateral_pixel_table_id, )).getroot()
		for pixel in collateral_pixel_table.xpath('./ccd[@cameraNumber="%d"][@ccdNumber="%d"]/pixel' % (self.camera, self.ccd)):
			index = int(pixel.get('index')) - 1
			collateral_rows[index] = int(pixel.get('row'))
			collateral_columns[index] = int(pixel.get('column'))

		unique_collateral_columns = np.unique(collateral_columns)
		Ncolcolumns = len(unique_collateral_columns)
		self.masked_smear = np.full((10, Ncolcolumns), np.nan, dtype='float64')
		self.virtual_smear = np.full((10, Ncolcolumns), np.nan, dtype='float64')
		for index, (row, column) in enumerate(zip(collateral_rows, collateral_columns)):
			index_column = np.where(column == unique_collateral_columns)[0]
			if column >= 2093 or column <= 44:
				# Virtual columns or Serial register columns
				pass
			elif row >= 2069:
				# Virtual rows
				index_row = (2078 - row)
				self.virtual_smear[index_row, index_column] = self.collateral_data[index]
			elif row >= 2059:
				# Smear rows
				index_row = (2068 - row)
				self.masked_smear[index_row, index_column] = self.collateral_data[index]
			elif row >= 2049:
				# Buffer rows
				pass
			else:
				print("Invalid collateral pixel: (%d,%d)" % (row, column))

		self.collateral_columns = unique_collateral_columns

		print(self.collateral_columns)
		print(self.masked_smear)
		print(self.virtual_smear)


	def seperate_to_targets(self):

		Ntime = 1
		time_index = 0
		byte_size = 64//8

		target_list = ET.parse('test_data/target-list.xml').getroot()
		for target in target_list.findall('./ccd/target'):
			tic = int(target.get('catId'))
			print(tic)

			pixel_index = np.array([int(p.get('index'))-1 for p in target.findall('./pixels/pixel')], dtype='int64')

			img = self.target_data[pixel_index]
			rows = self.rows[pixel_index]
			columns = self.columns[pixel_index]

			Nrows = rows.max() - rows.min() + 1
			Ncols = columns.max() - columns.min() + 1

			if not os.path.exists('cache/%09d.memmap' % tic):
				a = np.memmap(filename='cache/%09d.memmap' % tic, mode='w+', dtype='float64', shape=(Nrows, Ncols, Ntime))
				del a

			img_reshaped = np.full((Nrows, Ncols), np.nan, dtype='float64')
			for i,r in enumerate(range(rows.min(), rows.max()+1)):
				for j,c in enumerate(range(columns.min(), columns.max()+1)):
					indx = (rows == r) & (columns == c)
					if np.any(indx):
						img_reshaped[i, j] = img[indx]

			print(img_reshaped)

			b = np.memmap(filename='cache/%09d.memmap' % tic, mode='r+', dtype='float64', shape=(Nrows, Ncols, 1), offset=time_index*Nrows*Ncols*byte_size)
			b[:,:,0] = img_reshaped
			del b



class TESS_CAL:

	def __init__(self, coadds=10):

		self.logger = logging.getLogger(__name__)
		self.logger.info("Starting calibration module")

		self.coadds = coadds
		self.exposure_time = 1.96 # seconds
		self.frametransfer_time = 0.04 # seconds
		self.readout_time = 0.5 # seconds

		# Load 2D black images from protobufs:
		self.logger.info("Loading 2D Black model...")
		twodblack_model = TwoDBlackModel()
		with gzip.open('test_data/twodblack.pb.gz', 'rb') as fid:
			twodblack_model.ParseFromString(fid.read())

		self.twodblack_image = {}
		for ff in twodblack_model.images:
			Nrows = ff.ccd_rows_range.upperLimit - ff.ccd_rows_range.lowerLimit + 1
			Ncols = ff.ccd_columns_range.upperLimit - ff.ccd_columns_range.lowerLimit + 1
			self.twodblack_image[ff.camera_number, ff.ccd_number] = self.coadds * np.array(ff.image_data, dtype='int32').reshape((Nrows, Ncols))

		# Load the gain- and linearity models from XML files:
		self.logger.info("Loading gain and linearity models...")
		gain_model = ET.parse('test_data/gain.xml').getroot()
		linearity_model = ET.parse('test_data/linearity.xml').getroot()

		#
		self.linearity_gain_model = {}
		for channel in linearity_model.findall('./channel'):
			camera = int(channel.get('cameraNumber'))
			ccd = int(channel.get('ccdNumber'))
			output = channel.get('ccdOutput')

			gain = gain_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd:d}'][@ccdOutput='{output}']".format(
				camera=camera,
				ccd=ccd,
				output=output
			)).get('gainElectronsPerDN')

			self.linearity_gain_model[camera, ccd, output] = {
				'gain': float(gain),
				'linpoly': polynomial(channel.find("./linearityPoly"))
			}

		# Load flatfields from protobuf files:
		self.logger.info("Loading flatfield models...")
		flat_field_model = FlatFieldModel()
		with gzip.open('test_data/flatfield.pb.gz', 'rb') as fid:
			flat_field_model.ParseFromString(fid.read())

		self.flatfield_image = {}
		for ff in flat_field_model.images:
			Nrows = ff.ccd_rows_range.upperLimit - ff.ccd_rows_range.lowerLimit + 1
			Ncols = ff.ccd_columns_range.upperLimit - ff.ccd_columns_range.lowerLimit + 1
			self.flatfield_image[ff.camera_number, ff.ccd_number] = np.array(ff.image_data, dtype='float64').reshape((Nrows, Ncols))


	def twodblack(self, img):
		"""2D black-level correction.

		TODO:
			 - Correct collateral pixels as well.
		"""
		self.logger.info("Doing 2D black correction...")
		img.target_data -= self.twodblack_image[img.camera, img.ccd][img.index_rows, img.index_columns]
		#img.collateral_data -= self.twodblack_image[img.camera, img.ccd][img.index_collateral_columns, img.index_collateral_columns]
		return img


	def linearity_gain(self, img):
		"""CCD Linearity/gain correction."""

		self.logger.info("Doing gain/linearity correction...")

		for output in np.unique(img.outputs):
			gain_linearity_model = self.linearity_gain_model[img.camera, img.ccd, output]
			gain = gain_linearity_model['gain']
			linpoly = gain_linearity_model['linpoly']

			# Evaluate the polynomial and multiply the image values by it:
			DN0 = img.target_data[img.outputs == output]/self.coadds
			img.target_data[img.outputs == output] = DN0 * linpoly(DN0)
			img.target_data[img.outputs == output] *= gain * self.coadds

		return img


	def smear(self, img):
		"""CCD dark current and smear correction.

		TODO:
			 - Should we weight everything with the number of rows used in masked vs virtual regions?
			 - Should we take self.frametransfer_time into account?
			 - Cosmic ray rejection requires images before and after in time?
		"""
		self.logger.info("Doing smear correction...")

		# Remove cosmic rays in collateral data:
		# TODO: Can cosmic rays also show up in virtual pixels? If so, also include img.virtual_smear
		#index_collateral_cosmicrays = cosmic_rays(img.masked_smear)
		index_collateral_cosmicrays = np.zeros_like(img.masked_smear, dtype='bool')
		img.masked_smear[index_collateral_cosmicrays] = np.nan

		# Average the masked and virtual smear across their rows:
		masked_smear = nanmedian(img.masked_smear, axis=0)
		virtual_smear = nanmedian(img.virtual_smear, axis=0)

		# Estimate dark current:
		# TODO: Should this be self.frametransfer_time?
		fdark = nanmedian( masked_smear - virtual_smear * (self.exposure_time + self.readout_time) / self.exposure_time )
		img.dark = fdark # Save for later use
		self.logger.info('Dark current: %f', img.dark)
		if np.isnan(fdark):
			fdark = 0

		# Correct the smear regions for the dark current:
		masked_smear -= fdark
		virtual_smear -= fdark * (self.exposure_time + self.readout_time) / self.exposure_time

		# Weights from number of pixels in different regions:
		Nms = np.sum(~np.isnan(img.masked_smear), axis=0)
		Nvs = np.sum(~np.isnan(img.virtual_smear), axis=0)
		c_ms = Nms/np.maximum(Nms + Nvs, 1)
		c_vs = Nvs/np.maximum(Nms + Nvs, 1)

		# Weights as in Kepler where you only have one row in each sector:
		#g_ms = ~np.isnan(masked_smear)
		#g_vs = ~np.isnan(virtual_smear)
		#c_ms = g_ms/np.maximum(g_ms + g_vs, 1)
		#c_vs = g_vs/np.maximum(g_ms + g_vs, 1)

		# Estimate the smear for all columns, taking into account
		# that some columns could be missing:
		replace(masked_smear, np.nan, 0)
		replace(virtual_smear, np.nan, 0)
		fsmear = c_ms*masked_smear + c_vs*virtual_smear

		# Correct the science pixels for dark current and smear:
		img.target_data -= fdark
		for k,col in enumerate(img.collateral_columns):
			img.target_data[img.columns == col] -= fsmear[k]

		return img


	def flatfield(self, img):
		"""CCD flat-field correction."""
		self.logger.info("Doing flatfield correction...")
		img.target_data /= self.flatfield_image[img.camera, img.ccd][img.index_rows, img.index_columns]
		return img


	def to_counts_per_second(self, img):
		img.target_data /= self.exposure_time * self.coadds
		return img


	def calibrate(self, img):
		"""Perform all calibration steps in sequence."""
		img = self.twodblack(img)
		img = self.linearity_gain(img)
		img = self.smear(img)
		img = self.flatfield(img)
		img = self.to_counts_per_second(img)
		return img


	def test(self):
		self.logger.info("Running test...")

		# Plot the 2D black images:
		for camera in (1,2,3,4):
			fig, axes = plt.subplots(nrows=2, ncols=2)
			for ccd,ax in zip((1,2,3,4), axes.flat):
				s = self.twodblack_image[camera, ccd].shape
				im = ax.imshow(self.twodblack_image[camera, ccd], origin='lower', extent=[1, s[0], 1, s[1]], vmin=0, vmax=10*self.coadds)
				ax.set_title('CCD %d' % ccd)
			fig.colorbar(im, ax=axes.ravel().tolist())
			fig.savefig('test_data/2dblack_camera%d.png' % camera)

		# Plot the flatfield images:
		for camera in (1,2,3,4):
			fig, axes = plt.subplots(nrows=2, ncols=2)
			for ccd,ax in zip((1,2,3,4), axes.flat):
				s = self.twodblack_image[camera, ccd].shape
				im = ax.imshow(self.flatfield_image[camera, ccd], cmap=plt.get_cmap('coolwarm'), origin='lower', extent=[1, s[0], 1, s[1]], vmin=0.95, vmax=1.05)
				ax.set_title('CCD %d' % ccd)
			fig.colorbar(im, ax=axes.ravel().tolist())
			fig.savefig('test_data/flatfield_camera%d.png' % camera)

		# Plot the linearity polynomials:
		x = np.linspace(0, 100000, 500)
		fig = plt.figure()
		for camera in (1,2,3,4):
			ax = fig.add_subplot(2, 2, camera)
			ax.set_title('Camera %d' % camera)
			for ccd, output in itertools.product((1,2,3,4), ('A','B','C','D')):
				poly = self.linearity_gain_model[camera, ccd, output]['linpoly']
				ax.plot(x, poly(x), label='%d %s' % (ccd, output))
			ax.legend()
		fig.savefig('test_data/linpoly.png')

		# Load test image:
		img = CadenceImage('test_data/pixel-data.pb.gz')

		# Run calibration on the test image:
		img_cal = self.calibrate(img)

		img_cal.seperate_to_targets()

		# Load the true image:
		img_true = np.load('test_data/true-image.npy')

		# Plot the difference between the true image and the recovered one:
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.axhline(0, color='k', ls='--')
		ax.scatter(np.arange(1, len(img.target_data)+1), img_true.flatten() - img_cal.target_data)
		ax.set_xlabel('Pixel index')
		ax.set_ylabel('True - Calibrated')
		fig.savefig('test.png')

		self.logger.info("Tests done.")
		plt.show()


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
	cal = TESS_CAL()
	cal.test()

	"""
	files = ('test_data/pixel-data.pb.gz', )
	for img_file in files: # TODO: Should be run in parallel
		img = CadenceImage(img_file)
		img_cal = cal.calibrate(img)
		img_cal.seperate_to_targets()
		# TODO: Store the calibrated data somewhere?!

	# TODO: Put all the calibrated data pertaining to one file together in one FITS file
	"""