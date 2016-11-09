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
import logging
import gzip

# Packages generated from protobufs:
import .twodblack_pb2
import .flatfield_pb2

class TESS_CAL:

	def __init__(self, coadds=10):
		
		self.logger = logging.getLogger(__name__)
		
		self.coadds = coadds
		
		# Load 2D black images from protobufs:
		twodblack_model = twodblack_pb2.TwoDBlackModel()
		
		# Read the existing address book.
		with gzip.open('data/twodblack.pb.gz', 'rb') as fid:
			twodblack_model.ParseFromString(fid.read())
		
		self.twodblack = {}
		for ff in twodblack_model.CcdImage:
			self.twodblack[ff.camera_number] = self.coadds * np.array(ff.image_data)
		
		# Load the gain- and linearity models from XML files:
		gain_model = ET.parse('data/gain_model.xml').getroot()
		linearity_model = ET.parse('data/linearity_model.xml').getroot()
		
		# 
		self.linearity_gain_model = {}
		for channel in linearity_model.findall('./channel'):
			linpoly = channel.find("./linearityPoly")

			camera = int(channel.get('cameraNumber'))
			ccd = int(channel.get('ccdNumber'))
			output = channel.get('ccdOutput')

			#
			coeffs = np.zeros(int(linpoly.get('order'))+1, 'float64')
			for k,coeff in enumerate(linpoly.findall('./coeffs/coeff')):
				coeffs[k] = coeff.get('value')

			gain = gain_model.find("./channel[@cameraNumber='{camera:d}'][@ccdNumber='{ccd}'][@ccdOutput='{output}']".format(
				camera=camera,
				ccd=ccd,
				output=output
			)).get('gainElectronsPerDN')

			self.linearity_gain_model[camera, ccd, output] = {
				'gain': float(gain),
				'coeffs': coeffs,
				'offsetx': float(linpoly.get('offsetx')),
				'scalex': float(linpoly.get('scalex')),
				'originx': float(linpoly.get('originx'))
			}

		# Load flatfields from protobuf files:
		flat_field_model = flatfield_pb2.FlatFieldModel()

		# Read the existing address book.
		with gzip.open('data/flatfield.pb.gz', 'rb') as fid:
			flat_field_model.ParseFromString(fid.read())

		self.flatfield = {}
		for ff in flat_field_model.CcdImage:
			self.flatfield[ff.camera_number] = np.array(ff.image_data)


	def twodblack(self, img):
		"""2D black-level correction."""
	
		img -= self.twodblack[camera]
		return img


	def linearity_gain(self, img):
		"""CCD Linearity/gain correction."""

		# TODO: Extract these from img-header?
		camera = 1
		ccd = 1
		output = 1

		# Load gain model
		linpoly = self.linearity_gain_model[camera, ccd, output]

		# Evaluate the polynomial and multiply the image values by it:
		img *= linpoly['gain'] * np.polyval(linpoly['coeffs'], linpoly['offsetx'] + linpoly['scalex'] * (img/self.coadds - linpoly['originx']))

		return img


	def smear(self, img):
		"""CCD smear correction."""
		return img

	def flatfield(self, img):
		"""CCD flat-field correction."""

		img /= self.flatfield[camera]
		return img
	

	def calibrate(self, img):
		"""Perform all calibration steps in sequence."""
		
		img = self.twodblack(img)
		img = self.linearity_gain(img)
		img = self.smear(img)
		img = self.flatfield(img)
		return img
	
	def test(self):
		pass


if __name__ == '__main__':
	pass

	#cal = TESS_CAL()
	#img_cal = cal.calibrate(img)
