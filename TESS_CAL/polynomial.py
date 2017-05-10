#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class polynomial():
	def __init__(self, xmlpoly):

		self.type = xmlpoly.get('type')
		self.offsetx = float(xmlpoly.get('offsetx'))
		self.scalex = float(xmlpoly.get('scalex'))
		self.originx = float(xmlpoly.get('originx'))

		self.coeffs = np.zeros(int(xmlpoly.get('order'))+1, 'float64')
		for k,coeff in enumerate(xmlpoly.findall('./coeffs/coeff')):
			self.coeffs[k] = coeff.get('value')

	def __call__(self, x):
		if self.type == 'standard':
			return np.polyval(self.coeffs, self.offsetx + self.scalex * (x - self.originx))
		elif self.type == 'legendre':
			return np.polynomial.legendre.legval(self.offsetx + self.scalex * (x - self.originx), self.coeffs)
		elif self.type == 'NotScaled':
			#return np.polyval(self.coeffs, x)
			raise NotImplementedError
		else:
			raise NotImplementedError