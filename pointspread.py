#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Point spread function

Created on Thu Sep 28 18:51:55 2017

@author: tomoya
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

def get_coords(lonlatdata):
    """ calculate interstation distance and angle
    
    引数 : 
        lonlatdata: ccffilename,lat,lon
    """
    datfile = lonlatdata['sta']
    lat = lonlatdata['x']
    lon = lonlatdata['y']
    lat_center = np.mean(lat).squeeze()
    lon_center = np.mean(lon).squeeze()
    
    distance = np.sqrt(np.square(lat-lat_center) + np.square(lon-lon_center))
    angle = np.arctan2((lon-lon_center), (lat-lat_center))

    outcx = distance*np.cos(angle)*0.001 ## m to km
    outcy = distance*np.sin(angle)*0.001 ## m to km

    return [datfile, outcx, outcy]
    

def get_psf_wavenumber(cartesianx, cartesiany, klim, kstep):
    """ calculate array response function in wavenumber domain
    
    引数 :
        coords : coorinates xy(km)
        klim : limit of wavenumber
        kstep : step number of wavenumber
    """
    
    # setting of wavenumber range
    kxmin = -klim
    kxmax = klim
    kymin = -klim
    kymax = klim
    Nb_points = 201
    nkx = np.linspace(kxmin,kxmax,Nb_points)
    nky = nkx 
    
    #initialization
    transff = np.zeros((nkx.shape[0],nky.shape[0]))
       
    for ii, kx in enumerate(np.arange(kxmin, kxmax+kstep/10., kstep)):
        for jj, ky in enumerate(np.arange(kymin, kymax+kstep/10., kstep)):
            _sum = 0j
            for k in range(cartesianx.shape[0]):
                _sum += np.exp(complex(0., -1*(cartesianx[k]*kx + cartesiany[k]*ky)))
            transff[ii, jj] = abs(_sum) ** 2
    transff /= transff.max()
    return transff


def get_psf_slowness(cartesianx, cartesiany, slim, sstep, fmin, fmax, freq, fstep):
    """ calculate array response function in slowness domain
        
        parameters: 
            cartesianx : coordinates x (km)
            cartesiany : coordinates y (km)
            slim : limit of slowness(s/km)
            sstep : step number of slowness
            freq : frequency-band
            fstep : frequency step
    """
    # setting of slowness range
    sxmin = -slim
    sxmax = slim
    symin = sxmin
    symax = sxmax

    nf = int(np.ceil((fmax + fstep/10. - fmin)/ fstep))
    nsx = int(np.ceil((sxmax + sstep/10. - sxmin) / sstep))
    nsy = int(np.ceil((symax + sstep/10. - symin) / sstep))
    
    transff = np.empty((nsx, nsy))
    buff = np.zeros(nf)

    for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
        for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
            for k, f in enumerate(np.arange(fmin, fmax + fstep / 10., fstep)):
                _sum = 0j
                for l in range(cartesianx.shape[0]):
                    _sum += np.exp(complex(0., (cartesianx[l] * sx + cartesiany[l] * sy) * 2.0 * np.pi * f))
                buff[k] = abs(_sum) ** 2.0
            transff[i, j] = cumulative_trapezoid(buff, dx=fstep)[-1]
    

    transff /= transff.max()
    return transff


def get_psf_slowness_light(cartesianx, cartesiany, slim, sstep, fmin, fmax, freq, fstep):
    """ calculate array response function in slowness domain
        
        parameters: 
            cartesianx : coordinates x (km)
            cartesiany : coordinates y (km)
            slim : limit of slowness(s/km)
            sstep : step number of slowness
            freq : frequency-band
            fstep : frequency step
    """
    # setting of slowness range
    sxmin = -slim
    sxmax = slim
    symin = sxmin
    symax = sxmax

    nf = int(np.ceil((fmax + fstep/10. - fmin)/ fstep))
    nsx = int(np.ceil((sxmax + sstep/10. - sxmin) / sstep))
    nsy = int(np.ceil((symax + sstep/10. - symin) / sstep))
    
    transff = np.empty((nsx, nsy))
    buff = np.zeros(nf)
    f = (fmin+fmax)*0.5
    for i, sx in enumerate(np.arange(sxmin, sxmax + sstep / 10., sstep)):
        for j, sy in enumerate(np.arange(symin, symax + sstep / 10., sstep)):
            _sum = 0j
            for l in range(cartesianx.shape[0]):
                _sum += np.exp(complex(0., (cartesianx[l] * sx + cartesiany[l] * sy) * 2.0 * np.pi * f))
                
            transff[i, j] = abs(_sum) ** 2.0

    transff /= transff.max()
    return transff
                
def pad_psf(arf, spectrum):
    """ zero padding to extend the psf  

    parameters:
       psf: point spread function
       spectrum: extended psf                
    """
    out_psf = np.zeros( (spectrum.shape) )
    start = int(len(spectrum)*0.5 - len(arf)*0.5)
    end = int(start + len(arf))
    out_psf[start:end, start:end] = arf

    return out_psf
    

                
                
                
