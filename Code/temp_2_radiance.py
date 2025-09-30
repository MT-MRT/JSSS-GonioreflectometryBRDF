# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:55:37 2022

@author: GoettF

edited by Jannik Ebert.
"""
import numpy as np
import scipy.interpolate
import scipy.constants as const


def get_spectral_radiance(T, Npoints=1000, lmin = 2e-6, lmax = 10e-6):
    """
    returns spectral radiance W/(qm m sr)
    in wavelength interval lmin, lmax for temperature T
    """

    x = np.linspace(lmin, lmax, Npoints)
    y = 2*const.h*const.c**2 / x**5 *\
        1/(
            np.exp(const.h * const.c / (x*const.k*T)) -1
          )
    return x, y

def get_spectral_radiance_photons(T, Npoints=1000, lmin = 2e-6, lmax = 10e-6):
    """
    returns spectral radiance in photons/(qm s sr)
    in wavelength interval lmin, lmax for temperature T
    """

    x, y = get_spectral_radiance(T, Npoints=Npoints, lmin = lmin, lmax = lmax)
    Ephoton =  const.h * const.c / x
    y /= Ephoton

    return x, y


def get_radiance(T,
                 lmin,
                 lmax,
                 Nlambda,
                 transmission_function = None):
    """
    returns radiance in W/(qm sr)
    """
    if transmission_function is None:
        transmission_function = lambda x: 1
    l, spec_rad = get_spectral_radiance(
        T,
        Npoints = Nlambda,
        lmin = lmin,
        lmax = lmax
        )
    tcorr_spec_rad = spec_rad * transmission_function(l) # transmissionskorrigiert
    rad = np.sum(tcorr_spec_rad)*(l[1]-l[0]) ##assumes lambda is equally spaced
    return rad

def get_radiance_curve(Tmin,
                 Tmax,
                 lmin,
                 lmax,
                 NT = 1000,
                 Nlambda = 1000,
                 transmission_function = None):
    """
    Die Power / Temperature Kurve, T in K und lambda in m
    """
    if transmission_function is None:
        transmission_function = lambda x: 1

    Tarray = np.linspace(Tmin, Tmax, NT)
    rarray = np.zeros(NT)
    i = 0
    for T in Tarray:

        rad = get_radiance(
            T,
            lmin,
            lmax,
            Nlambda,
            transmission_function)
        rarray[i] = rad
        i+=1
    return Tarray, rarray

def get_radiance_photons(T,
                 lmin,
                 lmax,
                 Nlambda = 1000,
                 transmission_function = None):
    """
    returns radiance in photons / sr / m^2
    """
    if transmission_function is None:
        transmission_function = lambda x: 1
    l, spec_rad = get_spectral_radiance_photons(
        T,
        Npoints = Nlambda,
        lmin = lmin,
        lmax = lmax
        )
    tcorr_spec_rad = spec_rad * transmission_function(l) # transmissionskorrigiert
    rad = np.sum(tcorr_spec_rad)*(l[1]-l[0]) ##assumes lambda is equally spaced
    return rad

def get_radiance_curve_photons(Tmin,
                 Tmax,
                 lmin,
                 lmax,
                 NT = 1000,
                 Nlambda = 1000,
                 transmission_function = None):
    """
    Die Power / Temperature Kurve, T in K und lambda in m
    """
    if transmission_function is None:
        transmission_function = lambda x: 1

    Tarray = np.linspace(Tmin, Tmax, NT)
    rarray = np.zeros(NT)
    i = 0
    for T in Tarray:

        rad = get_radiance_photons(
            T,
            lmin,
            lmax,
            Nlambda,
            transmission_function)
        rarray[i] = rad
        i+=1
    return Tarray, rarray

def radiance_2_power(
        radiance,
        aperture,
        pixel_size,
        ):
    """
    returns the power (W) on a detector pixel from object radiance
    (transmission correction must be already included in radiance!)
    """
    p = np.pi*(1-(np.cos(np.arctan(1/2/aperture)))**2) *\
        pixel_size**2 * radiance

    return p

_optimization_helper = [0,0,0]
def thermogram_to_radiance(
        Tarray,
        lambda_min_um,
        lambda_max_um,
        ):
    """
    returns radiance in W/(qm sr)
    find radiance value with interpolation of the radiance curve
    """
    global _optimization_helper
    Tmin = 0.0001
    Tmax = 3300
    if _optimization_helper[0] != lambda_min_um \
        or _optimization_helper[1] != lambda_max_um:
        T, r = get_radiance_curve(
            Tmin,
            Tmax,
            lambda_min_um*1e-6,
            lambda_max_um*1e-6,
            )
        r_of_T = scipy.interpolate.interp1d(T,r)
    else:
        r_of_T = _optimization_helper[2]

    _optimization_helper = [lambda_min_um, lambda_max_um,r_of_T]
    rarray = r_of_T(Tarray)
    return rarray

_optimization_helper = [0,0,0]
def radiance_to_thermogram(
        rarray,
        lambda_min_um,
        lambda_max_um,
        ):
    """
    returns temperature in Kelvin
    find temperature value with interpolation of the radiance curve
    (author: ebert)
    """
    global _optimization_helper
    Tmin = 0.0001
    Tmax = 3300
    # TODO: Falsche Ergebnisse oder Error wegen lambda Grenzen -> quick fix in nächster Zeile
    lambda_max_um = lambda_max_um*0.99999999 # führt zu Fehler der Größenordnung e-06 K im Messbereich 0...3300 K
    if _optimization_helper[0] != lambda_min_um \
        or _optimization_helper[1] != lambda_max_um:
        T, r = get_radiance_curve(
            Tmin,
            Tmax,
            lambda_min_um*1e-6,
            lambda_max_um*1e-6,
            )
        T_of_r = scipy.interpolate.interp1d(r,T)
    else:
        T_of_r = _optimization_helper[2]

    _optimization_helper = [lambda_min_um, lambda_max_um,T_of_r]
    Tarray = T_of_r(rarray)
    return Tarray
