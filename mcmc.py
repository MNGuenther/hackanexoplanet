#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  22 14:25:21 2023

@author:
Dr. Maximilian N. Günther
European Space Agency (ESA)
European Space Research and Technology Centre (ESTEC)
Keplerlaan 1, 2201 AZ Noordwijk, The Netherlands
Email: maximilian.guenther@esa.int
GitHub: mnguenther
Twitter: m_n_guenther
Web: www.mnguenther.com
"""



#::: modules
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import UnivariateSpline

#::: non-standard modules
import emcee

#::: local modules
import computer

#::: globals (I know, I know, ...)
data = {}
params = {}
bounds = []
mcmc = {'nwalkers':50, #50
        'total_steps':400, #400
        'burn_steps':300, #300
        'ndim':3}



###############################################################################
#::: MCMC log likelihood
###############################################################################
def mcmc_lnlike(theta):

    #::: globals
    global data
    global params
    # global bounds
    
    #::: set
    lnlike_total = 0
    params['radius_planet'] = theta[0]
    params['radius_star'] = theta[1]
    params['epoch'] = theta[2]
    
    #::: first, check and add external priors 
    #lnprior_external = calculate_external_priors(params) #TODO
    #lnlike_total += lnprior_external       
    
    #::: directly catch any issues
    if np.isnan(lnlike_total) or np.isinf(lnlike_total):
        return -np.inf
    
    #::: calculate the model; if there are any NaN, return -np.inf
    model = computer.calc_flux_model(params['radius_planet'], params['radius_star'], params['epoch'], params['period'], params['a'], params['incl'], params['q1'], params['q2'], data['time'])
    #TODO: not nice that this is also called model, rename it to sth better
    if any(np.isnan(model)) or any(np.isinf(model)): 
        return -np.inf

    #::: compute errors (simplification: not fitting the errors here but assuming the instrument error's are correct)
    yerr_w = data['flux_err']
    
    #::: compute baseline (simplification: only use hybrid spline)
    yerr_weights = data['flux_err']/np.nanmean(data['flux_err'])
    weights = 1./yerr_weights
    spl = UnivariateSpline(data['time'], data['flux']-model, w=weights, s=np.sum(weights)) #train a spline on the data time grid
    baseline = spl(data['time'])
    
    #::: calculate residuals and inv_simga2
    residuals = data['flux'] - model - baseline
    if any(np.isnan(residuals)): 
        return -np.inf
    inv_sigma2_w = 1./yerr_w**2

    #::: calculate lnlike
    lnlike_total += -0.5*(np.sum((residuals)**2 * inv_sigma2_w - np.log(inv_sigma2_w/2./np.pi))) #use np.sum to catch any nan and then set lnlike to nan
            
    return lnlike_total

    
        
###############################################################################
#::: MCMC log prior
###############################################################################
def mcmc_lnprior(theta):
    '''
    bounds has to be list of len(theta), containing tuples of form
    ('none'), ('uniform', lower bound, upper bound), or ('normal', mean, std)
    '''
    
    #::: globals
    # global data
    # global params
    global bounds
    
    lnp = 0.
    
    for th, b in zip(theta, bounds):
        if b[0] == 'uniform':
            if not (b[1] <= th <= b[2]): 
                return -np.inf
        elif b[0] == 'normal':
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[2]) * np.exp( - (th - b[1])**2 / (2.*b[2]**2) ) )
        elif b[0] == 'trunc_normal':
            if not (b[1] <= th <= b[2]): 
                return -np.inf
            lnp += np.log( 1./(np.sqrt(2*np.pi) * b[4]) * np.exp( - (th - b[3])**2 / (2.*b[4]**2) ) )
        else:
            raise ValueError('Bounds have to be "uniform" or "normal". Input from "params.csv" was "'+b[0]+'".')
            
    return lnp



###############################################################################
#::: MCMC log probability
###############################################################################    
def mcmc_lnprob(theta):
    
    lp = mcmc_lnprior(theta)
    
    if not np.isfinite(lp):
        return -np.inf
    else:
        ln = mcmc_lnlike(theta)
        return lp + ln
        


###########################################################################
#::: MCMC fit
###########################################################################
def mcmc_fit(target_name, params0):
        
    #::: globals
    global data
    global params
    global bounds
    data['time'], data['flux'], data['flux_err'], _ = computer.load_data(target_name)
    data['flux_err_scales'] = data['flux_err']/np.nanmean(data['flux_err'])
    params = params0
    bounds = computer.load_bounds(target_name)
    
    #::: setup
    theta_0 = [params['radius_planet'], params['radius_star'], params['epoch']]
        
    #::: set up a fresh results folder and backend
    if not os.path.exists(os.path.join('results',target_name)):
        os.makedirs(os.path.join('results',target_name))
    if os.path.exists(os.path.join('results',target_name,'mcmc_save.h5')):
        os.remove(os.path.join('results',target_name,'mcmc_save.h5'))
    backend = emcee.backends.HDFBackend(os.path.join('results',target_name,'mcmc_save.h5'))
    
    #::: set initial walker positions
    p0 = theta_0 + 1e-4 * np.random.randn(mcmc['nwalkers'], mcmc['ndim'])
    
    #::: run (without multiprocessing for now)
    #print("\nRoger that, Detective.") 
    #print("Thanks for all the clues.")
    #print("We are running a full investigation now...")
    sampler = emcee.EnsembleSampler(mcmc['nwalkers'],
                                    mcmc['ndim'],
                                    mcmc_lnprob,
                                    backend=backend)
    sampler.run_mcmc(p0, mcmc['total_steps'], progress=True)
    
    