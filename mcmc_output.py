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
from matplotlib.ticker import ScalarFormatter, FixedLocator
import os
from IPython.display import FileLink

#::: non-standard modules
import emcee

#::: globals (I know, I know, ...)
data = {}
params = {}
bounds = []
fitkeys = ['radius_planet', 'radius_star', 'epoch']
labels = ['Radius of the planet (Earth radii)', 'Radius of the star (Solar radii)', 'Mid-transit time (days)']
mcmc = {'nwalkers':50, #50
        'total_steps':400, #400
        'burn_steps':300, #300
        'ndim':3}



###############################################################################
#::: MCMC output
###############################################################################
def mcmc_output(target_name):
        
    reader = emcee.backends.HDFBackend( os.path.join('results',target_name,'mcmc_save.h5'), read_only=True )

    #::: print something nice
    samples = draw_mcmc_posterior_samples(reader, Nsamples=None, as_type='dic')
    print('The investigation was successful.')
    print('We nailed down exactly what happened.')

    #::: plot histograms
    fig1, axes1 = plot_MCMC_histograms(reader)
    fig1.savefig( os.path.join('results',target_name,'mcmc_hist.pdf'), bbox_inches='tight' )
    
    #::: plot the chains
    fig2, axes2 = plot_MCMC_chains(reader)
    fig2.savefig( os.path.join('results',target_name,'mcmc_chains.jpg'), bbox_inches='tight' )

    # #::: plot the corner
    # fig = plot_MCMC_corner(reader)
    # fig.savefig( os.path.join(config.BASEMENT.outdir,'mcmc_corner.pdf'), bbox_inches='tight' )
    # plt.close(fig)
    
    #::: return 20 samples for plotting
    posterior_samples = draw_mcmc_posterior_samples(reader, Nsamples=20, as_type='dic')
    
    print('Have a look, we prepared you a case file with all the details:')

    return posterior_samples, fig1, fig2
    

    
###############################################################################
#::: draw samples from the MCMC save.5 (internally in the code)
###############################################################################
def draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='2d_array'):
    '''
    Default: return all possible sampels
    Set e.g. Nsamples=20 for plotting
    '''
    posterior_samples = sampler.get_chain(flat=True, discard=mcmc['burn_steps'])
    
    if Nsamples:
        posterior_samples = posterior_samples[np.random.randint(len(posterior_samples), size=Nsamples)]

    if as_type=='2d_array':
        return posterior_samples
    
    elif as_type=='dic':
        posterior_samples_dic = {}
        for key in fitkeys:
            ind = np.where(np.array(fitkeys)==key)[0] #fitkeys must be a numpy array for this operation
            posterior_samples_dic[key] = posterior_samples[:,ind].flatten()
        return posterior_samples_dic


    
###############################################################################
#::: plot the MCMC chains
###############################################################################
def plot_MCMC_chains(sampler):
    
    chain = sampler.get_chain()
    log_prob = sampler.get_log_prob()
    
    #plot chains; emcee_3.0.0 format = (nsteps, nwalkers, nparameters)
    fig, axes = plt.subplots(mcmc['ndim']+1, 1, figsize=(6,3*mcmc['ndim']) )
    
    #::: plot the lnprob_values; emcee_3.0.0 format = (nsteps, nwalkers)
    axes[0].plot(log_prob, '-', rasterized=True)
    axes[0].axvline( mcmc['burn_steps'], color='k', linestyle='--' )
    mini = np.min(log_prob[mcmc['burn_steps'],:])
    maxi = np.max(log_prob[mcmc['burn_steps'],:])
    axes[0].set( title='lnprob', xlabel='steps', rasterized=True, ylim=[mini, maxi] )
    axes[0].xaxis.set_major_locator(FixedLocator(axes[0].get_xticks())) #useless line to bypass useless matplotlib warnings
    axes[0].set_xticklabels( [int(label) for label in axes[0].get_xticks()] )
    
    #:::plot all chains of parameters
    for i in range(mcmc['ndim']):
        ax = axes[i+1]
        ax.set(title=labels[i], xlabel='steps')
        ax.plot(chain[:,:,i], '-', rasterized=True)
        ax.axvline( mcmc['burn_steps'], color='k', linestyle='--' )
        ax.xaxis.set_major_locator(FixedLocator(ax.get_xticks())) #useless line to bypass useless matplotlib warnings
        ax.set_xticklabels( [int(label) for label in ax.get_xticks()] )

    plt.tight_layout()
    return fig, axes

    
    
###############################################################################
#::: plot the MCMC histograms (instead of the Corner plot)
###############################################################################
def plot_MCMC_histograms(sampler):
    
    samples = draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='dic')
    
    fig, axes = plt.subplots(1, mcmc['ndim'], figsize=(12,4))
    for i, (key, label) in enumerate(zip(fitkeys, labels)):
        ax = axes[i]
        ax.hist(samples[key])
        ax.axvline(np.median(samples[key]), color='k', linestyle='--')
        ax.set(title=label, ylabel='', yticklabels=[])
    axes[0].set(ylabel='Statistical evidence')
        
    return fig, axes