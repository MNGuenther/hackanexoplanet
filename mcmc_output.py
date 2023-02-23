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
from tabulate import tabulate

#::: non-standard modules
import emcee

#::: local modules
import latex_printer

#::: globals (I know, I know, ...)
data = {}
params = {}
bounds = []
fitkeys = ['radius_planet', 'radius_star', 'epoch']
labels = ['Radius of the planet\n(in units of Earth radii)', 'Radius of the star\n(in units of Solar radii)', 'Mid-transit time\n(in units of days)']
mcmc = {'nwalkers':50, #50
        'total_steps':400, #400
        'burn_steps':300, #300
        'ndim':3}



###############################################################################
#::: MCMC output
###############################################################################
def mcmc_output(target_name, params=None):
        
    reader = emcee.backends.HDFBackend( os.path.join('results',target_name,'mcmc_save.h5'), read_only=True )

    #::: return 20 samples for plotting
    posterior_samples = draw_mcmc_posterior_samples(reader, Nsamples=20, as_type='dic')
    
    #::: print something nice
    samples = draw_mcmc_posterior_samples(reader, Nsamples=None, as_type='dic')
    #print("\nDetective, look at this!")
    #print('The investigation was successful.')
    #print('We nailed down exactly what happened.')

    #::: plot histograms
    fig_hist = plot_MCMC_histograms(reader, target_name=target_name)[0]
    fig_hist.savefig( os.path.join('results',target_name,'histograms.pdf'), bbox_inches='tight' )
    
    #::: plot the chains
    fig_chains = plot_MCMC_chains(reader)[0]
    fig_chains.savefig( os.path.join('results',target_name,'chains.jpg'), bbox_inches='tight' )
    plt.close(fig_chains) #close it, as it is only needed for internal bookkeeping

    # #::: get the table
    table = get_MCMC_table(reader, params=params, target_name=target_name)
    table2 = get_MCMC_table(reader, params=params, tablefmt='pretty', target_name=target_name)
    with open(os.path.join('results',target_name,'table.txt'), 'w') as f:
        f.write(tabulate(table2, headers=['Name', 'Median value', 'Lower error', 'Upper error', 'Case note', 'Target']))
    
    #print('We also found some other observations from the Archive that gave us extra insights.')
    #print('Have a look, we prepared you a case file with all the details:')

    return posterior_samples, fig_hist, table
    

    
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
#::: compute the posterior values (median and uncertainties) from the MCMC save.5 (internally in the code)
###############################################################################
def compute_posterior_values(sampler):
    samples = draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='dic')
    
    values = {}
    for key in fitkeys:
        values[key] = {}
        values[key]['median'] = np.nanmedian(samples[key])
        values[key]['lower_error'] = np.nanmedian(samples[key]) - np.nanpercentile(samples[key],16)
        values[key]['upper_error'] = np.nanpercentile(samples[key],84) - np.nanmedian(samples[key])
        
    return values


    
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
def plot_MCMC_histograms(sampler, target_name=None):
    
    samples = draw_mcmc_posterior_samples(sampler, Nsamples=None, as_type='dic')
    values = compute_posterior_values(sampler)
    
    fig, axes = plt.subplots(1, mcmc['ndim'], figsize=(12,4), tight_layout=True)
    if target_name is None:
        fig.suptitle('Histograms of the statistical probability of all parameter values')
    else:
        fig.suptitle('Histograms of the statistical probability of all parameter values of '+target_name)
    for i, (key, label) in enumerate(zip(fitkeys, labels)):
        ax = axes[i]
        ax.hist(samples[key],bins=20,alpha=0.5)
        ax.axvline(values[key]['median'], color='r', linestyle='-')
        ax.axvline(values[key]['median']-values[key]['lower_error'], color='r', linestyle='--')
        ax.axvline(values[key]['median']+values[key]['upper_error'], color='r', linestyle='--')
        result_str = r'$'+latex_printer.round_tex(values[key]['median'], values[key]['lower_error'], values[key]['upper_error'])+'$'
        ax.set(xlabel='Possible solutions', title=label+'\n'+result_str, ylabel='', yticklabels=[])
    axes[0].set(ylabel='Statistical probability')
        
    return fig, axes
    
    
    
###############################################################################
#::: return a pandas table
###############################################################################
def get_MCMC_table(sampler, params=None, tablefmt='html', target_name=None):
    
    values = compute_posterior_values(sampler)
    
    if target_name is None:
        table = {'name':[],
                 'median':[],
                 'lower_error':[],
                 'upper_error':[],
                 'comment':[]}
    
    else:
        table = {'name':[],
                 'median':[],
                 'lower_error':[],
                 'upper_error':[],
                 'comment':[],
                 'target':[]}

    for i, (key, label) in enumerate(zip(fitkeys, labels)):
        s1, s2, s3 = latex_printer.round_txt_separately(values[key]['median'], values[key]['lower_error'], values[key]['upper_error'])
        table['name'].append(label)
        table['median'].append(s1)
        table['lower_error'].append(s2)
        table['upper_error'].append(s3)
        table['comment'].append('Cheops observations')
        if target_name:
            table['target'].append(target_name)

    if params is not None and 'period' in params:
        table['name'].append('Orbital period (in units of days)')
        table['median'].append(str(params['period']))
        table['lower_error'].append('')
        table['upper_error'].append('')
        table['comment'].append('Other observations from the archive')
        if target_name:
            table['target'].append(target_name)  

    '''
    if params is not None and 'a' in params:
        table['name'].append('Orbital semi-major axis (in units of AU)')
        table['median'].append(str(params['a']))
        table['lower_error'].append('')
        table['upper_error'].append('')
        table['comment'].append('Other observations from the archive')
        if target_name:
            table['target'].append(target_name) 

    if params is not None and 'incl' in params:
        table['name'].append('Orbital inclination (in units of degree)')
        table['median'].append(str(params['incl']))
        table['lower_error'].append('')
        table['upper_error'].append('')
        table['comment'].append('Other observations from the archive')
        if target_name:
            table['target'].append(target_name)
    '''
    
    # return pd.DataFrame(table)
    return table