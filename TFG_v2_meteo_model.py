import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import yaml

def read_config_file():
    with open("config_meteo_model.yml", "r") as yamlfile:
        config_data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        print("Read successful config file")
    print(config_data)
    return(config_data)

def get_constants(config_data):
    print('Getting constants...')
    cp = config_data['constants']['cp']
    cv = config_data['constants']['cv']
    R = config_data['constants']['R']
    mu = config_data['constants']['mu']
    g = config_data['constants']['g']
    c = config_data['constants']['c']
    dx = config_data['constants']['dx']
    dz = config_data['constants']['dz']
    gamma = config_data['constants']['gamma']
    theta_mean0 = config_data['constants']['theta0']
    theta_c = config_data['constants']['theta_c']
    return(cp, cv, R, mu, g, c, dx, dz, gamma, theta_mean0, theta_c)

def get_simulation_parameters(config_data):
    if config_data['test_cases']['density_current'] == True:
        print('Setting horizontal and vertical scales for Density Current test...')
        #define horizontal and vertical scale length 
        L =  19200.0 #meters following Giraldo & Restelli (2008)
        H = 5000.0 #meters
        T = 900.0 #seconds
        return(L, H, T)   

    elif config_data['test_cases']['warm_bubble'] == True:
        #todo complete with rest of test cases
        print('Setting horizontal and vertical scales for Warm Bubble test...')
        #define horizontal and vertical scale length:
        L =  1000.0 #meters following Robert
        H = 1000.0 #meters
        T = 800.0 #seconds
        return(L,H,T)
    else:
        print('No defined test case. Please set to True any test case.')
     

def get_time_resolution(dx, dz, c, gamma):
    #get time resolution meeting cfl condition:
    #max dt that can be applied meeting cfl condition:
    dt_max = min(dx/c, dz/c)
    #apply a factor less than 1 for a better chance of 
    #numerical stability
    dt = gamma*dt_max
    return(dt_max, dt)

def get_iterations_number(dx, dz, dt, L, H, T):
    print('Getting total number of iterations...')
    #number of iterations in the x-axis:
    I = int(L/dx)
    J = int(H/dz)
    N = int(T/dt)
    return(I, J, N)

def get_meshgrid(dx, dz, L, H):
    #get meshgrid for integration adding ghost points to apply boundary conditions:
    x = np.arange(-2*dx, L + 3*dx, dx)
    z = np.arange(-2*dz, H + 3*dz, dz)

    xx, zz = np.meshgrid(x, z)
    return(xx, zz)

def get_initial_potential_temperature_perturbation(config_data, xx, zz):
    if config_data['test_cases']['density_current'] == True:
        #get mean potential temperature
        xc = config_data['parameters_test_cases']['density_current']['xc']
        zc = config_data['parameters_test_cases']['density_current']['zc']
        xr = config_data['parameters_test_cases']['density_current']['xr']
        zr = config_data['parameters_test_cases']['density_current']['zr']
        theta_c = config_data['constants']['theta_c']
        r_c = config_data['parameters_test_cases']['density_current']['r_c']
        r = np.sqrt( ((xx-xc)/xr)**2 + ((zz-zc)/zr)**2 )

        thetap0 = np.piecewise(r, [r <= r_c , r > r_c], 
                       [lambda r: (1./2.)*theta_c*( 1.0 + np.cos(np.pi*r/r_c) ), lambda r: 0.0])







