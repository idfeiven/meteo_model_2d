# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 10:52:03 2018

@author: Iván
"""

#TFG
#Integración de un modelo meteorológico compresible y no hidrostático. 

#TEST CASE #4

#En este programa estudiaremos la evolución de una burbuja de aire frío que desciende
#Las ecuaciones a integrar son las que aparecen en el set-1 de ecuaciones del paper 
#Giraldo c)2008. En este modelo se ignoran los efectos hidrometeorológicos y 
#todos los procesos relacionados con vapor de agua.

#En este programa implementaremos un algoritmo del tipo REA (Reconstruct-Evolve-Average)
#de segundo orden. Ahora la diferencia está en que como la perturbación
#de temperatura es negativa, el aire circundante es más cálido que el de la pro-
#pia burbuja, con lo que será más denso y empezará a hundirse. Al ser este un al-
#goritmo de 2º orden, se debería ver más estructura cuando la burbuja evolucione.

import numpy as np
import matplotlib.pyplot as plt

#Definimos parámetros iniciales: paso de malla, perturbación inicial, y condi-
#ciones iniciales y de contorno:
thetac = -30.0
pic = 4*np.arctan(1)
xc = 0.0
zc = 3000.0
xr = 4000.0
zr = 2000.0
rc = 1.0
g = 9.81
#Calor específico, constante del aire R y coeficiente de rozamiento D:
cp = 1004.0
cv = 717.0
R = 287.0
D = 75.0
#Probar resoluciones distintas que cumplan la condición CFL
dx = 75.0
dz = 75.0
dt = 0.1875

#Definimos el número de iteraciones espacial y temporal:
N = np.int(900.0/dt)
I = np.int(38850.0/dx)
J = np.int(5475.0/dz)

#Definimos malla de puntos en el dominio de integración. Para que el problema
#esté centrado, es decir, que el máximo de temperatura se de justo encima de un
#punto de malla, definiremos 2 mallas de puntos (x1,z1) y (x2,z2). En la malla
#1 será donde haremos las integraciones y la malla 2 la usaremos para hacer la
#representación de forma correcta.
#Malla centrada:
x1 = np.arange(-19425,19425,dx)
z1 = np.arange(-225,5250,dz)

#Malla de representación:
x2 = np.linspace(-19425,19425,I)
#Debido a las condiciones de contorno, el dominio físico vertical empezaría en
#2dz y no en 0 exactamente:
z2 = np.linspace(-225,5250,J)
xx2,zz2 = np.meshgrid(x2,z2)

#Pintamos la perturbación de temperatura inicial:
xx1,zz1 = np.meshgrid(x1,z1)
r = np.sqrt( ((xx1-xc)/xr)**2 + ((zz1-zc)/zr)**2 )

thetap0 = np.piecewise(r, [r<=rc , r>rc], 
                       [lambda r: (1./2.)*thetac*( 1.0 + np.cos(pic*r/rc) ), lambda r: 0.0])

fig = plt.figure()
fig, axes = plt.subplots(figsize=(15,3)) #Controla el tamaño de la figura 
#en figsize=('tamaño eje x en pulgadas', 'tamaño eje z en pulgadas') y píxels de
#la imagen dpi (dots per inch).
plt.contourf(xx2,zz2,thetap0,vmin=-30.0,vmax=0.0,levels=[-30.0,-27.0,-24.0,-21.0,-18.0,
                                                        -15.0,-12.0,-9.0,-6.0,-3.0,0.0])
plt.title(r"$\theta_0' \ (K) $")
plt.xlabel(r"$ x \ (m)$")
plt.ylabel(r"$ z \ (m)$")
plt.ylim([0,5000])
plt.xlim([0,19200])
plt.colorbar(ticks=[-30.0,-27.0,-24.0,-21.0,-18.0,-15.0,-12.0,-9.0,-6.0,-3.0,0.0])
plt.show()

#Tamaños de imagen que usar según el caso:
#Dominio 1 (-10000,10000): (15,3), dpi=200
#Dominio 2 (0,19200): (15,3.25)

#Definimos funciones y derivadas de los valores medios de la temperatura potencial y la pre-
#sión de Exner:
thetam = 300.0
dthetam = 0.0

#Ojo! pim es una función que depende solo de z, con lo que será una matriz que solo tenga una fila,
#pero a la hora de integrar debemos tener en cuenta que estamos trabajando con matrices
#de dimensiones IXJ...Hay que hacer que esta matriz tenga dimensiones IXJ:
#Hay que tener cuidado con la asignación de los elementos de matriz que hace Python.
#En todas las matrices el elemento 0 nuestro es el #1 de Python, el 1 es #2 en Pyhton,
#El J o I nuestro es I-1,J-1 en Python. Hay que tener esto en cuenta a la hora
#de integrar y de asignar las condiciones de contorno.
pim = np.zeros([J,I])
for i in range (0,I,1):
    for j in range (0,J,1):
        pim[j,i] = 1.0 - (g/(cp*thetam))*z1[j]
dpim = -g/(cp*thetam)

#Definimos las funciones que se van a integrar: u la velocidad de viento zonal
#(componente x), w la velocidad vertical, y thetap la evolución de la pertur-
#bación de temperatura potencial en el instante n, n+1 y n+2 (n, np1, np2):
#El comando np.meshgrid cambia las dimensiones de las matrices con las que 
#inicialmente trabajaríamos, que sería lógicamente [I,J], pero debido a este
#inconveniente debemos girar las matrices para que la representación sea correcta.
#J sigue siendo la coordenada vertical e I la horizontal.
#Funciones en el instante inicial:
u_np0 = np.zeros([J,I])
w_np0 = np.zeros([J,I])
thetap_np0 = np.zeros([J,I])
pip_np0 = np.zeros([J,I])
#Funciones para aplicar el REA:
u_np1 = np.zeros([J,I])
w_np1 = np.zeros([J,I])
thetap_np1 = np.zeros([J,I])
pip_np1 = np.zeros([J,I])
#Funciones para el instante dt/2->dt:
u_np2 = np.zeros([J,I])
w_np2 = np.zeros([J,I])
thetap_np2 = np.zeros([J,I])
pip_np2 = np.zeros([J,I])
#Funciones finales
u = np.zeros([J,I])
w = np.zeros([J,I])
pip = np.zeros([J,I])
thetap = np.zeros([J,I])

#Definimos términos de advección:
adhpip = np.zeros([J,I])
advpip = np.zeros([J,I])
adhthetap = np.zeros([J,I])
advthetap = np.zeros([J,I])
adhu = np.zeros([J,I])
advu = np.zeros([J,I])
adhw = np.zeros([J,I])
advw = np.zeros([J,I])

#Variables para el cálculo de la advección según REA:

#Definimos las pendientes de las rectas de reconstrucción (REA 2º orden), y los
#términos a y b para la evaluación del operador minmod:
a_pip_x = np.zeros([J,I])
b_pip_x = np.zeros([J,I])
a_thetap_x = np.zeros([J,I])
b_thetap_x = np.zeros([J,I])
a_u_x = np.zeros([J,I])
b_u_x = np.zeros([J,I])
a_w_x = np.zeros([J,I])
b_w_x = np.zeros([J,I])
a_pip_z = np.zeros([J,I])
b_pip_z = np.zeros([J,I])
a_thetap_z = np.zeros([J,I])
b_thetap_z = np.zeros([J,I])
a_u_z = np.zeros([J,I])
b_u_z = np.zeros([J,I])
a_w_z = np.zeros([J,I])
b_w_z = np.zeros([J,I])
sigmaxpip = np.zeros([J,I])
sigmaxthetap = np.zeros([J,I])
sigmaxu = np.zeros([J,I])
sigmaxw = np.zeros([J,I])
sigmazpip = np.zeros([J,I])
sigmazthetap = np.zeros([J,I])
sigmazu = np.zeros([J,I])
sigmazw = np.zeros([J,I])

#Extremos de la recta de reconstrucción (sin evolucionar, horizontal x y vertical
#z):
D_x_pip = np.zeros([J,I])
I_x_pip = np.zeros([J,I])
D_x_thetap = np.zeros([J,I])
I_x_thetap = np.zeros([J,I])
D_x_u = np.zeros([J,I])
I_x_u = np.zeros([J,I])
D_x_w = np.zeros([J,I])
I_x_w = np.zeros([J,I])

D_z_pip = np.zeros([J,I])
I_z_pip = np.zeros([J,I])
D_z_thetap = np.zeros([J,I])
I_z_thetap = np.zeros([J,I])
D_z_u = np.zeros([J,I])
I_z_u = np.zeros([J,I])
D_z_w = np.zeros([J,I])
I_z_w = np.zeros([J,I])

#Valores medios del viento u y w para la evolución de la pendiente:
u_np1_m = np.zeros([J,I])
w_np1_m = np.zeros([J,I])
u_np2_m = np.zeros([J,I])
w_np2_m = np.zeros([J,I])

#Distancias d_i, d_ip1 y distancia total d que avanzan los extremos de la recta:
d_t_i = np.zeros([J,I])
d_t_j = np.zeros([J,I])

#Pendientes evolucionadas:
sigmaxpip_np1 = np.zeros([J,I])
sigmaxthetap_np1 = np.zeros([J,I])
sigmaxu_np1 = np.zeros([J,I])
sigmaxw_np1 = np.zeros([J,I])
sigmazpip_np1 = np.zeros([J,I])
sigmazthetap_np1 = np.zeros([J,I])
sigmazu_np1 = np.zeros([J,I])
sigmazw_np1 = np.zeros([J,I])

#Áreas S de cada variable (sumadas da la integral):
S_pip = np.zeros([J,I])
S_thetap = np.zeros([J,I])
S_u = np.zeros([J,I])
S_w = np.zeros([J,I])

#Puntos de corte de las rectas de reconstrucción:
T_pip = np.zeros([J,I])
T_thetap = np.zeros([J,I])
T_u = np.zeros([J,I])
T_w = np.zeros([J,I])
T1_pip = np.zeros([J,I])
T1_thetap = np.zeros([J,I])
T1_u = np.zeros([J,I])
T1_w = np.zeros([J,I])
T2_pip = np.zeros([J,I])
T2_thetap = np.zeros([J,I])
T2_u = np.zeros([J,I])
T2_w = np.zeros([J,I])

#Condiciones iniciales: perturbación de temperatura, presión (u,w especificadas
#antes):
thetap_np0 = thetap0      
pip_np0 = np.zeros([J,I])

#Ahora el algoritmo temporal será un RK-2. El esquema de este algoritmo se basa 
#en integrar primero las tendencias (términos de las ecuaciones sin advección)
#Y después advectar todas las variables vertical y horizontalmente en este orden.
#Para la integración de las tendencias se sigue el esquema de Runge Kutta 2.

#Q=(pip_np0,thetap_np0,u_np0,w_np0) -> dt/2 -> Q_np1 -> dt -> Q_np2 (con derivada
#temporal parcial de Q_np0 y resto de variables evaluadas en np1)

#Integración del sistema.
#Euler-Forward/Backward. Para este caso, podemos añadir los procesos de disipa-
#ción que genera el rozamiento D, proporcional a las derivadas segundas de la
#velocidad del viento:
#Generamos un contador de n para poder representar los campos que nos interesan
#en el instante de interés (cada 500 iteraciones un plot-> 28 plots)
M=30
m=0
q = ["30s","60s","90s","120s","150s","180s","210s","240s","270s","300s","330s",
     "360s","390s","420s","450s","480s","510s","540s","570s","600s","630s",
     "660s","690s","720s","750s","780s","810s","840s","870s","900s"]
n=0
for n in range (0,N,1):
    #Ponemos en marcha el contador
    n= n+1     
    for i in range (1,I-1,1):        
        for j in range (1,J-1,1):
            #REA 2º orden términos de advección (adh= advección horizontal, adv=advección
            #vertical. Aplicamos RK-2. Esto implica una integración adicional con 
            #dt -> dt/2. Este algoritmo requiere más elaboración debido a la natu-
            #raleza de la advección. Para calcular adh(variable) y adv(variable)
            #seguiremos una serie de pasos siguiendo la filosofía del Reconstruct
            #Evolve-Average. Para ello seguiremos estos pasos:

#---------------------PASO DE N A N+1/2 CON dt/2------------------------------------
            
            #PASO 1: Integramos las tendencias sin los términos de advección. 
            pip_np1[j,i] = pip_np0[j,i] - (dt/2)*w_np0[j,i]*dpim - (R/cv)*(pim[j,i] + pip_np0[j,i])*( (dt/(4*dz))*(w_np0[j+1,i] 
                            - w_np0[j-1,i]) + (dt/(4*dx))*(u_np0[j,i+1] - u_np0[j,i-1]) )
          
            thetap_np1[j,i] = thetap_np0[j,i] - w_np0[j,i]*dthetam*(dt/2) #+ D*(dt/2)*( (thetap_np0[j,i+1] 
            #+ thetap_np0[j,i-1] - 2*thetap_np0[j,i])/(dx**2) + (thetap_np0[j+1,i] + thetap_np0[j-1,i] - 2*thetap_np0[j,i])/(dz**2) )
            
    for i in range (1,I-1,1):
        for j in range (1,J-1,1):
            #Aplicamos Euler-Forward Backward:    
            u_np1[j,i] = u_np0[j,i] - (dt/(4*dx))*cp*(thetam + thetap_np1[j,i])*(pip_np1[j,i+1] - pip_np1[j,i-1]) #+ D*(dt/2)*( (u_np0[j,i+1] 
            #+ u_np0[j,i-1] - 2*u_np0[j,i])/(dx**2) + (u_np0[j+1,i] + u_np0[j-1,i] - 2*u_np0[j,i])/(dz**2) )
                
            w_np1[j,i] = w_np0[j,i] - cp*(dt/(4*dz))*(thetam + thetap_np1[j,i])*(pip_np1[j+1,i] 
                        - pip_np1[j-1,i]) + g*(dt/2)*(thetap_np1[j,i]/thetam) #+ D*(dt/2)*( (w_np0[j,i+1] 
            #+ w_np0[j,i-1] - 2*w_np0[j,i])/(dx**2) + (w_np0[j+1,i] + w_np0[j-1,i] - 2*w_np0[j,i])/(dz**2) )

#---------------------PASO DE N+1/2 A N+1 USANDO dt USANDO RK-2--------------------------
                    
    #Ahora, una vez calculadas las tendencias, hacemos de nuevo la integración para pasar 
    #del instante n al instante n+1, usando las funciones que acabamos de obtener,
    #dejando evolucionar el sistema de n a n+1 y recalculamos los minmod de los 
    #términos de advección:    
    for i in range (1,I-1,1):        
        for j in range (1,J-1,1):
            #PASO 1: Integramos las tendencias sin los términos de advección.
            pip_np2[j,i] = pip_np0[j,i] - dt*w_np1[j,i]*dpim - (R/cv)*(pim[j,i] 
            + pip_np1[j,i])*( (dt/(2*dz))*(w_np1[j+1,i] - w_np1[j-1,i]) + (dt/(2*dx))*(u_np1[j,i+1] - u_np1[j,i-1]) )
            
            thetap_np2[j,i] = thetap_np0[j,i] - w_np1[j,i]*dthetam*dt #+ D*dt*( (thetap_np1[j,i+1] 
            #+ thetap_np1[j,i-1] - 2*thetap_np1[j,i])/(dx**2) + (thetap_np1[j+1,i] + thetap_np1[j-1,i] - 2*thetap_np1[j,i])/(dz**2) )
            
    for i in range (1,I-1,1):
        for j in range (1,J-1,1):
            #Aplicamos Euler-Forward Backward:    
            u_np2[j,i] = u_np0[j,i] - (dt/(2*dx))*cp*(thetam 
            + thetap_np2[j,i])*(pip_np2[j,i+1] - pip_np2[j,i-1]) #+ D*dt*( (u_np1[j,i+1] 
            #+ u_np1[j,i-1] - 2*u_np1[j,i])/(dx**2) + (u_np1[j+1,i] + u_np1[j-1,i] - 2*u_np1[j,i])/(dz**2) )
                
            w_np2[j,i] = w_np0[j,i] - cp*(dt/(2*dz))*(thetam 
            + thetap_np2[j,i])*(pip_np2[j+1,i] - pip_np2[j-1,i]) + g*dt*(thetap_np2[j,i]/thetam) #+ D*dt*( (w_np1[j,i+1] 
            #+ w_np1[j,i-1] - 2*w_np1[j,i])/(dx**2) + (w_np1[j+1,i] + w_np1[j-1,i] - 2*w_np1[j,i])/(dz**2) )

        #PASO 2: Calculamos la advección vertical advpip, advthetap, advu, advw:
            #a) Definimos limitador minmod para la pendiente de reconstrucción sigma
            #términos a,b de las funciones a advectar:            
            #Para la advección vertical:

    for i in range (0,I,1):
        for j in range (1,J-1,1):              
            a_pip_z[j,i] = (pip_np2[j,i]-pip_np2[j-1,i])/dz            
            a_thetap_z[j,i] = (thetap_np2[j,i]-thetap_np2[j-1,i])/dz            
            a_u_z[j,i] = (u_np2[j,i]-u_np2[j-1,i])/dz            
            a_w_z[j,i] = (w_np2[j,i]-w_np2[j-1,i])/dz
            b_pip_z[j,i] = (pip_np2[j+1,i]-pip_np2[j,i])/dz
            b_thetap_z[j,i] = (thetap_np2[j+1,i]-thetap_np2[j,i])/dz
            b_u_z[j,i] = (u_np2[j+1,i]-u_np2[j,i])/dz
            b_w_z[j,i] = (w_np2[j+1,i]-w_np2[j,i])/dz

    for i in range (0,I,1):
        for j in range (1,J-1,1):
            #Minmod para la advección vertical de pip:
            if  abs(a_pip_z[j,i]) < abs(b_pip_z[j,i]) and (a_pip_z[j,i])*(b_pip_z[j,i]) > 0.0:
                sigmazpip[j,i] = a_pip_z[j,i]
            elif abs(a_pip_z[j,i]) > abs(b_pip_z[j,i] ) and (a_pip_z[j,i])*(b_pip_z[j,i]) > 0.0:
                sigmazpip[j,i] = b_pip_z[j,i]
            else:
                sigmazpip[j,i] = 0.0
                
            #Minmod para la advección vertical de thetap:
            if abs(a_thetap_z[j,i]) < abs(b_thetap_z[j,i]) and (a_thetap_z[j,i])*(b_thetap_z[j,i]) > 0.0:
                sigmazthetap[j,i] = a_thetap_z[j,i]
            elif abs(a_thetap_z[j,i]) > abs(b_thetap_z[j,i] ) and (a_thetap_z[j,i])*(b_thetap_z[j,i]) > 0.0:
                sigmazthetap[j,i] = b_thetap_z[j,i]
            else:
                sigmazthetap[j,i] = 0.0
                
            #Minmod para la advección vertical de u:
            if abs(a_u_z[j,i]) < abs(b_u_z[j,i]) and (a_u_z[j,i])*(b_u_z[j,i]) > 0.0:
                sigmazu[j,i] = a_u_z[j,i]
            if abs(a_u_z[j,i]) > abs(b_u_z[j,i]) and (a_u_z[j,i])*(b_u_z[j,i]) > 0.0:
                sigmazu[j,i] = b_u_z[j,i]
            else:
                sigmazu[j,i] = 0.0
                
            #Minmod para la advección vertical w:
            if abs(a_w_z[j,i]) < abs(b_w_z[j,i]) and (a_w_z[j,i])*(b_w_z[j,i]) > 0.0:
                sigmazw[j,i] = a_w_z[j,i]
            elif abs(a_w_z[j,i]) > abs(b_w_z[j,i]) and (a_w_z[j,i])*(b_w_z[j,i]) > 0.0:
                sigmazw[j,i] = b_w_z[j,i]
            else:
                sigmazw[j,i] = 0.0
                
    #b) Calculamos los extremos D (Derecha) e I (Izquierda) de la pendiente
    #de reconstrucción:
            D_z_pip[j,i] = pip_np2[j,i] + sigmazpip[j,i]*(dz/2) 
            I_z_pip[j,i] = pip_np2[j,i] - sigmazpip[j,i]*(dz/2) 
            D_z_thetap[j,i] = thetap_np2[j,i] + sigmazthetap[j,i]*(dz/2) 
            I_z_thetap[j,i] = thetap_np2[j,i] - sigmazthetap[j,i]*(dz/2) 
            D_z_u[j,i] = u_np2[j,i] + sigmazu[j,i]*(dz/2) 
            I_z_u[j,i] = u_np2[j,i] - sigmazu[j,i]*(dz/2) 
            D_z_w[j,i] = w_np2[j,i] + sigmazw[j,i]*(dz/2) 
            I_z_w[j,i] = w_np2[j,i] - sigmazw[j,i]*(dz/2) 
            
    #Calculamos el valor medio del viento correspondiente a cada celda:       
    for i in range (0,I,1):
        for j in range (1,J,1):
            w_np2_m[j,i] = (w_np2[j,i] + w_np2[j-1,i])/2.
            
    #c) Calculamos la evolución de la pendiente. Para ello, calculamos la
    #distancia d_z que se ha avanzado en el proceso según los signos de la ve-
    #locidad del viento:
    d_t_j[0,:] = dz
    for i in range (0,I,1):
        for j in range (1,J,1):
            d_t_j[j,i] = dz + (w_np2_m[j,i] - w_np2_m[j-1,i])*dt
                
    for i in range (0,I,1):
        for j in range (0,J,1):                                              
            sigmazpip_np1[j,i] = ( D_z_pip[j,i] - I_z_pip[j,i] )/d_t_j[j,i]
            sigmazthetap_np1[j,i] = ( D_z_thetap[j,i] - I_z_thetap[j,i] )/d_t_j[j,i]
            sigmazu_np1[j,i] = ( D_z_u[j,i] - I_z_u[j,i] )/d_t_j[j,i]
            sigmazw_np1[j,i] = ( D_z_w[j,i] - I_z_w[j,i] )/d_t_j[j,i]
    
    #f) Calculamos el área correspondiente a cada celda. El área total
    #comprende el término de advección que corresponda y depende muy 
    #fuertemente de los signos de las velocidades en dos puntos de malla
    #consecutivos. Inicializamos la integracion haciendo que las áreas sean 0
    #para después ir sumando. El resultado de la suma son nuestras variables
    #advectadas verticalmente:
    #Inicializamos suma:
    for i in range (0,I,1):
        for j in range (0,J,1):
            S_pip[j,i] = 0.0 
            S_thetap[j,i] = 0.0 
            S_u[j,i] = 0.0
            S_w[j,i] = 0.0

    for i in range (0,I,1):
        for j in range (1,J-1,1):
            if w_np2_m[j,i] >= 0.0 and w_np2_m[j+1,i] >= 0.0:
                #Aquí hay contribución a las celdas j y j+1
                T_pip[j,i] = I_z_pip[j,i] + sigmazpip_np1[j,i]*(dz - w_np2_m[j,i]*dt)
                T_thetap[j,i] = I_z_thetap[j,i] + sigmazthetap_np1[j,i]*(dz - w_np2_m[j,i]*dt)
                T_u[j,i] = I_z_u[j,i] + sigmazu_np1[j,i]*(dz - w_np2_m[j,i]*dt)
                T_w[j,i] = I_z_w[j,i] + sigmazw_np1[j,i]*(dz - w_np2_m[j,i]*dt)
                
                S_pip[j,i] = S_pip[j,i] + (dz - w_np2_m[j,i]*dt)*( T_pip[j,i] + I_z_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + (dz - w_np2_m[j,i]*dt)*( T_thetap[j,i] + I_z_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + (dz - w_np2_m[j,i]*dt)*( T_u[j,i] + I_z_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + (dz - w_np2_m[j,i]*dt)*( T_w[j,i] + I_z_w[j,i] )/2.
                
                S_pip[j+1,i] = S_pip[j+1,i] + w_np2_m[j+1,i]*dt*( T_pip[j,i] + D_z_pip[j,i] )/2.
                S_thetap[j+1,i] = S_thetap[j+1,i] + w_np2_m[j+1,i]*dt*( T_thetap[j,i] + D_z_thetap[j,i] )/2.
                S_u[j+1,i] = S_u[j+1,i] + w_np2_m[j+1,i]*dt*( T_u[j,i] + D_z_u[j,i] )/2.
                S_w[j+1,i] = S_w[j+1,i] + w_np2_m[j+1,i]*dt*( T_w[j,i] + D_z_w[j,i] )/2.
                
            elif w_np2_m[j,i] >= 0.0 and w_np2_m[j+1,i] <= 0.0:
#                Aquí solo hay contribución a la celda j
                S_pip[j,i] = S_pip[j,i] + (dz + ( w_np2_m[j+1,i] 
                            - w_np2_m[j,i] )*dt)*( I_z_pip[j,i] + D_z_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + (dz + ( w_np2_m[j+1,i] 
                            - w_np2_m[j,i] )*dt)*( I_z_thetap[j,i] + D_z_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + (dz+ ( w_np2_m[j+1,i] 
                            - w_np2_m[j,i] )*dt)*( I_z_u[j,i] + D_z_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + (dz + ( w_np2_m[j+1,i] 
                            - w_np2_m[j,i] )*dt)*( I_z_w[j,i] + D_z_w[j,i] )/2.

            elif w_np2_m[j,i] <= 0.0 and w_np2_m[j+1,i] >= 0.0:
                #Aquí hay contribución a las celdas j-1,j y j+1:
                T1_pip[j,i] = I_z_pip[j,i] + sigmazpip_np1[j,i]*abs(w_np2_m[j,i])*dt
                T1_thetap[j,i] = I_z_thetap[j,i] + sigmazthetap_np1[j,i]*abs(w_np2_m[j,i])*dt
                T1_u[j,i] = I_z_u[j,i] + sigmazu_np1[j,i]*abs(w_np2_m[j,i])*dt
                T1_w[j,i] = I_z_w[j,i] + sigmazw_np1[j,i]*abs(w_np2_m[j,i])*dt
                T2_pip[j,i] = T1_pip[j,i] + sigmazpip_np1[j,i]*dz
                T2_thetap[j,i] = T1_thetap[j,i] + sigmazthetap_np1[j,i]*dz
                T2_u[j,i] = T1_u[j,i] + sigmazu_np1[j,i]*dz
                T2_w[j,i] = T1_w[j,i] + sigmazw_np1[j,i]*dz
                
                S_pip[j-1,i] = S_pip[j-1,i] + abs(w_np2_m[j,i])*dt*( I_z_pip[j,i] + T1_pip[j,i] )/2.
                S_thetap[j-1,i] = S_thetap[j-1,i] + abs(w_np2_m[j,i])*dt*( I_z_thetap[j,i] + T1_thetap[j,i] )/2.
                S_u[j-1,i] = S_u[j-1,i] + abs(w_np2_m[j,i])*dt*( I_z_u[j,i] + T1_u[j,i] )/2.
                S_w[j-1,i] = S_w[j-1,i] + abs(w_np2_m[j,i])*dt*( I_z_w[j,i] + T1_w[j,i] )/2.
                
                S_pip[j,i] = S_pip[j,i] + dz*(T1_pip[j,i] + T2_pip[j,i])/2.
                S_thetap[j,i] = S_thetap[j,i] + dz*(T1_thetap[j,i] + T2_thetap[j,i])/2.
                S_u[j,i] = S_u[j,i] + dz*(T1_u[j,i] + T2_u[j,i])/2.
                S_w[j,i] = S_w[j,i] + dz*(T1_w[j,i] + T2_w[j,i])/2.
                
                S_pip[j+1,i] = S_pip[j+1,i] + w_np2_m[j+1,i]*dt*( T2_pip[j,i] + D_z_pip[j,i] )/2.
                S_thetap[j+1,i] = S_thetap[j+1,i] + w_np2_m[j+1,i]*dt*( T2_thetap[j,i] + D_z_thetap[j,i] )/2.
                S_u[j+1,i] = S_u[j+1,i] + w_np2_m[j+1,i]*dt*( T2_u[j,i] + D_z_u[j,i] )/2.
                S_w[j+1,i] = S_w[j+1,i] + w_np2_m[j+1,i]*dt*( T2_w[j,i] + D_z_w[j,i] )/2.
                                                
            elif w_np2_m[j,i] <= 0.0 and w_np2_m[j+1,i] <= 0.0:
                #Aqui hay contribución a las celdas j y j-1:
                T_pip[j,i] = I_z_pip[j,i] + sigmazpip_np1[j,i]*abs(w_np2_m[j,i])*dt
                T_thetap[j,i] = I_z_thetap[j,i] + sigmazthetap_np1[j,i]*abs(w_np2_m[j,i])*dt
                T_u[j,i] = I_z_u[j,i] + sigmazu_np1[j,i]*abs(w_np2_m[j,i])*dt
                T_w[j,i] = I_z_w[j,i] + sigmazw_np1[j,i]*abs(w_np2_m[j,i])*dt
                
                S_pip[j-1,i] = S_pip[j-1,i] + abs(w_np2_m[j,i])*dt*( T_pip[j,i] + I_z_pip[j,i] )/2.
                S_thetap[j-1,i] = S_thetap[j-1,i] + abs(w_np2_m[j,i])*dt*( T_thetap[j,i] + I_z_thetap[j,i] )/2.
                S_u[j-1,i] = S_u[j-1,i] + abs(w_np2_m[j,i])*dt*( T_u[j,i] + I_z_u[j,i] )/2.
                S_w[j-1,i] = S_w[j-1,i] + abs(w_np2_m[j,i])*dt*( T_w[j,i] + I_z_w[j,i] )/2.
                
                S_pip[j,i] = S_pip[j,i] + ( dz + w_np2_m[j+1,i]*dt )*( T_pip[j,i] + D_z_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + ( dz + w_np2_m[j+1,i]*dt )*( T_thetap[j,i] + D_z_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + ( dz + w_np2_m[j+1,i]*dt )*( T_u[j,i] + D_z_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + ( dz + w_np2_m[j+1,i]*dt )*( T_w[j,i] + D_z_w[j,i] )/2.
                                            
    #g) Los términos de advección vertical son:
    for i in range (0,I,1):
        for j in range (0,J,1):
            advpip[j,i] = S_pip[j,i]/dz
            advthetap[j,i] = S_thetap[j,i]/dz
            advu[j,i] = S_u[j,i]/dz
            advw[j,i] = S_w[j,i]/dz
            
    #PASO 3: calculamos la advección horizontal, usando los resultados ya calculados.
    #Ahora toca advectar horizontalmente las variables adv_variable:
    for i in range (1,I-1,1):
        for j in range (0,J,1):
            #a) Definimos limitador minmod para la pendiente de reconstrucción sigma
            #términos a,b de las funciones a advectar:
            #Para la advección horizontal:
            a_pip_x[j,i] = (advpip[j,i]-advpip[j,i-1])/dx            
            a_thetap_x[j,i] = (advthetap[j,i]-advthetap[j,i-1])/dx           
            a_u_x[j,i] = (advu[j,i]-advu[j,i-1])/dx            
            a_w_x[j,i] = (advw[j,i]-advw[j,i-1])/dx
            b_pip_x[j,i] = (advpip[j,i+1]-advpip[j,i])/dx
            b_thetap_x[j,i] = (advthetap[j,i+1]-advthetap[j,i])/dx
            b_u_x[j,i] = (advu[j,i+1]-advu[j,i])/dx
            b_w_x[j,i] = (advw[j,i+1]-advw[j,i])/dx

    for i in range (0,I,1):
        for j in range (0,J,1):
            #Minmod para la advección horizontal de pip:
            if  abs(a_pip_x[j,i]) < abs(b_pip_x[j,i]) and (a_pip_x[j,i])*(b_pip_x[j,i]) > 0.0:
                sigmaxpip[j,i] = a_pip_x[j,i]
            elif  abs(a_pip_x[j,i]) > abs(b_pip_x[j,i]) and (a_pip_x[j,i])*(b_pip_x[j,i]) > 0.0:
                sigmaxpip[j,i] = b_pip_x[j,i]
            else:
                sigmaxpip[j,i] = 0.0
                
            #Minmod para la advección horizontal de thetap:
            if  abs(a_thetap_x[j,i]) < abs(b_thetap_x[j,i])  and (a_thetap_x[j,i])*(b_thetap_x[j,i]) > 0.0:
                sigmaxthetap[j,i] = a_thetap_x[j,i]
            elif abs(a_thetap_x[j,i]) > abs(b_thetap_x[j,i]) and (a_thetap_x[j,i])*(b_thetap_x[j,i]) > 0.0:
                sigmaxthetap[j,i] = b_thetap_x[j,i]
            else:
                sigmaxthetap[j,i] = 0.0
                
            #Minmod para la advección horizontal de u:
            if abs(a_u_x[j,i]) < abs(b_u_x[j,i]) and (a_u_x[j,i])*(b_u_x[j,i]) > 0.0:
                sigmaxu[j,i] = a_u_x[j,i]
            elif abs(a_u_x[j,i]) > abs(b_u_x[j,i]) and (a_u_x[j,i])*b_u_x[j,i] > 0.0:
                sigmaxu[j,i] = b_u_x[j,i]
            else:
                sigmaxu[j,i] = 0.0
                
            #Minmod para la advección horizontal w:
            if abs(a_w_x[j,i]) < abs(b_w_x[j,i]) and (a_w_x[j,i])*(b_w_x[j,i]) > 0.0:
                sigmaxw[j,i] = a_w_x[j,i]
            elif abs(a_w_x[j,i]) > abs(b_w_x[j,i]) and (a_w_x[j,i])*(b_w_x[j,i]) > 0.0:
                sigmaxw[j,i] = b_w_x[j,i]
            else:
                sigmaxw[j,i] = 0.0
                
    #b) Calculamos los extremos D (Derecha) e I (Izquierda) de la pendiente
    #de reconstrucción:
            D_x_pip[j,i] = advpip[j,i] + sigmaxpip[j,i]*(dx/2) 
            I_x_pip[j,i] = advpip[j,i] - sigmaxpip[j,i]*(dx/2) 
            D_x_thetap[j,i] = advthetap[j,i] + sigmaxthetap[j,i]*(dx/2) 
            I_x_thetap[j,i] = advthetap[j,i] - sigmaxthetap[j,i]*(dx/2)
            D_x_u[j,i] = advu[j,i] + sigmaxu[j,i]*(dx/2) 
            I_x_u[j,i] = advu[j,i] - sigmaxu[j,i]*(dx/2)
            D_x_w[j,i] = advw[j,i] + sigmaxw[j,i]*(dx/2) 
            I_x_w[j,i] = advw[j,i] - sigmaxw[j,i]*(dx/2)
            
    #Calculamos el valor medio del viento correspondiente a cada celda:
    for i in range (1,I,1):
        for j in range (0,J,1):
            u_np2_m[j,i] = (advu[j,i] + advu[j,i-1])/2.
            
    #c) Calculamos la evolución de la pendiente. Para ello, calculamos la
    #distancia d_x que se ha avanzado en el proceso.
    d_t_i[:,0] = dx
    for i in range (1,I,1):
        for j in range (0,J,1):
            d_t_i[j,i] = dx + (u_np2_m[j,i] - u_np2_m[j,i-1])*dt
                           
    for i in range (0,I,1):
        for j in range (0,J,1):                                   
            sigmaxpip_np1[j,i] = ( D_x_pip[j,i] - I_x_pip[j,i] )/d_t_i[j,i]
            sigmaxthetap_np1[j,i] = ( D_x_thetap[j,i] - I_x_thetap[j,i] )/d_t_i[j,i]
            sigmaxu_np1[j,i] = ( D_x_u[j,i] - I_x_u[j,i] )/d_t_i[j,i]
            sigmaxw_np1[j,i] = ( D_x_w[j,i] - I_x_w[j,i] )/d_t_i[j,i]

    #f) Calculamos el área correspondiente a cada celda. El área total
    #comprende el término de advección que corresponda y depende muy 
    #fuertemente de los signos de las velocidades en dos puntos de malla
    #consecutivos. Inicializamos la integracion haciendo que las áreas sean 0
    #para después ir sumando. El resultado de la suma son nuestras variables
    #advectadas totalmente:
    
    #Inicializamos suma:
    for i in range (0,I,1):
        for j in range (0,J,1):
            S_pip[j,i] = 0.0 
            S_thetap[j,i] = 0.0 
            S_u[j,i] = 0.0
            S_w[j,i] = 0.0
            
    for i in range (1,I-1,1):
        for j in range (0,J,1):
            if u_np2_m[j,i] >= 0.0 and u_np2_m[j,i+1] >= 0.0:
                #Aquí hay contribución a las celdas i e i+1
                T_pip[j,i] = I_x_pip[j,i] + sigmaxpip_np1[j,i]*(dx - u_np2_m[j,i]*dt)
                T_thetap[j,i] = I_x_thetap[j,i] + sigmaxthetap_np1[j,i]*(dx - u_np2_m[j,i]*dt)
                T_u[j,i] = I_x_u[j,i] + sigmaxu_np1[j,i]*(dx - u_np2_m[j,i]*dt)
                T_w[j,i] = I_x_w[j,i] + sigmaxw_np1[j,i]*(dx - u_np2_m[j,i]*dt)

                S_pip[j,i] = S_pip[j,i] + (dx - u_np2_m[j,i]*dt)*( T_pip[j,i] + I_x_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + (dx - u_np2_m[j,i]*dt)*( T_thetap[j,i] + I_x_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + (dx - u_np2_m[j,i]*dt)*( T_u[j,i] + I_x_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + (dx - u_np2_m[j,i]*dt)*( T_w[j,i] + I_x_w[j,i] )/2.
                
                S_pip[j,i+1] = S_pip[j,i+1] + u_np2_m[j,i+1]*dt*( T_pip[j,i] + D_x_pip[j,i] )/2.
                S_thetap[j,i+1] = S_thetap[j,i+1] + u_np2_m[j,i+1]*dt*( T_thetap[j,i] + D_x_thetap[j,i] )/2.
                S_u[j,i+1] = S_u[j,i+1] + u_np2_m[j,i+1]*dt*( T_u[j,i] + D_x_u[j,i] )/2.
                S_w[j,i+1] = S_w[j,i+1] + u_np2_m[j,i+1]*dt*( T_w[j,i] + D_x_w[j,i] )/2.

            elif u_np2_m[j,i] >= 0.0 and u_np2_m[j,i+1] <= 0.0:
                #Aquí solo hay contribución a la celda i
                S_pip[j,i] = S_pip[j,i] + (dx + ( u_np2_m[j,i+1] 
                            - u_np2_m[j,i] )*dt)*( I_x_pip[j,i] + D_x_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + (dx + ( u_np2_m[j,i+1] 
                            - u_np2_m[j,i] )*dt)*( I_x_thetap[j,i] + D_x_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + (dx + ( u_np2_m[j,i+1] 
                            - u_np2_m[j,i] )*dt)*( I_x_u[j,i] + D_x_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + (dx+ ( u_np2_m[j,i+1] 
                            - u_np2_m[j,i] )*dt)*( I_x_w[j,i] + D_x_w[j,i] )/2.
                
            elif u_np2_m[j,i] <= 0.0 and u_np2_m[j,i+1] >= 0.0:
                #Aquí hay contribución a las celdas i e i+1:
                T1_pip[j,i] = I_x_pip[j,i] + sigmaxpip_np1[j,i]*abs(u_np2_m[j,i])*dt
                T1_thetap[j,i] = I_x_thetap[j,i] + sigmaxthetap_np1[j,i]*abs(u_np2_m[j,i])*dt
                T1_u[j,i] = I_x_u[j,i] + sigmaxu_np1[j,i]*abs(u_np2_m[j,i])*dt
                T1_w[j,i] = I_x_w[j,i] + sigmaxw_np1[j,i]*abs(u_np2_m[j,i])*dt                
                T2_pip[j,i] = T1_pip[j,i] + sigmaxpip_np1[j,i]*dx
                T2_thetap[j,i] = T1_thetap[j,i] + sigmaxthetap_np1[j,i]*dx
                T2_u[j,i] = T1_u[j,i] + sigmaxu_np1[j,i]*dx
                T2_w[j,i] = T1_w[j,i] + sigmaxw_np1[j,i]*dx
                
                S_pip[j,i-1] = S_pip[j,i-1] + abs(u_np2_m[j,i])*dt*( I_x_pip[j,i] + T1_pip[j,i] )/2.
                S_thetap[j,i-1] = S_thetap[j,i-1] + abs(u_np2_m[j,i])*dt*( I_x_thetap[j,i] + T1_thetap[j,i] )/2.
                S_u[j,i-1] = S_u[j,i-1] + abs(u_np2_m[j,i])*dt*( I_x_u[j,i] + T1_u[j,i] )/2.
                S_w[j,i-1] = S_w[j,i-1] + abs(u_np2_m[j,i])*dt*( I_x_w[j,i] + T1_w[j,i] )/2.
                
                S_pip[j,i] = S_pip[j,i] + dx*(T1_pip[j,i] + T2_pip[j,i])/2.
                S_thetap[j,i] = S_thetap[j,i] + dx*(T1_thetap[j,i] + T2_thetap[j,i])/2.
                S_u[j,i] = S_u[j,i] + dx*(T1_u[j,i] + T2_u[j,i])/2.
                S_w[j,i] = S_w[j,i] + dx*(T1_w[j,i] + T2_w[j,i])/2.
                
                S_pip[j,i+1] = S_pip[j,i+1] + u_np2_m[j,i+1]*dt*( T2_pip[j,i] + D_x_pip[j,i] )/2.
                S_thetap[j,i+1] = S_thetap[j,i+1] + u_np2_m[j,i+1]*dt*( T2_thetap[j,i] + D_x_thetap[j,i] )/2.
                S_u[j,i+1] = S_u[j,i+1] + u_np2_m[j,i+1]*dt*( T2_u[j,i] + D_x_u[j,i] )/2.
                S_w[j,i+1] = S_w[j,i+1] + u_np2_m[j,i+1]*dt*( T2_w[j,i] + D_x_w[j,i] )/2.
                                                
            elif u_np2_m[j,i] <= 0.0 and u_np2_m[j,i+1] <= 0.0:
                #Aqui hay contribución a las celdas i e i-1:
                T_pip[j,i] = I_x_pip[j,i] + sigmaxpip_np1[j,i]*abs(u_np2_m[j,i])*dt
                T_thetap[j,i] = I_x_thetap[j,i] + sigmaxthetap_np1[j,i]*abs(u_np2_m[j,i])*dt
                T_u[j,i] = I_x_u[j,i] + sigmaxu_np1[j,i]*abs(u_np2_m[j,i])*dt
                T_w[j,i] = I_x_w[j,i] + sigmaxw_np1[j,i]*abs(u_np2_m[j,i])*dt
                
                S_pip[j,i-1] = S_pip[j,i-1] + abs(u_np2_m[j,i])*dt*( T_pip[j,i] + I_x_pip[j,i] )/2.
                S_thetap[j,i-1] = S_thetap[j,i-1] + abs(u_np2_m[j,i])*dt*( T_thetap[j,i] + I_x_thetap[j,i] )/2.
                S_u[j,i-1] = S_u[j,i-1] + abs(u_np2_m[j,i])*dt*( T_u[j,i] + I_x_u[j,i] )/2.
                S_w[j,i-1] = S_w[j,i-1] + abs(u_np2_m[j,i])*dt*( T_w[j,i] + I_x_w[j,i] )/2.
                
                S_pip[j,i] = S_pip[j,i] + ( dx + u_np2_m[j,i+1]*dt )*( T_pip[j,i] + D_x_pip[j,i] )/2.
                S_thetap[j,i] = S_thetap[j,i] + ( dx + u_np2_m[j,i+1]*dt )*( T_thetap[j,i] + D_x_thetap[j,i] )/2.
                S_u[j,i] = S_u[j,i] + ( dx + u_np2_m[j,i+1]*dt )*( T_u[j,i] + D_x_u[j,i] )/2.
                S_w[j,i] = S_w[j,i] + ( dx + u_np2_m[j,i+1]*dt )*( T_w[j,i] + D_x_w[j,i] )/2.

    #g) Las variables advectadas son:
    for i in range (0,I,1):
        for j in range (0,J,1):
            pip[j,i] = S_pip[j,i]/dx
            thetap[j,i] = S_thetap[j,i]/dx
            u[j,i] = S_u[j,i]/dx
            w[j,i] = S_w[j,i]/dx
                  
    #Aplicamos Euler-Forward:
    pip_np0 = pip
    thetap_np0 = thetap
    u_np0 = u
    w_np0 = w
    
    #Condición de flujo nulo en las paredes horizontales y verticales del dominio
    #de las funciones que se van a integrar. Esto implica que la velocidad vertical
    #en las paredes horizontales y la velocidad zonal en las paredes verticales
    #tienen que ser nulas para cualquier valor del punto de malla. Además, el
    #resto de funciones tienen condición de libre deslizamiento, es decir, que
    #adquieren el mismo valor en los contornos en instantes consecutivos de tiempo:
    
    #Paredes horizontales
    #Pared izquierda:
    #Sería lógico poner la condición de contorno en [:,0] pero como hemos calcu-
    #lado la advección desde 1 a I-1 debemos forzar la condición en estos 2
    #puntos. En el resto de casos, como las cc son de flujo nulo, se puede poner
    #las cc en 0 directamente, pero aquí tenemos un viento horizontal no nulo.
    #Si no definimos bien los contornos quedarían celdas que se quedarían vacías
    #y no se aplicaría el REA. De todas formas conviene ser consistente con la
    #notación.
    pip[:,1] = pip[:,3] 
    thetap[:,1] = thetap[:,3]
    u[:,1] = -u[:,3]
    w[:,1] = w[:,3]
    pip[:,2] = pip[:,3]
    thetap[:,2] = thetap[:,3]
    u[:,2] = np.zeros([J])
    w[:,2] = w[:,3]
    
    #Pared derecha:
    pip[:,I-1] = pip[:,I-3] 
    thetap[:,I-1] = thetap[:,I-3]
    u[:,I-1] = -u[:,I-3]
    w[:,I-1] = w[:,I-3]
    pip[:,I-2] = pip[:,I-3]
    thetap[:,I-2] = thetap[:,I-3]
    u[:,I-2] = np.zeros([J])
    w[:,I-2] = w[:,I-3]
    
    #Paredes verticales
    #Pared inferior:
    pip[1,:] = pip[3,:]
    thetap[1,:] = thetap[3,:]
    u[1,:] = u[3,:]
    w[1,:] = -w[3,:] 
    pip[2,:] = pip[3,:]
    thetap[2,:] = thetap[3,:]
    u[2,:] = u[3,:]
    w[2,:] = np.zeros([I])
    
    #Pared superior:
    pip[J-1,:] = pip[J-3,:]
    thetap[J-1,:] = thetap[J-3,:]
    u[J-1,:] = u[J-3,:]
    w[J-1,:] = -w[J-3,:] 
    pip[J-2,:] = pip[J-3,:]
    thetap[J-2,:] = thetap[J-3,:]
    u[J-2,:] = u[J-3,:]
    w[J-2,:] = np.zeros([I])
    
    #Cuando el contador llega al instante que toca, se hace una representación
    #gráfica de las funciones que queramos, además de generar un archivo de texto
    #donde se guarda la matriz numérica de las variables deseadas:
    for m in range (1,M+1,1):
        m=m-1
        if n==160*m:
            fig = plt.figure()
            fig, axes = plt.subplots(figsize=(15,3))#Controla el tamaño de la figura 
            #en figsize=('tamaño eje x en pulgadas', 'tamaño eje z en pulgadas') y píxels de
            #la imagen dpi (dots per inch).
            plt.contourf(xx2,zz2,thetap,vmin=-30.0,vmax=0.0,levels=[-30.0,-27.0,-24.0,-21.0,-18.0,
                                                        -15.0,-12.0,-9.0,-6.0,-3.0,0.0])
            plt.title(r"$\theta' \ (K) $")
            plt.xlabel(r"$ x \ (m)$")
            plt.ylabel(r"$ z \ (m)$")
            plt.ylim([0,5000])
            plt.xlim([0,19200])
            plt.colorbar(ticks=[-30.0,-27.0,-24.0,-21.0,-18.0,
                                                        -15.0,-12.0,-9.0,-6.0,-3.0,0.0])
            plt.show()
            fig.savefig(q[m-1] + '_thetap.png',dpi=250)
            #Creamos archivo de texto y guardamos valores numéricos:
            np.savetxt("test.txt", thetap[:,:],fmt='%2.4f')#controla formato
                       
#Pintamos todas las funciones.Plots de prueba: 
fig = plt.figure()
fig, axes = plt.subplots(figsize=(15,3))           
plt.contourf(xx2,zz2,pip)
plt.xlabel(r"$x \ (m)$")
plt.ylabel(r"$z \ (m)$")
plt.title(r"$\pi'$")
plt.ylim([0,5000])
plt.xlim([0,19200])
plt.colorbar()
plt.show()

fig = plt.figure()
fig, axes = plt.subplots(figsize=(15,3))           
plt.contourf(xx2,zz2,thetap,vmin=-30.0,vmax=0.0,levels=[-30.0,-27.0,-24.0,-21.0,-18.0,
                                                        -15.0,-12.0,-9.0,-6.0,-3.0,0.0])
plt.xlabel(r"$x \ (m)$")
plt.ylabel(r"$z \ (m)$")
plt.title(r"$\theta' \ (K) $")
plt.ylim([0,5000])
plt.xlim([0,19200])
plt.colorbar(ticks=[-30.0,-27.0,-24.0,-21.0,-18.0,
                                                        -15.0,-12.0,-9.0,-6.0,-3.0,0.0])
plt.show()
fig.savefig("900s_thetap.png",dpi=250)

fig = plt.figure()
fig, axes = plt.subplots(figsize=(15,3))
plt.contourf(xx2,zz2,u)
plt.xlabel(r"$x \ (m)$")
plt.ylabel(r"$z \ (m)$")
plt.title(r"$ u \ (m/s) $")
plt.ylim([0,5000])
plt.xlim([0,19200])
plt.colorbar()
plt.show()

fig = plt.figure()
fig, axes = plt.subplots(figsize=(15,3))
plt.contourf(xx2,zz2,w)
plt.xlabel(r"$x \ (m)$")
plt.ylabel(r"$z \ (m)$")
plt.title(r"$ w \ (m/s) $")
plt.ylim([0,5000])
plt.xlim([0,19200])
plt.colorbar()
plt.show()